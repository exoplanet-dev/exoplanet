/*
<%
setup_pybind11(cfg)
cfg['include_dirs'] = ['../../src/exoplanet/theano_ops/lib/include/exoplanet']
%>
*/
#include <pybind11/pybind11.h>

#include <sys/time.h>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "kepler.h"

namespace py = pybind11;

double get_timestamp() {
  struct timeval now;
  gettimeofday(&now, NULL);
  return double(now.tv_usec) * 1.0e-6 + double(now.tv_sec);
}

int sign(int x) { return (x > 0) - (x < 0); }

// Ref:
// https://github.com/California-Planet-Search/radvel/blob/master/src/kepler.c
inline double kepler_radvel(double M, double e) {
  int MAX_ITER = 30;
  double CONV_TOL = 1.0e-12;  // convergence criterion

  double k, E, fi, d1, fip, fipp, fippp;
  int count;
  k = 0.85;   // initial guess at input parameter
  count = 0;  // how many loops have we done?

  E = M + sign(sin(M)) * k * e;  // first guess at E, the eccentric anomaly

  // E - e * sin(E) - M should go to 0
  fi = (E - e * sin(E) - M);
  while (fabs(fi) > CONV_TOL && count < MAX_ITER) {
    count++;

    // first, second, and third order derivatives of fi with respect to E
    fip = 1 - e * cos(E);
    fipp = e * sin(E);
    fippp = 1 - fip;

    // first, second, and third order corrections to E
    d1 = -fi / fip;
    d1 = -fi / (fip + d1 * fipp / 2.0);
    d1 = -fi / (fip + d1 * fipp / 2.0 + d1 * d1 * fippp / 6.0);
    E += d1;

    fi = (E - e * sin(E) - M);
    // printf("E =  %f, count = %i\n", E , count); //debugging

    if (count == MAX_ITER) {
      // printf("Error: kepler step not converging after %d steps.\n",
      // MAX_ITER);
      // printf("E=%f,  M=%f,  e=%f\n", E, M, e);
      return E;
    }
  }
  return E;
}

// Ref: https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c
inline double kepler_batman(double M, double e)  // calculates the eccentric
                                                 // anomaly (see Seager
                                                 // Exoplanets book:  Murray &
                                                 // Correia eqn. 5 -- see
                                                 // section 3)
{
  double E = M, eps = 1.0e-7;
  double fe, fs;
  const int max_iter = 30;
  int i = 0;

  // modification from LK 05/07/2017:
  // add fmod to ensure convergence for diabolical inputs (following Eastman et
  // al. 2013; Section 3.1)
  while (fmod(fabs(E - e * sin(E) - M), 2. * M_PI) > eps && i < max_iter) {
    fe = fmod(E - e * sin(E) - M, 2. * M_PI);
    fs = fmod(1 - e * cos(E), 2. * M_PI);
    E = E - fe / fs;
    ++i;
  }
  return E;
}

struct Exoplanet {
  inline double operator()(double M, double ecc, double *sE, double *cE) {
    double E = exoplanet::kepler::solve_kepler<double>(M, ecc);
    *sE = sin(E);
    *cE = cos(E);
    return E;
  }
};

struct Radvel {
  inline double operator()(double M, double ecc, double *sE, double *cE) {
    double E = kepler_radvel(M, ecc);
    *sE = sin(E);
    *cE = cos(E);
    return E;
  }
};

struct Batman {
  inline double operator()(double M, double ecc, double *sE, double *cE) {
    double E = kepler_batman(M, ecc);
    *sE = sin(E);
    *cE = cos(E);
    return E;
  }
};

template <typename Operator>
std::tuple<double, double, double, double> do_benchmark(double ecc, const int N) {
  std::vector<double> M(N), E(N);
  for (int n = 0; n < N; ++n) {
    E[n] = 2 * M_PI * n / (N - 1);
    M[n] = E[n] - ecc * sin(E[n]);
  }

  std::vector<double> error(N);
  double start = get_timestamp();
  Operator func;
  double sE, cE;
  for (int n = 0; n < N; ++n) {
    func(M[n], ecc, &sE, &cE);
    error[n] = std::abs(sin(E[n]) - sE);
  }
  double end = get_timestamp();
  double max_err = 0.0;
  double mean_err = 0.0;
  for (int n = 0; n < N; ++n) {
    mean_err += error[n] / N;
    if (error[n] > max_err) {
      max_err = error[n];
    }
  }

  std::sort(error.begin(), error.end());
  double pct90 = error[(9 * N) / 10];

  return std::make_tuple(log10(end - start) - log10(N), log10(max_err), log10(mean_err),
                         log10(pct90));
}

PYBIND11_MODULE(kepler_bench, m) {
  const int N = 1000000;
  m.def("exoplanet", &do_benchmark<Exoplanet>, py::arg("ecc"), py::arg("N") = N);
  m.def("radvel", &do_benchmark<Radvel>, py::arg("ecc"), py::arg("N") = N);
  m.def("batman", &do_benchmark<Batman>, py::arg("ecc"), py::arg("N") = N);
}
