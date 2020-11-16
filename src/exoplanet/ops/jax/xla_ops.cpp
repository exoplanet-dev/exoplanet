#include <exoplanet/calcEA.h>
#include <pybind11/pybind11.h>

#include <cmath>

namespace py = pybind11;

namespace xla_ops {
void kepler(void *out_tuple, const void **in) {
  void **out = reinterpret_cast<void **>(out_tuple);
  double *sinf = reinterpret_cast<double *>(out[0]);
  double *cosf = reinterpret_cast<double *>(out[1]);

  const int N = *reinterpret_cast<const int *>(in[0]);
  const double *M = reinterpret_cast<const double *>(in[1]);
  const double *ecc = reinterpret_cast<const double *>(in[2]);

  const double nan = std::nan("");

  for (int n = 0; n < N; ++n) {
    if (ecc[n] < 0 || ecc[n] > 1) {
      sinf[n] = nan;
      cosf[n] = nan;
    } else {
      exoplanet::calcEA::solve_kepler(M[n], ecc[n], &(cosf[n]), &(sinf[n]));
    }
  }
}
}  // namespace xla_ops

PYBIND11_MODULE(xla_ops, m) {
  m.def("kepler", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&xla_ops::kepler, name);
  });
}
