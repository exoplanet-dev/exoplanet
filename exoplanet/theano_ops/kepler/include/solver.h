#ifndef _EXOPLANET_KEPLER_SOLVER_H_
#define _EXOPLANET_KEPLER_SOLVER_H_

// A solver for Kepler's equation based on: http://adsabs.harvard.edu/abs/1991CeMDA..51..319N
// Nijenhuis (1991)

#include <cmath>

namespace exoplanet {

  template <typename T>
  inline T get_starter (T M, T e) {

    const T mikkoleft = 0.45;
    const T mikkolup = 1 - mikkoleft;
    const T s3 = -0.16605;
    const T s5 = 0.00761;
    const T ds2 = 3 * s3;
    const T ds4 = 5 * s5;

    if (M <= 0) return T(0);

    T E;
    T M1 = M + e;
    T e1 = 1 - e;
    bool mikkola = false;
    if (M1 > M_PI - 1) {
      E = (M + e * M_PI) / (1 + e);
    } else if (M1 > 1) {
      if (M > mikkoleft) {
        E = M1;
      } else {
        mikkola = true;
      }
    } else if (e < mikkolup) {
      E = M / e1;
    } else {
      mikkola = true;
    }

    if (mikkola) {
      T den = 0.5 + 4 * e;
      T p = e1 / den;
      T q = 0.5 * M / den;
      T p2 = p*p;
      T zsq = exp(log(sqrt(p2*p + q*q) + q) / 1.5);
      T s = 2 * q / (zsq + p + p2 / zsq);
      T ssq = s*s;
      T sqq = ssq * ssq;
      s -= 0.075 * s * sqq / (e1 + den*ssq + 0.375*sqq);
      E = M + e * s * (3 - 4 * s*s);
    } else {
      bool big = E > M_PI_2;
      T x = (big) ? M_PI - E : E;
      T xsq = x*x;
      T sn = x * (1 + xsq * (s3 + xsq * s5));
      T dsn = 1 + xsq * (ds2 + xsq * ds4);
      if (big) dsn = -dsn;
      T f2 = e * sn;
      T f0 = E - f2 - M;
      T f1 = 1 - e * dsn;
      E -= f0 / (f1 - 0.5 * f0 * f2 / f1);
    }

    return E;
  }

  // Calculates x - sin(x) and 1 - cos(x) to 20 significant digits for x in [0, pi)
  template <typename T>
  inline void sin_cos_reduc (T x, T* SnReduc, T* CsReduc) {
    const T s[] = {T(1)/6, T(1)/20, T(1)/42, T(1)/72, T(1)/110, T(1)/156, T(1)/210, T(1)/272, T(1)/342, T(1)/420};
    const T c[] = {T(0.5), T(1)/12, T(1)/30, T(1)/56, T(1)/90, T(1)/132, T(1)/182, T(1)/240, T(1)/306, T(1)/380};

    bool bigg = x > M_PI_2;
    T u = (bigg) ? M_PI - x : x;
    bool big = u > M_PI_2;
    T v = (big) ? M_PI_2 - u : u;
    T w = v * v;

    T ss = T(1);
    T cc = T(1);
    for (int i = 9; i >= 1; --i) {
      ss = 1 - w * s[i] * ss;
      cc = 1 - w * c[i] * cc;
    }
    ss *= v * w * s[0];
    cc *= w * c[0];

    if (big) {
      *SnReduc = u - 1 + cc;
      *CsReduc = 1 - M_PI_2 + u + ss;
    } else {
      *SnReduc = ss;
      *CsReduc = cc;
    }
    if (bigg) {
      *SnReduc = 2 * x - M_PI + *SnReduc;
      *CsReduc = 2 - *CsReduc;
    }
  }

  template <typename T>
  inline T refine (T E, T M, T e, int order) {
    T e1 = 1 - e;
    T snr, csr;
    sin_cos_reduc(E, &snr, &csr);

    T F[] = {
      E * e1 + e * snr - M,
      e1 + e * csr,
      0.5 * e * (E - snr),
      e * (1 - csr) / 6, 0};
    F[4] = -F[2] / 12;
    T h[4];

    T delta;
    for (int i = 1; i <= order; ++i) {
      delta = F[i];
      for (int j = 1; j <= i-1; ++j) {
        delta = delta * h[j-1] + F[i-j];
      }
      h[i-1] = -F[0] / delta;
    }
    return E + h[order - 1];
  }

  template <typename T>
  inline T solve_kepler (T M, T e) {
    const T two_pi = 2 * M_PI;

    T M_ref = two_pi * floor(M / two_pi);
    M -= M_ref;

    bool high = M > M_PI;
    if (high) {
      M = two_pi - M;
    }

    // Initialize
    T E0 = get_starter<T>(M, e);

    // Refine the estimate
    T E = refine<T>(E0, M, e, 4);

    if (high) {
      E = two_pi - E;
    }

    return E + M_ref;
  }

}

#endif
