#ifndef _EXOPLANET_KEPLER_SOLVER_H_
#define _EXOPLANET_KEPLER_SOLVER_H_

// A solver for Kepler's equation based on:
//
// Nijenhuis (1991)
// http://adsabs.harvard.edu/abs/1991CeMDA..51..319N
//
// and
//
// Markley (1995)
// http://adsabs.harvard.edu/abs/1995CeMDA..63..101M

#include <cmath>

namespace exoplanet {

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

  const double FACTOR1 = 3*M_PI / (M_PI - 6/M_PI);
  const double FACTOR2 = 1.6 / (M_PI - 6/M_PI);

  template <typename T>
  inline T solve_kepler (T M, T ecc) {
    const T two_pi = 2 * M_PI;

    T M_ref = two_pi * floor(M / two_pi);
    M -= M_ref;

    bool high = M > M_PI;
    if (high) {
      M = two_pi - M;
    }

    T ome = 1.0 - ecc;

    // Get starter
    T M2 = M*M;
    T M3 = M2*M;
    T alpha = FACTOR1 + FACTOR2*(M_PI-std::abs(M))/(1+ecc);
    T d = 3*ome + alpha*ecc;
    T r = 3*alpha*d*(d-ome)*M + M3;
    T q = 2*alpha*d*ome - M2;
    T q2 = q*q;
    T w = pow(std::abs(r) + sqrt(q2*q + r*r), 2.0/3);
    T E = (2*r*w/(w*w + w*q + q2) + M) / d;

    // Approximate Mstar = E - e*sin(E) with numerically stability
    T sE, cE;
    sin_cos_reduc (E, &sE, &cE);

    // Refine the starter
    T f_0 = ecc * sE + E * ome - M;
    T f_1 = ecc * cE + ome;
    T f_2 = ecc * (E - sE);
    T f_3 = 1-f_1;
    T d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1);
    T d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3)*f_3/6);
    T d_42 = d_4*d_4;
    E -= f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_42*d_4*f_2/24);

    if (high) {
      E = two_pi - E;
    }

    return E + M_ref;
  }

}

#endif
