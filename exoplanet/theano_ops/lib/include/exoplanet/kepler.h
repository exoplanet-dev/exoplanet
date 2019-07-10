#ifndef _EXOPLANET_KEPLER_H_
#define _EXOPLANET_KEPLER_H_

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
namespace kepler {

    const double s[] = {1.0/6, 1.0/20, 1.0/42, 1.0/72, 1.0/110, 1.0/156, 1.0/210, 1.0/272, 1.0/342, 1.0/420};
    const double c[] = {0.5, 1.0/12, 1.0/30, 1.0/56, 1.0/90, 1.0/132, 1.0/182, 1.0/240, 1.0/306, 1.0/380};

  // Calculates x - sin(x) and 1 - cos(x) to 20 significant digits for x in [0, pi)
  template <typename T>
  inline void sin_cos_reduc (T x, T* SnReduc, T* CsReduc) {

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

  // Implementation from numpy
  template <typename T>
  inline T npy_mod (T a, T b) {
    T mod = fmod(a, b);

    if (!b) {
      // If b == 0, return result of fmod. For IEEE is nan
      return mod;
    }

    // adjust fmod result to conform to Python convention of remainder
    if (mod) {
      if ((b < 0) != (mod < 0)) {
        mod += b;
      }
    } else {
      // if mod is zero ensure correct sign
      mod = copysign(0, b);
    }

    return mod;
  }

  const double FACTOR1 = 3*M_PI / (M_PI - 6/M_PI);
  const double FACTOR2 = 1.6 / (M_PI - 6/M_PI);
  const double two_pi = 2 * M_PI;

  template <typename T>
  inline T solve_kepler (T M, T ecc) {

    M = npy_mod(M, T(two_pi));

    bool high = M > M_PI;
    if (high) {
      M = two_pi - M;
    }

    T ome = 1.0 - ecc;

    // Get starter
    T M2 = M*M;
    T alpha = FACTOR1 + FACTOR2*(M_PI-M)/(1+ecc);
    T d = 3*ome + alpha*ecc;
    T alphad = alpha*d;
    T r = (3*alphad*(d-ome) + M2) * M;
    T q = 2*alphad*ome - M2;
    T q2 = q*q;
    T w = pow(std::abs(r) + sqrt(q2*q + r*r), 2.0/3);
    T E = (2*r*w/(w*w + w*q + q2) + M) / d;

    // Approximate Mstar = E - e*sin(E) with numerically stability
    T sE, cE;
    sin_cos_reduc(E, &sE, &cE);

    // Refine the starter
    T f_0 = ecc * sE + E * ome - M;
    T f_1 = ecc * cE + ome;
    T f_2 = ecc * (E - sE);
    T f_3 = 1-f_1;
    T d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1);
    T d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3)*f_3/6);
    T d_42 = d_4*d_4;
    T dE = -f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_42*d_4*f_2/24);
    E += dE;

    if (high) {
      E = two_pi - E;
    }

    return E;  // + M_ref;
  }

}  // namespace kepler
}  // namespace exoplanet

#endif
