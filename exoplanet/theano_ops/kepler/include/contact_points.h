#ifndef _CONTACT_POINTS_H_
#define _CONTACT_POINTS_H_

#include <vector>
#include <cmath>
#include <algorithm>

namespace contact_points {
template <typename T>
class ContactPoint {
  private:
    T e, sinw, cosw, cosi2, factor, b2_0;

    inline T b2 (T delta) {
      T sd = sin(delta);
      T cd = cos(delta);
      T sdmw = cosw * sd - sinw * cd;  // sin(delta - w)
      T denom = e * sdmw + 1;
      return factor * (sd*sd + cd*cd*cosi2) / (denom*denom);
    }

    inline T find_root (T R2, T min, T max, bool flag, int maxiter, T tol) {
      T f_mid, mid = 0.5 * (min + max);
      for (int n = 0; n < maxiter; ++n) {
        mid = 0.5 * (min + max);
        f_mid = b2(mid);
        if (std::abs(f_mid - R2) < tol || 0.5 * (max - min) < tol) return mid;
        if ((f_mid > R2 && flag) || (f_mid < R2 && !flag)) {
          min = mid;
        } else {
          max = mid;
        }
      }
      return mid;
    }

  public:

    ContactPoint (T a, T e, T w, T i) : e(e) {
      sinw = sin(w);
      cosw = cos(w);
      cosi2 = pow(cos(i), 2);

      T e2 = e*e;
      factor = a * (e2 - 1);
      factor *= factor;

      T arg = e * sinw - 1;
      b2_0 = factor * cosi2 / (arg * arg);
    }

    int find_contacts (T R, T r, T* f1, T* f2, T* f3, T* f4, int maxiter, T tol) {
      T minus = -0.5 * M_PI;
      T plus = 0.5 * M_PI;
      T b2_minus = b2(minus);
      T b2_plus = b2(plus);
      T R2;

      R2 = (R + r) * (R + r);
      if (R2 >= b2_minus || R2 <= b2_0) {
        *f1 = minus;
      } else {
        *f1 = find_root(R2, minus, 0, true, maxiter, tol);
      }
      if (R2 >= b2_plus || R2 <= b2_0) {
        *f4 = plus;
      } else {
        *f4 = find_root(R2, 0, plus, false, maxiter, tol);
      }

      R2 = (R - r) * (R - r);
      if (R2 >= b2_minus || R2 <= b2_0) {
        *f2 = 0;
      } else {
        *f2 = find_root(R2, *f1, 0, true, maxiter, tol);
      }
      if (R2 >= b2_plus || R2 <= b2_0) {
        *f3 = 0;
      } else {
        *f3 = find_root(R2, 0, *f4, false, maxiter, tol);
      }

      return 0;
    }
};

}  // namespace contact_points

#endif
