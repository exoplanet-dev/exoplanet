#ifndef _VICE_INTEGRATE_H_
#define _VICE_INTEGRATE_H_

#include <cmath>
#include <tuple>
#include <algorithm>

namespace vice {
  namespace integrate {

  using std::abs;

  struct simpson_adapt {

    template <typename XScalar, typename YScalar, typename Functor>
    inline YScalar inner (Functor& func, XScalar x0, XScalar dx, YScalar ym, YScalar y0, YScalar yp, XScalar tol, unsigned max_depth, unsigned min_depth, unsigned depth) {
      XScalar x_m = x0 - 0.5*dx;
      YScalar val_m = func(x_m);
      YScalar int_m = dx * (4*val_m + ym + y0) / 6;

      XScalar x_p = x0 + 0.5*dx;
      YScalar val_p = func(x_p);
      YScalar int_p = dx * (4*val_p + yp + y0) / 6;

      YScalar pred = dx * (4*y0 + ym + yp) / 3;

      if (depth < min_depth || (depth < max_depth && abs(pred - (int_m + int_p)) > tol)) {
        int_m = inner<XScalar, YScalar, Functor>(func, x_m, 0.5*dx, ym, val_m, y0, tol, max_depth, min_depth, depth+1);
        int_p = inner<XScalar, YScalar, Functor>(func, x_p, 0.5*dx, y0, val_p, yp, tol, max_depth, min_depth, depth+1);
      }

      return int_m + int_p;
    }

    template <typename XScalar, typename YScalar, typename Functor>
    inline YScalar operator() (Functor& func, XScalar lower, XScalar upper, XScalar tol, unsigned max_depth=50, unsigned min_depth=0) {
      YScalar ym = func(lower);
      YScalar yp = func(upper);
      return this->operator()(func, lower, upper, ym, yp, tol, max_depth, min_depth);
    }

    template <typename XScalar, typename YScalar, typename Functor>
    inline YScalar operator() (Functor& func, XScalar lower, XScalar upper, YScalar ym, YScalar yp, XScalar tol, unsigned max_depth=50, unsigned min_depth=0) {
      XScalar x0 = 0.5 * (upper + lower);
      XScalar dx = 0.5 * (upper - lower);
      YScalar y0 = func(x0);

      YScalar int_trap = 0.5 * dx * (ym + 2 * y0 + yp);
      YScalar int_simp = dx * (ym + 4 * y0 + yp) / 3;

      if (min_depth == 0 && abs(int_trap - int_simp) < tol) {
        return int_simp;
      }

      return inner<XScalar, YScalar, Functor> (func, x0, dx, ym, y0, yp, tol, max_depth, min_depth, 0);
    }

  };

  }  // namespace integrate
}    // namespace vice

#endif  // _VICE_INTEGRATE_H_
