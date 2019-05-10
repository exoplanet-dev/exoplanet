#ifndef _VICE_INTEGRATE_H_
#define _VICE_INTEGRATE_H_

#include <cmath>
#include <tuple>
#include <algorithm>

namespace vice {
  namespace integrate {

    using std::abs;

  struct riemann {

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, unsigned points) {
      Scalar nothing = 0;
      return this->operator()(func, lower, upper, nothing, nothing, points);
    }

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, Scalar ym, Scalar yp, unsigned points) {
      points = std::max<unsigned>(1, points);

      Scalar dx = (upper - lower) / points;
      Scalar x = lower + 0.5 * dx;
      Scalar f = Scalar(0);
      for (unsigned i = 0; i < points; ++i) {
        f += func(x);
        x += dx;
      }

      return dx * f;
    }

  };

  struct trapezoid_adapt {

    template <typename Scalar, typename Functor>
    inline Scalar inner (Functor& func, Scalar x1, Scalar x2, Scalar y1, Scalar y2, Scalar tol, unsigned max_depth, unsigned depth) {
      Scalar x0 = 0.5 * (x1 + x2);
      Scalar val = func(x0);
      Scalar dx = (x2 - x1);
      Scalar pred = 0.5 * dx * (y1 + y2);
      Scalar integral = 0.25 * dx * (y1 + 2*val + y2);

      if (depth < max_depth && std::abs(pred - integral) > tol) {
        integral = inner<Scalar, Functor>(func, x1, x0, y1, val, tol, max_depth, depth+1);
        integral += inner<Scalar, Functor>(func, x0, x2, val, y2, tol, max_depth, depth+1);
      }

      return integral;
    }

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, Scalar tol, unsigned max_depth=50) {
      Scalar ym = func(lower);
      Scalar yp = func(upper);
      return this->operator()(func, lower, upper, ym, yp, tol, max_depth);
    }

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, Scalar ym, Scalar yp, Scalar tol, unsigned max_depth=50) {
      return inner<Scalar, Functor> (func, lower, upper, ym, yp, tol, max_depth, 0);
    }

  };

  struct trapezoid_fixed {

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, unsigned points) {
      Scalar ym = func(lower);
      Scalar yp = func(upper);
      return this->operator()(func, lower, upper, ym, yp, points);
    }

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, Scalar ym, Scalar yp, unsigned points) {
      points = std::max<unsigned>(2, points);

      Scalar dx = (upper - lower) / (points - 1);
      Scalar x = lower + dx;
      Scalar f = 0.5 * ym;
      for (unsigned i = 1; i < points - 1; ++i) {
        f += func(x);
        x += dx;
      }
      f += 0.5 * yp;

      return dx * f;
    }

  };

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

  struct simpson_fixed {

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, unsigned points) {
      Scalar ym = func(lower);
      Scalar yp = func(upper);
      return this->operator()(func, lower, upper, ym, yp, points);
    }

    template <typename Scalar, typename Functor>
    inline Scalar operator() (Functor& func, Scalar lower, Scalar upper, Scalar ym, Scalar yp, unsigned points) {
      points = std::max<unsigned>(3, points + (points + 1) % 2);  // points must be odd

      Scalar dx = (upper - lower) / (points - 1);
      Scalar x = lower + dx;
      Scalar f = ym;
      for (unsigned i = 1; i < points - 1; ++i) {
        f += (2 + 2 * (i % 2)) * func(x);
        x += dx;
      }
      f += yp;

      return dx * f / 3;
    }

  };

  }  // namespace integrate
}    // namespace vice

#endif  // _VICE_INTEGRATE_H_
