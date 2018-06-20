#ifndef _TRANSIT_TRANSIT_H_
#define _TRANSIT_TRANSIT_H_

#include <cmath>

namespace transit {

  template <typename Scalar>
  Scalar step_advance (const Scalar& step_scale, const Scalar& x) {
    return x + step_scale * acos(x);
  }

  template <typename Scalar>
  Scalar area (const Scalar& x, const Scalar& z, const Scalar& r) {
    if (x <= r - z) {
      return M_PI * x * x;
    } else if (x >= z + r) {
      return M_PI * r * r;
    }

    Scalar z2 = z*z, x2 = x*x, r2 = r*r;
    Scalar u = 0.5 * (z2 + x2 - r2) / (z * x);
    Scalar v = 0.5 * (z2 + r2 - x2) / (z * r);
    Scalar w = (-z + x + r) * (z + x - r) * (z - x + r) * (z + x + r);
    if (w < 0.0) w = 0.0;
    return x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w);
  }

  template <typename Scalar>
  Scalar area_fwd (const Scalar& x, const Scalar& z, const Scalar& r, Scalar* grad) {
    if (x <= r - z) {
      grad[0] = 0.0;
      grad[1] = 0.0;
      return M_PI * x * x;
    } else if (x >= z + r) {
      grad[0] = 0.0;
      grad[1] = 2.0 * M_PI * r;
      return M_PI * r * r;
    }

    // Compute the area as usual
    Scalar z2 = z*z, x2 = x*x, r2 = r*r;
    Scalar zx = z * x;
    Scalar zr = z * r;
    Scalar u = 0.5 * (z2 + x2 - r2) / zx;
    Scalar v = 0.5 * (z2 + r2 - x2) / zr;
    Scalar w = (-z + x + r) * (z + x - r) * (z - x + r) * (z + x + r);
    if (w < 0.0) w = 0.0;

    // Some shortcuts
    Scalar acosv = acos(v);
    Scalar sqrtw = sqrt(w);

    // Compute all the partials
    Scalar dudz = 0.5 * (1.0 / x - (x2 - r2) / (z2 * x));
    Scalar dudr = -r / zx;

    Scalar dvdz = 0.5 * (1.0 / r - (r2 - x2) / (z2 * r));
    Scalar dvdr = 0.5 * (1.0 / z - (z2 - x2) / (r2 * z));

    Scalar dwdz = 4.0 * z * (r2 - z2 + x2);
    Scalar dwdr = 4.0 * r * (z2 - r2 + x2);

    Scalar dacosu = -1.0 / sqrt(1.0 - u * u);
    Scalar dacosv = -1.0 / sqrt(1.0 - v * v);

    // Combine the partials as needed
    grad[0] = x2 * dacosu * dudz + r2 * dacosv * dvdz - 0.25 * dwdz / sqrtw;
    grad[1] = x2 * dacosu * dudr + 2.0 * r * acosv + r2 * dacosv * dvdr - 0.25 * dwdr / sqrtw;

    return x2 * acos(u) + r2 * acosv - 0.5 * sqrtw;
  }

  template <typename Scalar, typename LimbDarkening>
  Scalar delta (const LimbDarkening& ld, const Scalar& step_scale, const Scalar& z, const Scalar& r) {
    Scalar xmin = z - r;
    Scalar xmax = z + r;
    if (xmin < 0.0) xmin = 0.0;
    if (xmax > 1.0) xmax = 1.0;
    if (xmin >= xmax) return 0.0;
    Scalar delta = 0.0,
            x1 = xmin, x2,
            A1 = 0.0, A2;
    while (x1 < xmax) {
      x2 = step_advance(step_scale, x1);
      A2 = area(x2, z, r);
      delta += (A2 - A1) * ld.value(0.5*(x1+x2));
      x1 = x2;
      A1 = A2;
    }
    return delta;
  }

  template <typename Scalar, typename LimbDarkening>
  Scalar delta_fwd (const LimbDarkening& ld, const Scalar& step_scale, const Scalar& z, const Scalar& r, Scalar* grad) {
    Scalar xmin = z - r;
    Scalar xmax = z + r;
    if (xmin < 0.0) xmin = 0.0;
    if (xmax > 1.0) xmax = 1.0;
    if (xmin >= xmax) return 0.0;

    Scalar delta = 0.0,
           I,
           x1 = xmin, x2,
           A1 = 0.0, A2, dA;
    Scalar A1_grad[2], A2_grad[2];
    A1_grad[0] = 0.0;
    A2_grad[0] = 0.0;

    while (x1 < xmax) {
      x2 = step_advance(step_scale, x1);
      A2 = area_fwd(x2, z, r, A2_grad);
      dA = A2 - A1;
      I = ld.value_fwd(0.5*(x1+x2), dA, &(grad[2]));

      delta += dA * I;

      grad[0] += (A2_grad[0] - A1_grad[0]) * I;
      grad[1] += (A2_grad[1] - A1_grad[1]) * I;

      x1 = x2;
      A1 = A2;
      A1_grad[0] = A2_grad[0];
      A1_grad[1] = A2_grad[1];
    }
    return delta;
  }

};

#endif
