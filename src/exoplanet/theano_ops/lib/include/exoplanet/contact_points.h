#ifndef _EXOPLANET_CONTACT_POINTS_H_
#define _EXOPLANET_CONTACT_POINTS_H_

#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

namespace exoplanet {
namespace contact_points {

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename Scalar>
class ContactPointSolver {
 private:
  Scalar a, e, cosw, sinw, cosi, sini;
  Scalar Efactor;
  std::vector<Scalar> quad;

 public:
  ContactPointSolver(Scalar a_, Scalar e_, Scalar cosw_, Scalar sinw_,
                     Scalar cosi_, Scalar sini_)
      : a(a_),
        e(e_),
        cosw(cosw_),
        sinw(sinw_),
        cosi(cosi_),
        sini(-sini_),
        quad(6) {
    build_quadratic();
    Efactor = sqrt((1 - e) / (1 + e));
  }

  std::tuple<int, Scalar, Scalar> find_roots(Scalar L,
                                             Scalar tol = 1e-8) const {
    L /= a;
    auto L2 = L * L;

    Scalar x_left = -L;
    Scalar x_right = L;
    if (std::abs(cosi) > tol) {
      // Check the signs
      auto const f_left = objective(L2, -L);
      auto const f_mid = objective(L2, 0);
      auto const f_right = objective(L2, L);
      if (!std::get<0>(f_left) || !std::get<0>(f_mid) ||
          !std::get<0>(f_right) ||
          sgn(std::get<1>(f_left)) == sgn(std::get<1>(f_mid)) ||
          sgn(std::get<1>(f_mid)) == sgn(std::get<1>(f_right))) {
        return std::make_tuple<int, Scalar, Scalar>(1, 0, 0);
      }

      // Find the roots
      auto const root_left = bisect(L2, -L, 0, tol);
      auto const root_right = bisect(L2, 0, L, tol);
      if (!std::get<0>(root_left) || !std::get<0>(root_right)) {
        return std::make_tuple<int, Scalar, Scalar>(2, 0, 0);
      }

      // Compute the angles
      x_left = std::get<1>(root_left);
      x_right = std::get<1>(root_right);
    }

    auto const M_left = convert_to_mean_anomaly(x_left);
    auto const M_right = convert_to_mean_anomaly(x_right);
    if (!std::get<0>(M_left) || !std::get<0>(M_right)) {
      return std::make_tuple<int, Scalar, Scalar>(3, 0, 0);
    }

    return std::make_tuple(0, std::get<1>(M_left), std::get<1>(M_right));
  }

 private:
  void build_quadratic() {
    auto const e2 = e * e;
    auto const e2mo = e2 - 1;
    quad[0] = e2 * cosw * cosw - 1;
    quad[1] = 2 * e2 * sinw * cosw / sini;
    quad[2] = (e2mo - e2 * cosw * cosw) / (sini * sini);
    quad[3] = -2 * e * e2mo * cosw;
    quad[4] = -2 * e * e2mo * sinw / sini;
    quad[5] = e2mo * e2mo;
  }

  std::tuple<bool, Scalar> convert_to_mean_anomaly(Scalar x) const {
    auto const coords = get_coords(x);
    auto const z = std::get<2>(coords);
    auto const f =
        atan2(-x * sinw + z * cosw / sini, x * cosw + z * sinw / sini) - M_PI;
    auto const E = 2 * atan(Efactor * tan(0.5 * f));
    auto const M = E - e * sin(E);
    return std::make_tuple(std::get<0>(coords), M);
  }

  std::tuple<bool, Scalar, Scalar> get_coords(Scalar x) const {
    auto const b0 = quad[0] * x * x + quad[3] * x + quad[5];
    auto const b1 = quad[1] * x + quad[4];
    auto const b2 = quad[2];
    auto const z1 = -0.5 * b1 / b2;
    auto const arg = b1 * b1 - 4 * b0 * b2;
    if (arg < 0) {
      return std::make_tuple<bool, Scalar, Scalar>(false, 0, 0);
    }
    auto const z2 = 0.5 * sqrt(arg) / b2;
    Scalar z = z1 + z2;
    if (z < 0) {
      z = z1 - z2;
      if (z < 0) {
        return std::make_tuple<bool, Scalar, Scalar>(false, 0, 0);
      }
    }
    Scalar y = z * cosi / sini;
    return std::make_tuple(true, y, z);
  }

  std::tuple<bool, Scalar> objective(Scalar L2, Scalar x) const {
    auto const coords = get_coords(x);
    auto const flag = std::get<0>(coords);
    auto const y = std::get<1>(coords);
    return std::make_tuple(flag, y * y + x * x - L2);
  }

  // https://codereview.stackexchange.com/questions/179516/finding-the-root-of-a-function-by-bisection-method
  std::tuple<bool, Scalar> bisect(Scalar L2, Scalar min, Scalar max,
                                  Scalar epsilon) const {
    auto f_min = std::get<1>(objective(L2, min));
    while (min + epsilon < max) {
      auto const mid = 0.5 * min + 0.5 * max;
      auto const result = objective(L2, mid);
      if (!std::get<0>(result)) return std::make_tuple(false, mid);
      auto const f_mid = std::get<1>(result);
      if ((f_min < 0) == (f_mid < 0)) {
        min = mid;
        f_min = f_mid;
      } else {
        max = mid;
      }
    }
    return std::make_tuple(true, 0.5 * min + 0.5 * max);
  }
};

}  // namespace contact_points
}  // namespace exoplanet

#endif
