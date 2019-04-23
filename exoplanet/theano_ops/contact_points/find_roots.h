#include <cmath>
#include <tuple>
#include <vector>
#include <Eigen/Core>
#include "theia/find_polynomial_roots_companion_matrix.h"

namespace contact_points {

  template <typename Scalar>
  Eigen::Matrix<Scalar, 6, 1> get_quadratic (Scalar a, Scalar e, Scalar cosw, Scalar sinw, Scalar cosi, Scalar sini) {
    Scalar e2 = e*e;
    Scalar e2mo = e2 - 1;
    Eigen::Matrix<Scalar, 6, 1> quad;
    quad << e2*cosw*cosw - 1,
            2*e2*sinw*cosw/sini,
            (e2mo - e2*cosw*cosw)/(sini*sini),
            -2*a*e*e2mo*cosw,
            -2*a*e*e2mo*sinw/sini,
            a*a*e2mo*e2mo;
    return quad;
  }

  template <typename Scalar>
  Eigen::Matrix<Scalar, 5, 1> get_quartic (const Eigen::Matrix<Scalar, 6, 1>& quad, Scalar T, Scalar L) {
    Scalar A = quad(0);
    Scalar B = quad(1);
    Scalar C = quad(2);
    Scalar D = quad(3);
    Scalar E = quad(4);
    Scalar F = quad(5);
    Scalar A2 = A*A;
    Scalar B2 = B*B;
    Scalar C2 = C*C;
    Scalar D2 = D*D;
    Scalar E2 = E*E;
    Scalar F2 = F*F;
    Scalar T2 = T*T;
    Scalar L2 = L*L;
    Eigen::Matrix<Scalar, 5, 1> quartic;
    quartic << A2*T2 - 2*A*C*T + B2*T + C2,
               2*T*(A*D*T + B*E - C*D),
               2*A*C*L2*T + 2*A*F*T2 - B2*L2*T - 2*C2*L2 - 2*C*F*T + D2*T2 + E2*T,
               -2*T*(B*E*L2 - C*D*L2 - D*F*T),
               C2*L2*L2 + 2*C*F*L2*T - E2*L2*T + F2*T2;
    return quartic;
  }

  template <typename Scalar>
  std::tuple<int, Scalar, Scalar> find_roots (Scalar a, Scalar e, Scalar cosw, Scalar sinw, Scalar cosi, Scalar sini, Scalar L, Scalar tol = 1e-8) {
    L /= a;
    a = 1.0;

    // Precompute the relevant quantities
    auto f0 = 2.0 * atan2(cosw, 1 + sinw);

    // Get the coefficients of the base quadratic equation
    auto quad = get_quadratic(a, e, cosw, sinw, cosi, sini);
    auto T = cosi / sini;
    T *= T;

    // Deal with special edge cases
    std::vector<Scalar> roots;
    if (std::abs(e) < tol) {

      // A circular orbit
      auto x2 = (quad(2)*L*L + quad(5)*T) / (quad(2) + quad(5));
      if (x2 < 0) {
        return std::make_tuple(1, f0 - M_PI, f0 + M_PI);
      }
      roots.push_back(sqrt(x2));
      roots.push_back(-sqrt(x2));

    } else if (std::abs(sinw) < tol) {

      // Small omega
      auto b0 = quad(5)*T + quad(2)*L*L;
      auto b1 = quad(3)*T;
      auto b2 = quad(0)*T - quad(2);
      auto x1 = -0.5 * b1 / b2;
      auto arg = b1*b1 - 4*b0*b2;
      if (arg < 0) {
        return std::make_tuple(2, f0 - M_PI, f0 + M_PI);
      }
      auto x2 = 0.5 * sqrt(arg) / b2;
      roots.push_back(x1 - x2);
      roots.push_back(x1 + x2);

    } else {

      // General case
      auto quartic = get_quartic(quad, T, L);
      Eigen::Matrix<Scalar, 4, 1> real(4), imag(4);
      bool flag = theia::FindPolynomialRootsCompanionMatrix(quartic, real, imag);
      if (!flag) {
        return std::make_tuple(3, f0 - M_PI, f0 + M_PI);
      }
      // Only select real roots
      for (int i = 0; i < 4; ++i) {
        if (std::abs(imag(i)) < tol) {
          roots.push_back(real[i]);
        }
      }

    }

    // Fail now if the roots don't span zero
    bool flag_less = false, flag_greater = false;
    for (int i = 0; i < roots.size(); ++i) {
      if (roots[i] <= 0) flag_less = true;
      else flag_greater = true;
    }
    if (!flag_less || !flag_greater) {
      return std::make_tuple(4, f0 - M_PI, f0 + M_PI);
    }

    // Loop over the roots and compute the z coordinates
    std::vector<Scalar> angles, angle_roots;
    for (int xi = 0; xi < roots.size(); ++xi) {

      // Solve the quadratic for z
      auto x = roots[xi];
      auto b0 = quad(0)*x*x + quad(3)*x + quad(5);
      auto b1 = quad(1)*x + quad(4);
      auto b2 = quad(2);
      auto z1 = -0.5 * b1 / b2;
      auto arg = b1*b1 - 4*b0*b2;
      if (arg < 0) continue;
      auto z2 = 0.5 * sqrt(arg) / b2;

      // Loop over the two allowed zs
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        auto z = z1 + sgn * z2;
        if (z > 0) continue;
        auto y = z * cosi / sini;
        if (std::abs(x*x + y*y - L*L) > tol) continue;
        auto x0 = x*cosw + z*sinw/sini;
        auto y0 = -x*sinw + z*cosw/sini;
        auto angle = atan2(y0, x0) - M_PI;
        while (angle < -M_PI) angle += 2*M_PI;
        angles.push_back(angle);
        angle_roots.push_back(x);
      }

    }

    // Fail if no angles were valid or if they don't span anymore
    // then deal with multiplicity
    if (angles.size() < 2) {
      return std::make_tuple(5, f0 - M_PI, f0 + M_PI);
    } else if (angles.size() > 2) {

      // If there are more than 2 roots, we select the one that gives the
      // closest to the correct impact parameter splitting the space into
      // negative and positive roots
      Scalar dist_less, dist_greater, angle_less, angle_greater;
      flag_less = true;
      flag_greater = true;
      for (int i = 0; i < angles.size(); ++i) {
        auto cosf = std::cos(angles[i]);
        auto sinf = std::sin(angles[i]);
        auto factor = (e*e - 1) / (e*cosf + 1);
        factor *= factor;
        auto dist = factor*(cosi*cosi*std::pow(cosw*sinf + sinw*cosf, 2) +
                            std::pow(cosw*cosf - sinw*sinf, 2)) - L*L;
        if (angle_roots[i] <= 0) {
          if (flag_less || (dist < dist_less)) {
            dist_less = dist;
            angle_less = angles[i];
            flag_less = false;
          }
        } else {
          if (flag_greater || (dist < dist_greater)) {
            dist_greater = dist;
            angle_greater = angles[i];
            flag_greater = false;
          }
        }

      }

      if (flag_less || flag_greater) {
        return std::make_tuple(7, f0 - M_PI, f0 + M_PI);
      }

      angles.resize(2);
      angles[0] = angle_less;
      angles[1] = angle_greater;
    }

    // Make sure that the angles span the transit
    Scalar angle1 = angles[0] - f0;
    Scalar angle2 = angles[1] - f0;
    if ((angle1 > 0) && (angle2 > 0)) {
      Scalar tmp = angle1;
      angle1 = angle2 - 2 * M_PI;
      angle2 = tmp;
    } else if ((angle1 < 0) && (angle2 < 0)) {
      Scalar tmp = angle2;
      angle2 = angle1 + 2 * M_PI;
      angle1 = tmp;
    }

    return std::make_tuple(0, angle1, angle2);
  }

}
