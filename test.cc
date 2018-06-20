#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include "transit/transit.h"
#include "transit/limb_darkening.h"

using namespace transit;

int main () {
  typedef Eigen::AutoDiffScalar<Eigen::Matrix<double, 6, 1> > Scalar;

  Scalar c1(0.5, 6, 2), c2(0.1, 6, 3), c3(0.5, 6, 4), c4(0.1, 6, 5),
         z(0.5, 6, 0), r(0.01, 6, 1);
  double step_scale = 1e-4;

  NonLinearLimbDarkening<Scalar> ld(c1, c2, c3, c4);
  Scalar result = delta<Scalar, NonLinearLimbDarkening<Scalar> >(ld, step_scale, z, r);
  std::cout << result.derivatives().transpose() << "\n\n";


  Eigen::Matrix<double, 6, 1> grad(6);
  grad.setZero();

  NonLinearLimbDarkening<double> ld_2(c1.value(), c2.value(), c3.value(), c4.value());
  double result_2 = delta_fwd<double, NonLinearLimbDarkening<double> >(ld_2, step_scale, z.value(), r.value(), grad.data());
  std::cout << grad.transpose() << "\n";

  return 0;
}
