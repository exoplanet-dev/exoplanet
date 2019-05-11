#ifndef _VICE_FUNCTORS_H_
#define _VICE_FUNCTORS_H_

#include <cmath>
#include <vector>
#include <algorithm>

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include <limbdark.h>

namespace vice {
  namespace functors {

    using std::abs;

    template <typename T, typename Scalar>
    class LimbDarkFunctor {
      private:
        T r_, b0_, v0_, a0_;
        Scalar dt_;
        starry::limbdark::GreensLimbDark<Scalar>* L_;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec_;

      public:
        LimbDarkFunctor (starry::limbdark::GreensLimbDark<Scalar>* L,
                         Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec)
        : cvec_(cvec.rows())
        , L_(L)
        {
          for (int i = 0; i < cvec.rows(); ++i) cvec_(i) = cvec(i);
        }

        int setup (T r, T b, T v, T a, Scalar dt, std::vector<Scalar>& limits, bool include_contacts=true) {
          r_ = abs(r);
          b0_ = b;
          v0_ = v;
          a0_ = 0.5 * a;
          dt_ = dt;
          auto v2 = v0_ * v0_;

          Scalar start = -0.5*dt_;
          Scalar stop = 0.5*dt_;
          Scalar limit;
          limits.resize(10);
          limits[0] = start;
          limits[1] = stop;

          int j = 2;
          // if (include_contacts) {
          //   if (std::abs(a0_) < 1e-10) {
          //     for (int sgn1 = -1; sgn1 <= 1; sgn1 += 2) {
          //       for (int sgn2 = -1; sgn2 <= 1; sgn2 += 2) {
          //         auto limit = (-b0_ + sgn2 * (1 + sgn1 * r)) / v0_;
          //         if (start < limit && limit < stop) limits[j++] = limit;
          //       }
          //     }
          //   } else {
          //     auto t1 = -0.5 * v0_ / a0_;
          //     for (int sgn1 = -1; sgn1 <= 1; sgn1 += 2) {
          //       for (int sgn2 = -1; sgn2 <= 1; sgn2 += 2) {
          //         auto arg = v2 - 4 * a0_ * (b0_ - sgn2 * (1 + sgn1 * r));
          //         if (arg < 0) continue;
          //         auto t2 = 0.5 * std::sqrt(arg) / a0_;
          //         limit = t1 + t2;
          //         if (start < limit && limit < stop) limits[j++] = limit;
          //         limit = t1 - t2;
          //         if (start < limit && limit < stop) limits[j++] = limit;
          //       }
          //     }
          //   }
          // }

          // std::sort(limits.begin(), limits.begin() + j);

          return j;
        }

        T operator() (Scalar t) {
          T b = abs(b0_ + v0_ * t + a0_ * t * t);
          if (b >= 1 + r_) return 0.0 * b;
          L_->compute(b.value(), r_.value(), true);

          auto S = L_->S;
          auto grad = L_->dSdb.dot(cvec_) * b.derivatives() + L_->dSdr.dot(cvec_) * r_.derivatives();
          T val = T(S.dot(cvec_) - 1, grad);
          val.derivatives().tail(grad.rows() - 4) += S.transpose();

          return val;
        }

    };

  }  // namespace functors
}    // namespace vice

#endif  // _VICE_FUNCTORS_H_