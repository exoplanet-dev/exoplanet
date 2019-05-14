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
        long int num_eval_;
        T r_, x_, xt_, xtt_, y_, yt_, ytt_;
        Scalar z_, zt_, dt_;
        starry::limbdark::GreensLimbDark<Scalar>* L_;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec_;

      public:
        LimbDarkFunctor (starry::limbdark::GreensLimbDark<Scalar>* L,
                         Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec)
        : num_eval_(0)
        , cvec_(cvec.rows())
        , L_(L)
        {
          for (int i = 0; i < cvec.rows(); ++i) cvec_(i) = cvec(i);
        }

        int setup (T r, T x, T xt, T xtt, T y, T yt, T ytt, Scalar z, Scalar zt, Scalar dt, std::vector<Scalar>& limits, bool include_contacts=true) {
          r_ = abs(r);
          x_   = x;
          xt_  = xt;
          xtt_ = 0.5 * xtt;
          y_   = y;
          yt_  = yt;
          ytt_ = 0.5 * ytt;
          z_   = z;
          zt_  = zt;
          dt_  = dt;

          Scalar start = -0.5*dt_;
          Scalar stop = 0.5*dt_;
          Scalar limit;
          limits.resize(10);
          limits[0] = start;
          limits[1] = stop;

          int j = 2;
          // if (include_contacts) {
          //   if (abs(a0_.value()) < 1e-10) {
          //     for (int sgn1 = -1; sgn1 <= 1; sgn1 += 2) {
          //       for (int sgn2 = -1; sgn2 <= 1; sgn2 += 2) {
          //         auto limit = (-b0_.value() + sgn2 * (1 + sgn1 * r_.value())) / v0_.value();
          //         if (start < limit && limit < stop) limits[j++] = limit;
          //       }
          //     }
          //   } else {
          //     auto t1 = -0.5 * v0_.value() / a0_.value();
          //     for (int sgn1 = -1; sgn1 <= 1; sgn1 += 2) {
          //       for (int sgn2 = -1; sgn2 <= 1; sgn2 += 2) {
          //         auto arg = v2 - 4 * a0_.value() * (b0_.value() - sgn2 * (1 + sgn1 * r_.value()));
          //         if (arg < 0) continue;
          //         auto t2 = 0.5 * sqrt(arg) / a0_.value();
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

        T get_b (Scalar t) {
          auto t2 = t*t;
          auto x = x_ + xt_ * t + xtt_ * t2;
          auto y = y_ + yt_ * t + ytt_ * t2;
          return sqrt(x*x + y*y);
        }

        T operator() (Scalar t) {
          if (z_ + zt_ * t <= 0) return 0.0 * r_;
          T b = get_b(t);
          if (b >= 1 + r_ - 1e-8) return 0.0 * b;
          num_eval_++;

          L_->compute(b.value(), r_.value(), true);

          auto S = L_->S;
          auto grad = L_->dSdb.dot(cvec_) * b.derivatives() + L_->dSdr.dot(cvec_) * r_.derivatives();
          T val = T(S.dot(cvec_) - 1, grad);
          val.derivatives().tail(grad.rows() - 7) += S.transpose();

          return val;
        }

        long int num_eval () const { return num_eval_; }

    };

  }  // namespace functors
}    // namespace vice

#endif  // _VICE_FUNCTORS_H_