#ifndef _TRANSIT_LIMB_DARKENING_H_
#define _TRANSIT_LIMB_DARKENING_H_

#include <cmath>

namespace transit {

  template <typename Scalar>
  class QuadraticLimbDarkening {

    public:
      QuadraticLimbDarkening (Scalar c1, Scalar c2) : c1_(c1), c2_(c2) {
        I0_ = M_PI * (1.0 - (2.0 * c1_ + c2) / 6.0);
      }

      Scalar value (const Scalar& x) const {
        Scalar mu = 1.0 - sqrt(1.0 - x * x);
        return (1.0 - c1_ * mu - c2_ * mu * mu) / I0_;
      }

      Scalar value_fwd (const Scalar& x, const Scalar& scale, Scalar* grad) const {
        Scalar mu = 1.0 - sqrt(1.0 - x * x), mu2 = mu*mu;
        Scalar I = (1.0 - c1_ * mu - c2_ * mu2) / I0_;
        Scalar factor = M_PI * scale * I / (6.0 * I0_);

        grad[0] += -scale * mu / I0_  + 2.0 * factor;
        grad[1] += -scale * mu2 / I0_ + factor;

        return I;
      }

    private:
      Scalar c1_, c2_, I0_;

  };

  template <typename Scalar>
  class NonLinearLimbDarkening {

    public:
      NonLinearLimbDarkening (Scalar c1, Scalar c2, Scalar c3, Scalar c4) : c1_(c1), c2_(c2), c3_(c3), c4_(c4) {
        I0_ = M_PI * (1.0 - c1_ / 5.0 - c2 / 3.0 - 3.0 * c3_ / 7.0 - c4 / 2.0);
      }

      int size () const { return 4; }

      Scalar value (const Scalar& x) const {
        Scalar mu = sqrt(1.0 - x * x), sqrtmu = sqrt(mu);
        return (1.0 - c1_ * (1.0 - sqrtmu) - c2_ * (1.0 - mu) - c3_ * (1.0 - mu*sqrtmu) - c4_ * (1.0 - mu*mu)) / I0_;
      }

      Scalar value_fwd (const Scalar& x, const Scalar& scale, Scalar* grad) const {
        Scalar mu = sqrt(1.0 - x * x), sqrtmu = sqrt(mu);
        Scalar a1 = 1.0 - sqrtmu,
               a2 = 1.0 - mu,
               a3 = 1.0 - mu*sqrtmu,
               a4 = 1.0 - mu*mu;
        Scalar I = (1.0 - c1_ * a1 - c2_ * a2 - c3_ * a3 - c4_ * a4) / I0_;
        Scalar factor = M_PI * I * scale / I0_;

        grad[0] += -scale * a1 / I0_ + factor / 5.0;
        grad[1] += -scale * a2 / I0_ + factor / 3.0;
        grad[2] += -scale * a3 / I0_ + factor * 3.0 / 7.0;
        grad[3] += -scale * a4 / I0_ + factor / 2.0;

        return I;
      }

    private:
      Scalar c1_, c2_, c3_, c4_, I0_;

  };

};

#endif
