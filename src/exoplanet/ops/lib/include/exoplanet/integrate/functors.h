#ifndef _EXOPLANET_INTEGRATE_FUNCTORS_H_
#define _EXOPLANET_INTEGRATE_FUNCTORS_H_

#include <cmath>
#include <limits>
#include <tuple>

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include "exoplanet/kepler.h"
#include "exoplanet/starry/limbdark.h"

namespace exoplanet {
namespace integrate {

template <typename Scalar>
inline std::tuple<Scalar, Scalar> do_kepler(const Scalar& M, const Scalar& ecc) {
  const Scalar E = kepler::solve_kepler(M, ecc);
  const Scalar sinE = std::sin(E);
  const Scalar cosE = std::cos(E);
  auto denom = 1 + cosE;
  if (denom > std::numeric_limits<Scalar>::epsilon()) {
    const auto tanf2 = sqrt((1 + ecc) / (1 - ecc)) * sinE / denom;  // tan(0.5*f)
    const auto tanf2_2 = tanf2 * tanf2;

    // Then we compute sin(f) and cos(f) using:
    // sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    // cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tanf2_2);
    Scalar sinf = 2 * tanf2 * denom;
    Scalar cosf = (1 - tanf2_2) * denom;
    return std::make_tuple(sinf, cosf);
  }
  return std::make_tuple(Scalar(0), Scalar(-1));
}

template <typename Grad>
inline std::tuple<Eigen::AutoDiffScalar<Grad>, Eigen::AutoDiffScalar<Grad>> do_kepler(
    const Eigen::AutoDiffScalar<Grad>& M, const Eigen::AutoDiffScalar<Grad>& ecc) {
  typedef typename Grad::Scalar Scalar;

  const Scalar Mval = M.value();
  const Scalar eccval = ecc.value();

  Scalar sinf, cosf;
  std::tie(sinf, cosf) = do_kepler(Mval, eccval);

  // e * cos(f)
  const Scalar ecosf = eccval * cosf;

  // 1 - e^2
  const Scalar ome2 = 1 - eccval * eccval;

  // Partials
  const Scalar opecof = 1 + ecosf;
  const Scalar dfdM = opecof * opecof / (ome2 * std::sqrt(ome2));
  const Scalar dfde = (2 + ecosf) * sinf / ome2;
  const Grad grad = dfdM * M.derivatives() + dfde * ecc.derivatives();

  return std::make_tuple(Eigen::AutoDiffScalar<Grad>(sinf, cosf * grad),
                         Eigen::AutoDiffScalar<Grad>(cosf, -sinf * grad));
}

template <typename T, typename Scalar>
class CircLimbDarkFunctor {
 private:
  long int num_eval_;

  // Parameters of the model
  T n_, a_, sini_, cosi_, r_;

 protected:
  starry::limbdark::GreensLimbDark<Scalar>* L_;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec_;

 public:
  CircLimbDarkFunctor(starry::limbdark::GreensLimbDark<Scalar>* L,
                      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec)
      : num_eval_(0), cvec_(cvec.rows()), L_(L) {
    for (int i = 0; i < cvec.rows(); ++i) cvec_(i) = cvec(i);
  }

  long int num_eval() const { return num_eval_; }

  void set_r(T r) { r_ = r; }

  virtual void set_parameters(T n, T a, T sini, T cosi, T r) {
    n_ = n;
    a_ = a;
    sini_ = sini;
    cosi_ = cosi;
    this->set_r(r);
  }

  virtual std::tuple<T, T, T> get_coords(const T& dt) {
    const T M = dt * n_;
    const auto sinf = sin(M);
    const auto cosf = cos(M);

    const auto x0 = a_ * cosf;
    const auto y0 = a_ * sinf;

    const auto y2 = cosi_ * y0;
    const auto z2 = -sini_ * y0;

    return std::make_tuple(x0, y2, z2);
  }

  T operator()(const T& t) {
    num_eval_++;
    T x, y, z;
    std::tie(x, y, z) = this->get_coords(t);

    // Not transiting
    if (z <= 0) return 0 * r_;

    // Compute the impact parameter
    const T b = sqrt(x * x + y * y);
    if (b >= (1 - 1e-8) + r_) return 0 * r_;

    // Use starry to compute the solution vector
    L_->template compute<true>(b.value(), r_.value());
    auto sT = L_->sT;

    // Propagate the gradient
    auto grad = L_->dsTdb.dot(cvec_) * b.derivatives() + L_->dsTdr.dot(cvec_) * r_.derivatives();
    T val = T(sT.dot(cvec_) - 1, grad);

    // Update the limb darkening gradient
    val.derivatives().tail(cvec_.rows()) += sT;

    return val;
  }
};

template <typename T, typename Scalar>
class LimbDarkFunctor : public CircLimbDarkFunctor<T, Scalar> {
 private:
  // Parameters of the model
  T n_, aome2_, e_, sinw_, cosw_, sini_, cosi_;

 public:
  LimbDarkFunctor(starry::limbdark::GreensLimbDark<Scalar>* L,
                  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> cvec)
      : CircLimbDarkFunctor<T, Scalar>(L, cvec) {}

  void set_parameters(T n, T aome2, T e, T sinw, T cosw, T sini, T cosi, T r) {
    n_ = n;
    aome2_ = aome2;
    e_ = e;
    sinw_ = sinw;
    cosw_ = cosw;
    sini_ = sini;
    cosi_ = cosi;
    this->set_r(r);
  }

  std::tuple<T, T, T> get_coords(const T& dt) {
    const T M = dt * n_;
    T sinf, cosf;
    std::tie(sinf, cosf) = do_kepler(M, e_);
    const T r = aome2_ / (1 + e_ * cosf);

    const T x0 = r * cosf;
    const T y0 = r * sinf;

    const T x1 = x0 * cosw_ - y0 * sinw_;
    const T y1 = x0 * sinw_ + y0 * cosw_;

    const T y2 = cosi_ * y1;
    const T z2 = -sini_ * y1;

    return std::make_tuple(x1, y2, z2);
  }
};

}  // namespace integrate
}  // namespace exoplanet

#endif  // _EXOPLANET_INTEGRATE_FUNCTORS_H_
