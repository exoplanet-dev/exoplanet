#ifndef _EXOPLANET_CELERITE_H_
#define _EXOPLANET_CELERITE_H_

#include <Eigen/Core>

#ifndef CELERITE_J
#define CELERITE_J Eigen::Dynamic
#define CELERITE_J2 Eigen::Dynamic
#define CELERITE_J_ORDER Eigen::RowMajor
#endif

#ifndef CELERITE_NRHS
#define CELERITE_NRHS Eigen::Dynamic
#define CELERITE_JNRHS Eigen::Dynamic
#define CELERITE_NRHS_ORDER Eigen::RowMajor
#define CELERITE_JNRHS_ORDER Eigen::RowMajor
#endif

namespace exoplanet {
namespace celerite {

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T1, typename T2, typename T3, typename T4>
void to_dense(const Eigen::MatrixBase<T1>& a,  // (N)
              const Eigen::MatrixBase<T2>& U,  // (N, J)
              const Eigen::MatrixBase<T2>& V,  // (N, J)
              const Eigen::MatrixBase<T3>& P,  // (N-1, J)
              Eigen::MatrixBase<T4>& K         // (N, N)
) {
  int N = U.rows(), J = U.cols();
  Eigen::Matrix<typename T2::Scalar, 1, T2::ColsAtCompileTime> u, v;
  Eigen::Matrix<typename T3::Scalar, T3::ColsAtCompileTime, 1> p(J);
  for (int n = 0; n < N; ++n) {
    v = V.row(n);
    p.setConstant(1.0);
    K(n, n) = a(n);
    for (int m = n + 1; m < N; ++m) {
      p.array() *= P.row(m - 1).array();
      u = U.row(m);
      K(n, m) = u * p.asDiagonal() * v.transpose();
      K(m, n) = K(n, m);
    }
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6>
void matmul(const Eigen::MatrixBase<T1>& a,  // (N)
            const Eigen::MatrixBase<T2>& U,  // (N, J)
            const Eigen::MatrixBase<T2>& V,  // (N, J)
            const Eigen::MatrixBase<T3>& P,  // (N-1, J)
            const Eigen::MatrixBase<T4>& Z,  // (N, Nrhs)
            Eigen::MatrixBase<T5>& Y,        // (N, Nrhs)
            Eigen::MatrixBase<T6>& F_plus,   // (J, Nrhs)
            Eigen::MatrixBase<T6>& F_minus   // (J, Nrhs)
) {
  int N = U.rows();

  Y.row(N - 1) = a(N - 1) * Z.row(N - 1);

  F_plus.setZero();
  for (int n = N - 2; n >= 0; --n) {
    F_plus = P.row(n).asDiagonal() * (F_plus + U.row(n + 1).transpose() * Z.row(n + 1));
    Y.row(n) = a(n) * Z.row(n) + V.row(n) * F_plus;
  }

  F_minus.setZero();
  for (int n = 1; n < N; ++n) {
    F_minus = P.row(n - 1).asDiagonal() * (F_minus + V.row(n - 1).transpose() * Z.row(n - 1));
    Y.row(n) += U.row(n) * F_minus;
  }
}

template <typename T1, typename T2, typename T3, typename T4>
void dotL(const Eigen::MatrixBase<T1>& U,  // (N, J)
          const Eigen::MatrixBase<T2>& P,  // (N-1, J)
          const Eigen::MatrixBase<T3>& d,  // (N)
          const Eigen::MatrixBase<T1>& W,  // (N, J)
          Eigen::MatrixBase<T4>& Z         // (N, Nrhs); initially set to Y
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T3::Scalar, T3::RowsAtCompileTime, 1> sqrtd = sqrt(d.array());
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T4::ColsAtCompileTime> F(J, nrhs);
  Eigen::Matrix<typename T4::Scalar, 1, T4::ColsAtCompileTime> tmp(1, nrhs);

  F.setZero();
  Z.row(0) *= sqrtd(0);
  tmp = Z.row(0);

  for (int n = 1; n < N; ++n) {
    F = P.row(n - 1).asDiagonal() * (F + W.row(n - 1).transpose() * tmp);
    tmp = sqrtd(n) * Z.row(n);
    Z.row(n) = tmp + U.row(n) * F;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
int factor(const Eigen::MatrixBase<T1>& U,  // (N, J)
           const Eigen::MatrixBase<T2>& P,  // (N-1, J)
           Eigen::MatrixBase<T3>& d,        // (N);    initially set to A
           Eigen::MatrixBase<T4>& W,        // (N, J); initially set to V
           Eigen::MatrixBase<T5>& S         // (N, J*J)
) {
  int N = U.rows(), J = U.cols();

  Eigen::Matrix<typename T1::Scalar, 1, T1::ColsAtCompileTime> tmp(1, J);
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T1::ColsAtCompileTime> S_(J, J);

  // First row
  S_.setZero();
  S.row(0).setZero();
  W.row(0) /= d(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
    // Update S = diag(P) * (S + d*W*W.T) * diag(P)
    S_.noalias() += d(n - 1) * W.row(n - 1).transpose() * W.row(n - 1);
    S_ = P.row(n - 1).asDiagonal() * S_;
    // S_.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < J; ++k) S(n, j * J + k) = S_(j, k);
    S_ *= P.row(n - 1).asDiagonal();

    // Update d = a - U * S * U.T
    tmp = U.row(n) * S_;
    d(n) -= tmp * U.row(n).transpose();
    if (d(n) <= 0.0) return n;

    // Update W = (V - U * S) / d
    W.row(n).noalias() -= tmp;
    W.row(n) /= d(n);
  }

  return 0;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7>
void factor_grad(const Eigen::MatrixBase<T1>& U,  // (N, J)
                 const Eigen::MatrixBase<T2>& P,  // (N-1, J)
                 const Eigen::MatrixBase<T3>& d,  // (N)
                 const Eigen::MatrixBase<T1>& W,  // (N, J)
                 const Eigen::MatrixBase<T4>& S,  // (N, J*J)

                 Eigen::MatrixBase<T5>& bU,  // (N, J)
                 Eigen::MatrixBase<T6>& bP,  // (N-1, J)
                 Eigen::MatrixBase<T7>& ba,  // (N)
                 Eigen::MatrixBase<T5>& bV   // (N, J)
) {
  int N = U.rows(), J = U.cols();

  // Make local copies of the gradients that we need.
  typedef Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T1::ColsAtCompileTime,
                        T1::IsRowMajor>
      S_t;
  S_t S_(J, J), bS = S_t::Zero(J, J);
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, 1> bSWT;

  bV.array().colwise() /= d.array();
  for (int n = N - 1; n > 0; --n) {
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < J; ++k) S_(j, k) = S(n, j * J + k);

    // Step 6
    ba(n) -= W.row(n) * bV.row(n).transpose();
    bU.row(n).noalias() = -(bV.row(n) + 2.0 * ba(n) * U.row(n)) * S_ * P.row(n - 1).asDiagonal();
    bS.noalias() -= U.row(n).transpose() * (bV.row(n) + ba(n) * U.row(n));

    // Step 4
    bP.row(n - 1).noalias() = (bS * S_ + S_.transpose() * bS).diagonal();

    // Step 3
    bS = P.row(n - 1).asDiagonal() * bS * P.row(n - 1).asDiagonal();
    bSWT = bS * W.row(n - 1).transpose();
    ba(n - 1) += W.row(n - 1) * bSWT;
    bV.row(n - 1).noalias() += W.row(n - 1) * (bS + bS.transpose());
  }

  bU.row(0).setZero();
  ba(0) -= bV.row(0) * W.row(0).transpose();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void solve(const Eigen::MatrixBase<T1>& U,  // (N, J)
           const Eigen::MatrixBase<T2>& P,  // (N-1, J)
           const Eigen::MatrixBase<T3>& d,  // (N)
           const Eigen::MatrixBase<T1>& W,  // (N, J)
           Eigen::MatrixBase<T4>& Z,        // (N, Nrhs); initially set to Y
           Eigen::MatrixBase<T5>& F,        // (N, J*Nrhs)
           Eigen::MatrixBase<T5>& G         // (N, J*Nrhs)
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, T4::ColsAtCompileTime> F_(J, nrhs);
  F_.setZero();
  F.row(0).setZero();

  for (int n = 1; n < N; ++n) {
    F_.noalias() += W.row(n - 1).transpose() * Z.row(n - 1);
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k) F(n, j * nrhs + k) = F_(j, k);
    F_ = P.row(n - 1).asDiagonal() * F_;
    Z.row(n).noalias() -= U.row(n) * F_;
  }

  Z.array().colwise() /= d.array();

  F_.setZero();
  G.row(N - 1).setZero();
  for (int n = N - 2; n >= 0; --n) {
    F_.noalias() += U.row(n + 1).transpose() * Z.row(n + 1);
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k) G(n, j * nrhs + k) = F_(j, k);
    F_ = P.row(n).asDiagonal() * F_;
    Z.row(n).noalias() -= W.row(n) * F_;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9>
void solve_grad(const Eigen::MatrixBase<T1>& U,   // (N, J)
                const Eigen::MatrixBase<T2>& P,   // (N-1, J)
                const Eigen::MatrixBase<T3>& d,   // (N)
                const Eigen::MatrixBase<T1>& W,   // (N, J)
                const Eigen::MatrixBase<T4>& Z,   // (N, Nrhs)
                const Eigen::MatrixBase<T5>& F,   // (N, J*Nrhs)
                const Eigen::MatrixBase<T5>& G,   // (N, J*Nrhs)
                const Eigen::MatrixBase<T4>& bZ,  // (N, Nrhs)
                Eigen::MatrixBase<T6>& bU,        // (N, J)
                Eigen::MatrixBase<T7>& bP,        // (N-1, J)
                Eigen::MatrixBase<T8>& bd,        // (N)
                Eigen::MatrixBase<T6>& bW,        // (N, J)
                Eigen::MatrixBase<T9>& bY         // (N, Nrhs)
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor>
      Z_ = Z;
  typedef Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, Eigen::RowMajor> F_t;
  F_t F_(J, nrhs), bF = F_t::Zero(J, nrhs);

  bY = bZ;
  for (int n = 0; n <= N - 2; ++n) {
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k) F_(j, k) = G(n, j * nrhs + k);

    // Grad of: Z.row(n).noalias() -= W.row(n) * G;
    bW.row(n).noalias() -= bY.row(n) * (P.row(n).asDiagonal() * F_).transpose();
    bF.noalias() -= W.row(n).transpose() * bY.row(n);

    // Inverse of: Z.row(n).noalias() -= W.row(n) * G;
    Z_.row(n).noalias() += W.row(n) * (P.row(n).asDiagonal() * F_);

    // Grad of: g = P.row(n).asDiagonal() * G;
    bP.row(n).noalias() += (F_ * bF.transpose()).diagonal();
    bF = P.row(n).asDiagonal() * bF;

    // Grad of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    bU.row(n + 1).noalias() += Z_.row(n + 1) * bF.transpose();
    bY.row(n + 1).noalias() += U.row(n + 1) * bF;
  }

  bY.array().colwise() /= d.array();
  bd.array() -= (Z_.array() * bY.array()).rowwise().sum();

  // Inverse of: Z.array().colwise() /= d.array();
  Z_.array().colwise() *= d.array();

  bF.setZero();
  for (int n = N - 1; n >= 1; --n) {
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < nrhs; ++k) F_(j, k) = F(n, j * nrhs + k);

    // Grad of: Z.row(n).noalias() -= U.row(n) * f;
    bU.row(n).noalias() -= bY.row(n) * (P.row(n - 1).asDiagonal() * F_).transpose();
    bF.noalias() -= U.row(n).transpose() * bY.row(n);

    // Grad of: F = P.row(n-1).asDiagonal() * F;
    bP.row(n - 1).noalias() += (F_ * bF.transpose()).diagonal();
    bF = P.row(n - 1).asDiagonal() * bF;

    // Grad of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    bW.row(n - 1).noalias() += Z_.row(n - 1) * bF.transpose();
    bY.row(n - 1).noalias() += W.row(n - 1) * bF;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void conditional_mean(
    const Eigen::MatrixBase<T1>& U,       // (N, J)
    const Eigen::MatrixBase<T1>& V,       // (N, J)
    const Eigen::MatrixBase<T2>& P,       // (N-1, J)
    const Eigen::MatrixBase<T3>& z,       // (N)  ->  The result of a solve
    const Eigen::MatrixBase<T4>& U_star,  // (M, J)
    const Eigen::MatrixBase<T4>& V_star,  // (M, J)
    const Eigen::MatrixBase<T5>& inds,    // (M)  ->  Index where the mth data point should be
                                          // inserted (the output of search_sorted)
    Eigen::MatrixBase<T6>& mu) {
  int N = U.rows(), J = U.cols(), M = U_star.rows();

  Eigen::Matrix<typename T1::Scalar, 1, T1::ColsAtCompileTime> q(1, J);

  // Forward pass
  int m = 0;
  q.setZero();
  while (m < M && inds(m) <= 0) {
    mu(m) = 0;
    ++m;
  }
  for (int n = 0; n < N - 1; ++n) {
    q += z(n) * V.row(n);
    q *= P.row(n).asDiagonal();
    while ((m < M) && (inds(m) <= n + 1)) {
      mu(m) = U_star.row(m) * q.transpose();
      ++m;
    }
  }
  q += z(N - 1) * V.row(N - 1);
  while (m < M) {
    mu(m) = U_star.row(m) * q.transpose();
    ++m;
  }

  // Backward pass
  m = M - 1;
  q.setZero();
  while ((m >= 0) && (inds(m) > N - 1)) {
    --m;
  }
  for (int n = N - 1; n > 0; --n) {
    q += z(n) * U.row(n);
    q *= P.row(n - 1).asDiagonal();
    while ((m >= 0) && (inds(m) > n - 1)) {
      mu(m) += V_star.row(m) * q.transpose();
      --m;
    }
  }
  q += z(0) * U.row(0);
  while (m >= 0) {
    mu(m) = V_star.row(m) * q.transpose();
    --m;
  }
}

}  // namespace celerite
}  // namespace exoplanet

#endif
