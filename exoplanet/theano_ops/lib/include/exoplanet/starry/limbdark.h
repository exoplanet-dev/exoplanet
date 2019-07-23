/**
\file limbdark.h
\brief Limb darkening utilities from Agol, Luger & Foreman-Mackey (2019).

\todo Loop downward in `v` until `J[v] != 0`
\todo Test all special cases

*/

#ifndef _EXOPLANET_STARRY_LIMBDARK_H_
#define _EXOPLANET_STARRY_LIMBDARK_H_

#include <cmath>
#include <iostream>
#include "exoplanet/starry/ellip.h"
#include "exoplanet/starry/utils.h"

namespace exoplanet {
namespace starry {
namespace limbdark {

using std::abs;
using std::max;
using namespace utils;

/**
Return (something like the) Wallis ratio,

    Gamma(1 + n / 2) / Gamma(3 / 2 + n / 2)

Computes it recursively. Using double precision, the error is
below 1e-14 well past n = 100.

*/
template <typename T>
inline T wallis(int n) {
  int z, dz;
  if (is_even(n)) {
    z = 1 + n / 2;
    dz = -1;
  } else {
    z = 1 + (n - 1) / 2;
    dz = 0;
  }
  T A = 1.0;
  T B = root_pi<T>();
  for (int i = 1; i < z + dz; ++i) {
    A *= i + 1;
    B *= i - 0.5;
  }
  for (int i = max(1, z + dz); i < z + 1; ++i) B *= i - 0.5;
  if (is_even(n))
    return A / B;
  else
    return B / A;
}

/**
Greens integration housekeeping data.

*/
template <class T>
class GreensLimbDark {
 public:
  // Indices
  int lmax;

  // Basic variables
  T b;
  T r;
  T k;
  T ksq;
  T kc;
  T kcsq;
  T kkc;
  T kap0;
  T kap1;
  T invksq;
  T fourbr;
  T invfourbr;
  T b2;
  T r2;
  T invr;
  T invb;
  T bmr;
  T bpr;
  T onembmr2;
  T onembmr2inv;
  T sqonembmr2;
  T onembpr2;
  T b2mr22;
  T onemr2mb2;
  T sqarea;
  T sqbr;
  T kite_area2;
  T third;
  T Eofk;
  T Em1mKdm;

  // Helper intergrals
  RowVector<T> M;
  RowVector<T> N;
  Matrix<T> M_coeff;
  Matrix<T> N_coeff;

  // Helper arrays
  RowVector<T> n_;
  RowVector<T> invn;
  RowVector<T> ndnp2;

  // The solution vector
  RowVector<T> sT;
  RowVector<T> dsTdb;
  RowVector<T> dsTdr;

  // Constructor
  explicit GreensLimbDark(int lmax)
      : lmax(lmax),
        M(lmax + 1),
        N(lmax + 1),
        M_coeff(4, STARRY_MN_MAX_ITER),
        N_coeff(2, STARRY_MN_MAX_ITER),
        n_(lmax + 3),
        invn(lmax + 3),
        ndnp2(lmax + 3),
        sT(RowVector<T>::Zero(lmax + 1)),
        dsTdb(RowVector<T>::Zero(lmax + 1)),
        dsTdr(RowVector<T>::Zero(lmax + 1)) {
    // Constants
    computeMCoeff();
    computeNCoeff();
    third = T(1.0) / T(3.0);
    for (int n = 0; n < lmax + 3; ++n) {
      n_(n) = n;
      invn(n) = T(1.0) / n;
      ndnp2(n) = n / (n + 2.0);
    }
  }

  template <bool GRADIENT = false>
  inline void computeS1();

  inline void computeMCoeff();

  inline void computeM0123();

  inline void upwardM();

  inline void downwardM();

  inline void computeNCoeff();

  inline void computeN01();

  inline void upwardN();

  inline void downwardN();

  template <bool GRADIENT = false>
  inline void compute(const T& b_, const T& r_);
};

/**
The linear limb darkening flux term.

*/
template <class T>
template <bool GRADIENT>
inline void GreensLimbDark<T>::computeS1() {
  T Lambda1 = 0;
  if ((b >= 1.0 + r) || (r == 0.0)) {
    // No occultation (Case 1)
    Lambda1 = 0;
    if (GRADIENT) {
      dsTdb(1) = 0;
      dsTdr(1) = 0;
    }
    Eofk = 0;     // Check
    Em1mKdm = 0;  // Check
  } else if (b <= r - 1.0) {
    // Full occultation (Case 11)
    Lambda1 = 0;
    if (GRADIENT) {
      dsTdb(1) = 0;
      dsTdr(1) = 0;
    }
    Eofk = 0;     // Check
    Em1mKdm = 0;  // Check
  } else {
    if (unlikely(b == 0)) {
      // Case 10
      T sqrt1mr2 = sqrt(1.0 - r2);
      Lambda1 = -2.0 * pi<T>() * sqrt1mr2 * sqrt1mr2 * sqrt1mr2;
      Eofk = 0.5 * pi<T>();
      Em1mKdm = 0.25 * pi<T>();
      if (GRADIENT) {
        dsTdb(1) = 0;
        dsTdr(1) = -2.0 * pi<T>() * r * sqrt1mr2;
      }
    } else if (unlikely(b == r)) {
      if (unlikely(r == 0.5)) {
        // Case 6
        Lambda1 = pi<T>() - 4.0 * third;
        Eofk = 1.0;
        Em1mKdm = 1.0;
        if (GRADIENT) {
          dsTdb(1) = 2.0 * third;
          dsTdr(1) = -2.0;
        }
      } else if (r < 0.5) {
        // Case 5
        T m = 4 * r2;
        Eofk = ellip::CEL(m, T(1.0), T(1.0), T(1.0 - m));
        Em1mKdm = ellip::CEL(m, T(1.0), T(1.0), T(0.0));
        Lambda1 = pi<T>() + 2.0 * third * ((2 * m - 3) * Eofk - m * Em1mKdm);
        if (GRADIENT) {
          dsTdb(1) = -4.0 * r * third * (Eofk - 2 * Em1mKdm);
          dsTdr(1) = -4.0 * r * Eofk;
        }
      } else {
        // Case 7
        T m = 4 * r2;
        T minv = T(1.0) / m;
        Eofk = ellip::CEL(minv, T(1.0), T(1.0), T(1.0 - minv));
        Em1mKdm = ellip::CEL(minv, T(1.0), T(1.0), T(0.0));
        Lambda1 = pi<T>() + third * invr * (-m * Eofk + (2 * m - 3) * Em1mKdm);
        if (GRADIENT) {
          dsTdb(1) = 2 * third * (2 * Eofk - Em1mKdm);
          dsTdr(1) = -2 * Em1mKdm;
        }
      }
    } else {
      if (ksq < 1) {
        // Case 2, Case 8
        T sqbrinv = T(1.0) / sqbr;
        T Piofk;
        ellip::CEL(ksq, kc, T((b - r) * (b - r) * kcsq), T(0.0), T(1.0), T(1.0),
                   T(3 * kcsq * (b - r) * (b + r)), kcsq, T(0.0), Piofk, Eofk,
                   Em1mKdm);
        Lambda1 = onembmr2 * (Piofk + (-3 + 6 * r2 + 2 * b * r) * Em1mKdm -
                              fourbr * Eofk) *
                  sqbrinv * third;
        if (GRADIENT) {
          dsTdb(1) = 2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) * sqbrinv * third;
          dsTdr(1) = -2 * r * onembmr2 * Em1mKdm * sqbrinv;
        }
      } else if (ksq > 1) {
        // Case 3, Case 9
        T bmrdbpr = (b - r) / (b + r);
        T mu = 3 * bmrdbpr * onembmr2inv;
        T p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv;
        T Piofk;
        ellip::CEL(invksq, kc, p, T(1 + mu), T(1.0), T(1.0), T(p + mu), kcsq,
                   T(0.0), Piofk, Eofk, Em1mKdm);
        Lambda1 = 2 * sqonembmr2 *
                  (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) * third;
        if (GRADIENT) {
          dsTdb(1) = -4 * r * third * sqonembmr2 * (Eofk - 2 * Em1mKdm);
          dsTdr(1) = -4 * r * sqonembmr2 * Eofk;
        }
      } else {
        // Case 4
        T rootr1mr = sqrt(r * (1 - r));
        Lambda1 = 2 * acos(1.0 - 2.0 * r) -
                  4 * third * (3 + 2 * r - 8 * r2) * rootr1mr -
                  2 * pi<T>() * int(r > 0.5);
        Eofk = 1.0;
        Em1mKdm = 1.0;
        if (GRADIENT) {
          dsTdr(1) = -8 * r * rootr1mr;
          dsTdb(1) = -dsTdr(1) * third;
        }
      }
    }
  }
  sT(1) = ((1.0 - int(r > b)) * 2 * pi<T>() - Lambda1) * third;
}

/**
Compute the coefficients of the series expansion
for the highest four terms of the `M` integral.

*/
template <class T>
inline void GreensLimbDark<T>::computeMCoeff() {
  T coeff;
  int n;

  // ksq < 1
  for (int j = 0; j < 4; ++j) {
    n = lmax - 3 + j;
    coeff = root_pi<T>() * wallis<T>(n);

    // Add leading term to M
    M_coeff(j, 0) = coeff;
    // Now, compute higher order terms until
    // desired precision is reached
    for (int i = 1; i < STARRY_MN_MAX_ITER; ++i) {
      coeff *= T((2 * i - 1) * (2 * i - 1)) / T(2 * i * (1 + n + 2 * i));
      M_coeff(j, i) = coeff;
    }
  }
}

/**
Compute the first four terms of the M integral.

*/
template <class T>
inline void GreensLimbDark<T>::computeM0123() {
  if (ksq < 1.0) {
    M(0) = kap0;
    M(1) = 2 * sqbr * 2 * ksq * Em1mKdm;
    M(2) = kap0 * onemr2mb2 + kite_area2;
    M(3) = (2.0 * sqbr) * (2.0 * sqbr) * (2.0 * sqbr) * 2.0 * third * ksq *
           (Eofk + (3.0 * ksq - 2.0) * Em1mKdm);
  } else {
    M(0) = pi<T>();
    M(1) = 2.0 * sqonembmr2 * Eofk;
    M(2) = pi<T>() * onemr2mb2;
    M(3) = sqonembmr2 * sqonembmr2 * sqonembmr2 * 2.0 * third *
           ((3.0 - 2.0 * invksq) * Eofk + invksq * Em1mKdm);
  }
}

/**
Compute the terms in the M integral by upward recursion.

*/
template <class T>
inline void GreensLimbDark<T>::upwardM() {
  // Compute lowest four exactly
  computeM0123();

  // Recurse upward
  for (int n = 4; n < lmax + 1; ++n)
    M(n) =
        (2.0 * (n - 1) * onemr2mb2 * M(n - 2) + (n - 2) * sqarea * M(n - 4)) *
        invn(n);
}

/**
Compute the terms in the M integral by downward recursion.

*/
template <class T>
inline void GreensLimbDark<T>::downwardM() {
  T val, k2n, tol, fac, term;
  T invsqarea = T(1.0) / sqarea;

  // Compute highest four using a series solution
  if (ksq < 1) {
    // Compute leading coefficient (n=0)
    tol = mach_eps<T>() * ksq;
    term = 0.0;
    fac = 1.0;
    for (int n = 0; n < lmax - 3; ++n) fac *= sqonembmr2;
    fac *= k;

    // Now, compute higher order terms until
    // desired precision is reached
    for (int j = 0; j < 4; ++j) {
      // Add leading term to M
      val = M_coeff(j, 0);
      k2n = 1.0;

      // Compute higher order terms
      for (int n = 1; n < STARRY_MN_MAX_ITER; ++n) {
        k2n *= ksq;
        term = k2n * M_coeff(j, n);
        val += term;
        if (abs(term) < tol) break;
      }
      M(lmax - 3 + j) = val * fac;
      fac *= sqonembmr2;
    }

  } else {
    throw std::runtime_error(
        "Downward recursion in `M` not implemented for `k^2` >= 1.");
  }

  // Recurse downward
  for (int n = lmax - 4; n > 3; --n)
    M(n) = ((n + 4) * M(n + 4) - 2.0 * (n + 3) * onemr2mb2 * M(n + 2)) *
           invsqarea * invn(n + 2);

  // Compute lowest four exactly
  computeM0123();
}

/**
Compute the coefficients of the series expansion
for the highest two terms of the `N` integral.

*/
template <class T>
inline void GreensLimbDark<T>::computeNCoeff() {
  T coeff = 0.0;
  int n;

  // ksq < 1
  for (int j = 0; j < 2; ++j) {
    n = lmax - 1 + j;

    // Add leading term to N
    coeff = root_pi<T>() * wallis<T>(n) / (n + 3.0);
    N_coeff(j, 0) = coeff;

    // Now, compute higher order terms until
    // desired precision is reached
    for (int i = 1; i < STARRY_MN_MAX_ITER; ++i) {
      coeff *= T(4 * i * i - 1) / T(2 * i * (3 + n + 2 * i));
      N_coeff(j, i) = coeff;
    }
  }
}

/**
Compute the first two terms of the N integral.

*/
template <class T>
inline void GreensLimbDark<T>::computeN01() {
  if (ksq <= 1.0) {
    N(0) = 0.5 * kap0 - k * kc;
    N(1) = 4.0 * third * sqbr * ksq * (-Eofk + 2.0 * Em1mKdm);
  } else {
    N(0) = 0.5 * pi<T>();
    N(1) = 4.0 * third * sqbr * k * (2.0 * Eofk - Em1mKdm);
  }
}

/**
Compute the terms in the N integral by upward recursion.

*/
template <class T>
inline void GreensLimbDark<T>::upwardN() {
  // Compute lowest two exactly
  computeN01();

  // Recurse upward
  for (int n = 2; n < lmax + 1; ++n)
    N(n) = (M(n) + n * onembpr2 * N(n - 2)) * invn(n + 2);
}

/**
Compute the terms in the N integral by downward recursion.

*/
template <class T>
inline void GreensLimbDark<T>::downwardN() {
  // Compute highest two using a series solution
  if (ksq < 1) {
    // Compute leading coefficient (n=0)
    T val, k2n;
    T tol = mach_eps<T>() * ksq;
    T term = 0.0;
    T fac = 1.0;
    for (int n = 0; n < lmax - 1; ++n) fac *= sqonembmr2;
    fac *= k * ksq;

    // Now, compute higher order terms until
    // desired precision is reached
    for (int j = 0; j < 2; ++j) {
      val = N_coeff(j, 0);
      k2n = 1.0;
      for (int n = 1; n < STARRY_MN_MAX_ITER; ++n) {
        k2n *= ksq;
        term = k2n * N_coeff(j, n);
        val += term;
        if (abs(term) < tol) break;
      }
      N(lmax - 1 + j) = val * fac;
      fac *= sqonembmr2;
    }

  } else {
    throw std::runtime_error(
        "Downward recursion in `N` not implemented for `k^2` >= 1.");
  }

  // Recurse downward
  T onembpr2inv = T(1.0) / onembpr2;
  for (int n = lmax - 2; n > 1; --n)
    N(n) = ((n + 4) * N(n + 2) - M(n + 2)) * onembpr2inv * invn(n + 2);

  // Compute lowest two exactly
  computeN01();
}

/**
Compute the `s^T` occultation solution vector

*/
template <class T>
template <bool GRADIENT>
inline void GreensLimbDark<T>::compute(const T& b_, const T& r_) {
  // Initialize the basic variables
  b = b_;
  r = r_;

  // HACK: Fix an instability that exists *really* close to b = r = 0.5
  if (unlikely(abs(b - r) < 5 * mach_eps<T>())) {
    if (unlikely(abs(r - T(0.5)) < 5 * mach_eps<T>())) {
      b += 5 * mach_eps<T>();
    }
  }

  // Special case: complete occultation
  if (unlikely(b < r - 1)) {
    sT.setZero();
    if (GRADIENT) {
      dsTdb.setZero();
      dsTdr.setZero();
    }
    return;
  }

  // Special case: no occultation
  if (unlikely(r == 0) || (b > r + 1)) {
    sT.setZero();
    sT(0) = pi<T>();
    sT(1) = 2.0 * pi<T>() / 3.0;
    if (GRADIENT) {
      dsTdb.setZero();
      dsTdr.setZero();
    }
    return;
  }

  b2 = b * b;
  r2 = r * r;
  invr = T(1.0) / r;
  invb = T(1.0) / b;
  bmr = b - r;
  bpr = b + r;
  fourbr = 4 * b * r;
  invfourbr = 0.25 * invr * invb;
  onembmr2 = (1.0 + bmr) * (1.0 - bmr);
  onembmr2inv = T(1.0) / onembmr2;
  onembpr2 = (1.0 + bpr) * (1.0 - bpr);
  sqonembmr2 = sqrt(onembmr2);
  b2mr22 = (b2 - r2) * (b2 - r2);
  onemr2mb2 = (1.0 - r) * (1.0 + r) - b2;
  sqbr = sqrt(b * r);

  // Compute the kite area and the k^2 variables
  T p0 = T(1.0), p1 = b, p2 = r;
  if (p0 < p1) swap(p0, p1);
  if (p1 < p2) swap(p1, p2);
  if (p0 < p1) swap(p0, p1);
  sqarea =
      (p0 + (p1 + p2)) * (p2 - (p0 - p1)) * (p2 + (p0 - p1)) * (p0 + (p1 - p2));
  kite_area2 = sqrt(max(T(0.0), sqarea));

  if (unlikely((b == 0) || (r == 0))) {
    ksq = T(INFINITY);
    k = T(INFINITY);
    kc = 1;
    kcsq = 1;
    kkc = T(INFINITY);
    invksq = 0;
    kap0 = 0;  // Not used!
    kap1 = 0;  // Not used!
    sT(0) = pi<T>() * (1 - r2);
    if (GRADIENT) {
      dsTdb(0) = 0;
      dsTdr(0) = -2 * pi<T>() * r;
    }
  } else {
    ksq = onembpr2 * invfourbr + 1.0;
    invksq = T(1.0) / ksq;
    k = sqrt(ksq);
    if (ksq > 1) {
      kcsq = onembpr2 * onembmr2inv;
      kc = sqrt(kcsq);
      kkc = k * kc;
      kap0 = 0;  // Not used!
      kap1 = 0;  // Not used!
      sT(0) = pi<T>() * (1 - r2);
      if (GRADIENT) {
        dsTdb(0) = 0;
        dsTdr(0) = -2 * pi<T>() * r;
      }
    } else {
      kcsq = -onembpr2 * invfourbr;
      kc = sqrt(kcsq);
      kkc = kite_area2 * invfourbr;
      kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b2);
      kap1 = atan2(kite_area2, (1 - r) * (1 + r) + b2);
      T Alens = kap1 + r2 * kap0 - kite_area2 * 0.5;
      sT(0) = pi<T>() - Alens;
      if (GRADIENT) {
        dsTdb(0) = kite_area2 * invb;
        dsTdr(0) = -2.0 * r * kap0;
      }
    }
  }

  // Special case
  if (unlikely(lmax == 0)) return;

  // Compute the linear limb darkening term
  // and the elliptic integrals
  computeS1<GRADIENT>();

  // Special case
  if (unlikely(lmax == 1)) return;

  // Special case
  if (unlikely(b == 0)) {
    T term = 1 - r2;
    T dtermdr = -2 * r;
    T fac = sqrt(term);
    T dfacdr = -r / fac;
    for (int n = 2; n < lmax + 1; ++n) {
      sT(n) = -term * r2 * 2 * pi<T>();
      if (GRADIENT) {
        dsTdb(n) = 0;
        dsTdr(n) = -2 * pi<T>() * r * (2 * term + r * dtermdr);
        dtermdr = dfacdr * term + fac * dtermdr;
      }
      term *= fac;
    }
    return;
  }

  // Compute the quadratic term
  T r2pb2 = (r2 + b2);
  T eta2 = 0.5 * r2 * (r2pb2 + b2);
  T four_pi_eta;
  T detadb, detadr;
  if (ksq > 1) {
    four_pi_eta = 4 * pi<T>() * (eta2 - 0.5);
    if (GRADIENT) {
      T deta2dr = 2 * r * r2pb2;
      T deta2db = 2 * b * r2;
      detadr = 4 * pi<T>() * deta2dr;
      detadb = 4 * pi<T>() * deta2db;
    }
  } else {
    four_pi_eta = 2 * (-(pi<T>() - kap1) + 2 * eta2 * kap0 -
                       0.25 * kite_area2 * (1.0 + 5 * r2 + b2));
    if (GRADIENT) {
      detadr = 8 * r * (r2pb2 * kap0 - kite_area2);
      detadb = 2.0 * invb * (4 * b2 * r2 * kap0 - (1 + r2pb2) * kite_area2);
    }
  }
  sT(2) = 2 * sT(0) + four_pi_eta;
  if (GRADIENT) {
    dsTdb(2) = 2 * dsTdb(0) + detadb;
    dsTdr(2) = 2 * dsTdr(0) + detadr;
  }

  if (lmax == 2) return;

  // Now onto the higher order terms...
  if ((ksq < 0.5) && (lmax > 3))
    downwardM();
  else
    upwardM();

  // Compute the remaining terms in the `sT` vector
  sT.segment(3, lmax - 2) =
      -2.0 * r2 * M.segment(3, lmax - 2) +
      ndnp2.segment(3, lmax - 2)
          .cwiseProduct(onemr2mb2 * M.segment(3, lmax - 2) +
                        sqarea * M.segment(1, lmax - 2));

  // Compute gradients
  if (GRADIENT) {
    // Compute ds/dr
    dsTdr.segment(3, lmax - 2) =
        -2 * r * (n_.segment(5, lmax - 2).cwiseProduct(M.segment(3, lmax - 2)) -
                  n_.segment(3, lmax - 2).cwiseProduct(M.segment(1, lmax - 2)));

    if (b > STARRY_BCUT) {
      // Compute ds/db
      dsTdb.segment(3, lmax - 2) =
          (-invb * n_.segment(3, lmax - 2))
              .cwiseProduct((r2 + b2) * (M.segment(3, lmax - 2) -
                                         M.segment(1, lmax - 2)) +
                            b2mr22 * M.segment(1, lmax - 2));
    } else {
      // Compute ds/db using the small b reparametrization
      T r3 = r2 * r;
      T b3 = b2 * b;
      if ((ksq < 0.5) && (lmax > 3))
        downwardN();
      else
        upwardN();
      dsTdb.segment(3, lmax - 2) =
          -n_.segment(3, lmax - 2)
               .cwiseProduct((2.0 * r3 + b3 - b - 3.0 * r2 * b) *
                                 M.segment(1, lmax - 2) +
                             b * M.segment(3, lmax - 2) -
                             4.0 * r3 * N.segment(1, lmax - 2));
    }
  }
}

}  // namespace limbdark
}  // namespace starry
}  // namespace exoplanet

#endif
