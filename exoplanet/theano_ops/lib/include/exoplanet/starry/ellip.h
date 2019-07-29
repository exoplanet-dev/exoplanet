/**
\file ellip.h
\brief Elliptic integral computation.

Elliptic integrals computed following

    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353

adapted from the implementation by E. Agol in
<a href="https://github.com/rodluger/limbdark/">limbdark</a>.
Based in part on Daniel Foreman-Mackey's
<a href="https://github.com/dfm/AstroFlow/">AstroFlow</a>.

*/

#ifndef _EXOPLANET_STARRY_ELLIP_H_
#define _EXOPLANET_STARRY_ELLIP_H_

#include <cmath>
#include "exoplanet/starry/utils.h"

namespace exoplanet {
namespace starry {
namespace ellip {

using std::abs;
using utils::pi;
using utils::mach_eps;

/**
Computes the function `cel(kc, p, a, b)` from Bulirsch (1969)

*/
template <typename T>
T CEL(T ksq, T kc, T p, T a, T b) {
  // In some rare cases, k^2 is so close to zero that it can actually
  // go slightly negative. Let's explicitly force it to zero.
  if (ksq < 0) ksq = 0.0;
  if (kc < 0) kc = 0.0;

  // If k^2 is very small, we get better precision
  // evaluating `kc` like this
  if (ksq < 1e-5) kc = sqrt(1 - ksq);

  // We actually need kc to be nonzero, so let's
  // set it to a very small number
  if ((ksq == 1) || (kc == 0)) kc = mach_eps<T>() * ksq;

  // I haven't encountered cases where k^2 > 1 due to
  // roundoff error, but they could happen. If so, change the
  // line below to avoid an exception
  if (ksq > 1)
    throw std::out_of_range(
        "Elliptic integral `CEL` "
        "was called with `ksq` > 1.");
  T ca = sqrt(mach_eps<T>() * ksq);

  if (ca <= 0) ca = std::numeric_limits<T>::min();
  T m = 1.0;
  T q, g, f, ee;
  ee = kc;

  if (p > 0) {
    p = sqrt(p);
    b /= p;
  } else {
    q = ksq;
    g = 1.0 - p;
    f = g - ksq;
    q *= (b - a * p);
    p = sqrt(f / g);
    a = (a - b) / g;
    b = -q / (g * g * p) + a * p;
  }

  f = a;
  a += b / p;
  g = ee / p;
  b += f * g;
  b += b;
  p += g;
  g = m;
  m += kc;

  for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
    kc = sqrt(ee);
    kc += kc;
    ee = kc * m;
    f = a;
    a += b / p;
    g = ee / p;
    b += f * g;
    b += b;
    p += g;
    g = m;
    m += kc;
    if (abs(g - kc) < g * ca)
      return 0.5 * pi<T>() * (a * m + b) / (m * (m + p));
  }
  throw std::runtime_error("Elliptic integral CEL did not converge.");
}

/**
Computes the function `cel(kc, p, a, b)` from Bulirsch (1969)

*/
template <typename T>
T CEL(T ksq, T p, T a, T b) {
  T kc;
  // Avoid undefined k2=1 case:
  if (ksq != 1.0)
    kc = sqrt(1.0 - ksq);
  else
    kc = mach_eps<T>() * ksq;
  return CEL(ksq, kc, p, a, b);
}

/**
Computes the function `cel(kc, p, a, b)` from Bulirsch (1969).
Vectorized version to improve speed when computing multiple
elliptic integrals with the same value of `kc`.
This assumes first value of a and b uses p; the rest have p = 1.

*/
template <typename T>
inline void CEL(T k2, T kc, T p, T a1, T a2, T a3, T b1, T b2, T b3, T& Piofk,
                T& Eofk, T& Em1mKdm) {
  // Bounds checks
  if (unlikely(k2 > 1))
    throw std::invalid_argument(
        "Invalid value of `k2` passed to `ellip::CEL`.");
  else if (unlikely((k2 == 1.0) || (kc == 0.0)))
    kc = mach_eps<T>() * k2;
  else if (unlikely(k2 < mach_eps<T>()))
    k2 = mach_eps<T>();

  // Tolerance
  T ca = sqrt(mach_eps<T>() * k2);

  // Temporary vars
  T p1, pinv, pinv1, q, g, g1, ginv, f, f1, f2, f3;

  // Initialize values:
  T ee = kc;
  T m = 1.0;
  if (p > 0.0) {
    p = sqrt(p);
    pinv = 1.0 / p;
    b1 *= pinv;
  } else {
    q = k2;
    g = 1.0 - p;
    f = g - k2;
    q *= (b1 - a1 * p);
    ginv = 1.0 / g;
    p = sqrt(f * ginv);
    a1 = (a1 - b1) * ginv;
    pinv = 1.0 / p;
    b1 = -q * ginv * ginv * pinv + a1 * p;
  }
  // Compute recursion:
  f1 = a1;
  // First compute the first integral with p:
  a1 += b1 * pinv;
  g = ee * pinv;
  b1 += f1 * g;
  b1 += b1;
  p += g;
  g = m;
  // Next, compute the remainder with p = 1:
  p1 = 1.0;
  g1 = ee;
  f2 = a2;
  f3 = a3;
  a2 += b2;
  b2 += f2 * g1;
  b2 += b2;
  a3 += b3;
  b3 += f3 * g1;
  b3 += b3;
  p1 += g1;
  g1 = m;
  m += kc;
  size_t iter = 0;
  while (((abs(g - kc) > g * ca) || (abs(g1 - kc) > g1 * ca)) &&
         (iter < STARRY_ELLIP_MAX_ITER)) {
    kc = sqrt(ee);
    kc += kc;
    ee = kc * m;
    f1 = a1;
    f2 = a2;
    f3 = a3;
    pinv = 1.0 / p;
    pinv1 = 1.0 / p1;
    a1 += b1 * pinv;
    a2 += b2 * pinv1;
    a3 += b3 * pinv1;
    g = ee * pinv;
    g1 = ee * pinv1;
    b1 += f1 * g;
    b2 += f2 * g1;
    b3 += f3 * g1;
    b1 += b1;
    b2 += b2;
    b3 += b3;
    p += g;
    p1 += g1;
    g = m;
    m += kc;
    ++iter;
  }
  if (iter == STARRY_ELLIP_MAX_ITER)
    throw std::runtime_error("Elliptic integral CEL did not converge.");
  Piofk = 0.5 * pi<T>() * (a1 * m + b1) / (m * (m + p));
  Eofk = 0.5 * pi<T>() * (a2 * m + b2) / (m * (m + p1));
  Em1mKdm = 0.5 * pi<T>() * (a3 * m + b3) / (m * (m + p1));
}

}  // namespace ellip
}  // namespace starry
}  // namespace exoplanet

#endif
