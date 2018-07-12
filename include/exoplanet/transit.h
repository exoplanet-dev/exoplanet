#ifndef _EXOPLANET_TRANSIT_H_
#define _EXOPLANET_TRANSIT_H_

#include <cmath>
#include <limits>

#include "exoplanet/cuda_utils.h"

namespace exoplanet {
  namespace transit {

    //
    // This module computes a transit light curve for any limb darkening
    // profile
    //

    ///
    /// Compute the overlapping area of two disks
    ///
    /// @param x  The radius of one of the disks
    /// @param r  The radius of the second disk
    /// @param z  The distance between the circle centers
    ///
    /// @return A  The overlapping area
    ///
    template <typename T>
    EXOPLANET_CUDA_CALLABLE
    inline T compute_area (T x, T r, T z)
    {
      T rmz = r - z,
        rpz = r + z,
        x2 = x*x,
        r2 = r*r;

      if (fabs(rmz) < x && x < rpz) {
        T z2 = z*z;
        T u = fmax(fmin(T(0.5) * (z2 + x2 - r2) / (z * x), T(1.0)), T(-1.0));
        T v = fmax(fmin(T(0.5) * (z2 + r2 - x2) / (z * r), T(1.0)), T(-1.0));
        T w = fmax((x + rmz) * (x - rmz) * (rpz - x) * (rpz + x), T(0.0));
        return x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w);
      } else if (x >= rpz) {
        return M_PI * r2;
      } else if (x <= rmz) {
        return M_PI * x2;
      }
      return 0.0;
    }

    ///
    /// Numerically integrate the limb darkening model to compute the depth
    /// of a transit with a given configuration
    ///
    /// @param N          The number of rings defining the stellar surface
    /// @param radius     The precomputed radii where the stellar intensity is
    ///                   defined
    /// @param intensity  The brightness of the stellar surface defined at the
    ///                   radii in ``radius``
    /// @param n_min      The index where ``radius[n_min]`` is the lower
    ///                   integration bound (there is assumed to be no
    ///                   occultation for ``radius < radius[n_min]``)
    /// @param n_max      The index where ``radius[n_max]`` is the upper
    ///                   integration bound
    /// @param z          The projected sky distance between the center of
    ///                   the star and the transiting body
    /// @param r          The radius of the transiting body in units of the
    ///                   stellar radius
    ///
    /// @return delta     The transit depth for this configuration
    ///
    template <typename T>
    EXOPLANET_CUDA_CALLABLE
    inline T compute_transit_depth (
        int                         N,
        const T* __restrict__ const radius,
        const T* __restrict__ const intensity,
        int                         n_min,
        int                         n_max,
        T                           z,
        T                           r)
    {
      if (z - r >= 1.0) return 0.0;

      T delta = 0.0,
        A1 = 0.0, A2,
        I1 = intensity[n_min], I2;
      for (int n = n_min+1; n <= n_max; ++n) {
        A2 = compute_area<T>(radius[n], r, z);
        I2 = intensity[n];
        delta += 0.5 * (I1 + I2) * (A2 - A1);
        A1 = A2;
        I1 = I2;
      }
      return delta;
    }

    ///
    /// Compute the overlapping area of two disks and its gradient
    ///
    /// @param x    The radius of one of the disks
    /// @param r    The radius of the second disk
    /// @param z    The distance between the circle centers
    /// @param d_r  (output) The gradient of A with respect to r
    /// @param d_z  (output) The gradient of A with respect to z
    ///
    /// @return A  The overlapping area
    ///
    template <typename T>
    EXOPLANET_CUDA_CALLABLE
    inline T compute_area_fwd (T x, T r, T z, T* d_r, T* d_z)
    {
      T rmz = r - z,
        rpz = r + z,
        x2 = x*x,
        r2 = r*r;

      *d_z = 0.0;
      *d_r = 0.0;

      if (fabs(rmz) < x && x < rpz) {
        T z2 = z*z;
        T zx = z * x;
        T zr = z * r;
        T u = fmax(fmin(0.5 * (z2 + x2 - r2) / zx, T(1.0)), T(-1.0));
        T v = fmax(fmin(0.5 * (z2 + r2 - x2) / zr, T(1.0)), T(-1.0));
        T w = fmax((x + rmz) * (x - rmz) * (rpz - x) * (rpz + x), T(0.0));

        T acosu = acos(u);
        T acosv = acos(v);
        T sqrtw = sqrt(w);

        *d_z = -sqrtw / z;
        *d_r = 2.0 * r * acosv;

        return x2 * acosu + r2 * acosv - 0.5 * sqrtw;
      } else if (x >= rpz) {
        *d_r = 2.0 * M_PI * r;
        return M_PI * r2;
      } else if (x <= rmz) {
        return M_PI * x2;
      }
      return 0.0;
    }

    ///
    /// Compute the reverse-mode gradient of ``compute_transit_depth``
    ///
    /// @param N          The number of rings defining the stellar surface
    /// @param radius     The precomputed radii where the stellar intensity is
    ///                   defined
    /// @param intensity  The brightness of the stellar surface defined at the
    ///                   radii in ``radius``
    /// @param n_min      The index where ``radius[n_min]`` is the lower
    ///                   integration bound (there is assumed to be no
    ///                   occultation for ``radius < radius[n_min]``)
    /// @param n_max      The index where ``radius[n_max]`` is the upper
    ///                   integration bound
    /// @param z          The projected sky distance between the center of
    ///                   the star and the transiting body
    /// @param r          The radius of the transiting body in units of the
    ///                   stellar radius
    /// @param b_delta    The reverse-mode gradient of delta ($\bar{delta}$)
    /// @param b_z        (output) The propagated reverse mode gradient for
    ///                   ``z``
    /// @param b_r        (output) The propagated reverse mode gradient for
    ///                   ``r``
    ///
    template <typename T>
    EXOPLANET_CUDA_CALLABLE
    inline void compute_transit_depth_rev (
        int                         N,
        const T* __restrict__ const radius,
        const T* __restrict__ const intensity,
        int                         n_min,
        int                         n_max,
        T                           z,
        T                           r,
        T                           b_delta,
        // Outputs
        T*                          b_intensity,
        T*                          b_z,
        T*                          b_r)
    {
      if (z - r >= 1.0) return;

      T A0 = 0.0, A1 = 0.0, A2 = 0.0,
        I1 = intensity[n_min], I2;
      T dA1_dr = 0.0, dA2_dr, dA1_dz = 0.0, dA2_dz, I;
      for (int n = n_min+1; n <= n_max; ++n) {
        A2 = compute_area_fwd<T>(radius[n], r, z, &dA2_dr, &dA2_dz);
        I2 = intensity[n];

        b_intensity[n-1] += 0.5 * b_delta * (A2 - A0);

        I = 0.5 * b_delta * (I1 + I2);
        *b_z += I * (dA2_dz - dA1_dz);
        *b_r += I * (dA2_dr - dA1_dr);

        A0 = A1;
        A1 = A2;
        I1 = I2;
        dA1_dz = dA2_dz;
        dA1_dr = dA2_dr;
      }
      b_intensity[n_max] += 0.5 * b_delta * (A2 - A0);
    }
  };
};

#endif
