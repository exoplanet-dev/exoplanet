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
      static const T eps = 4*std::numeric_limits<T>::epsilon();
      T rmz = r - z,
        rpz = r + z,
        x2 = x*x,
        r2 = r*r;

      if (fabs(rmz) + eps < x && x < rpz - eps) {
        T z2 = z*z;
        T u = fmax(fmin(0.5 * (z2 + x2 - r2) / (z * x), 1.0), -1.0);
        T v = fmax(fmin(0.5 * (z2 + r2 - x2) / (z * r), 1.0), -1.0);
        T w = fmax((x + rmz) * (x - rmz) * (rpz - x) * (rpz + x), 0.0);
        return x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w);
      } else if (x >= rpz - eps) {
        return M_PI * r2;
      } else if (x <= rmz + eps) {
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
      static const T eps = 4*std::numeric_limits<T>::epsilon();
      T rmz = r - z,
        rpz = r + z,
        x2 = x*x,
        r2 = r*r;

      *d_z = 0.0;
      *d_r = 0.0;

      if (fabs(rmz) + eps < x && x < rpz - eps) {
        T z2 = z*z;
        T zx = z * x;
        T zr = z * r;
        T u = fmax(fmin(0.5 * (z2 + x2 - r2) / zx, 1.0), -1.0);
        T v = fmax(fmin(0.5 * (z2 + r2 - x2) / zr, 1.0), -1.0);
        T w = (x + rmz) * (x - rmz) * (rpz - x) * (rpz + x);
        if (w < 0.0) w = 0.0;


        // Some shortcuts
        T acosu = (fabs(u - 1.0) <= eps) ? 0.0 : acos(u);
        T acosv = (fabs(v - 1.0) <= eps) ? 0.0 : acos(v);
        T sqrtw = sqrt(w);

        // Compute all the partials
        T dudz = 0.5 * (1.0 / x - (x2 - r2) / (z2 * x));
        T dudr = -r / zx;

        T dvdz = 0.5 * (1.0 / r - (r2 - x2) / (z2 * r));
        T dvdr = 0.5 * (1.0 / z - (z2 - x2) / (r2 * z));

        T dwdz = 4.0 * z * (r2 - z2 + x2);
        T dwdr = 4.0 * r * (z2 - r2 + x2);

        // FIXME: what happens when u == 1 or v == 1?
        T dacosu = -1.0 / sqrt(1.0 - u * u);
        T dacosv = -1.0 / sqrt(1.0 - v * v);

        // Combine the partials as needed
        *d_z = x2 * dacosu * dudz + r2 * dacosv * dvdz - 0.25 * dwdz / sqrtw;
        *d_r = x2 * dacosu * dudr + 2.0 * r * acosv + r2 * dacosv * dvdr - 0.25 * dwdr / sqrtw;

        return x2 * acosu + r2 * acosv - 0.5 * sqrt(w);
      } else if (x >= rpz - eps) {
        *d_r = 2.0 * M_PI * r;
        return M_PI * r2;
      } else if (x <= rmz + eps) {
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

      T d_z, d_r;

      T A2 = compute_area_fwd<T>(radius[n_max], r, z, &d_r, &d_z), A1,
        I2 = intensity[n_max], I1, I, b_I;
      for (int n = n_max-1; n >= n_min; --n) {
        I1 = intensity[n];
        I = 0.5 * (I1 + I2);
        *b_z += b_delta * I * d_z;
        *b_r += b_delta * I * d_r;

        A1 = compute_area_fwd(radius[n], r, z, &d_r, &d_z);

        *b_z -= b_delta * I * d_z;
        *b_r -= b_delta * I * d_r;

        b_I = 0.5 * b_delta * (A2 - A1);
        b_intensity[n+1] += b_I;
        b_intensity[n]   += b_I;

        A2 = A1;
        I2 = I1;
      }
    }

  };
};

#endif
