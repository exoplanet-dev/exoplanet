#ifndef _TRANSIT_TRANSIT_H_
#define _TRANSIT_TRANSIT_H_
#include <cmath>

namespace transit {

#ifdef __CUDACC__
#define TRANSIT_CUDA_CALLABLE __host__ __device__
#else
#define TRANSIT_CUDA_CALLABLE
#endif

  template <typename T>
  TRANSIT_CUDA_CALLABLE
  inline T index_to_coord (int size, int index)
  {
    T val = 1.0 - T(index) / (size - 1);
    return 1.0 - val * val;
  }

  template <typename T>
  TRANSIT_CUDA_CALLABLE
  inline T coord_to_index (int size, T coord)
  {
    return (size - 1) * (1.0 - sqrt(1.0 - coord));
  }

  template <typename T>
  TRANSIT_CUDA_CALLABLE
  inline T compute_area (T x, T z, T r)
  {
    T rmz = r - z,
      rpz = r + z,
      x2 = x*x,
      r2 = r*r;

    if (fabs(rmz) < x && x < rpz) {
      T z2 = z*z;
      T u = 0.5 * (z2 + x2 - r2) / (z * x);
      T v = 0.5 * (z2 + r2 - x2) / (z * r);
      T w = fmax((x + rmz) * (x - rmz) * (rpz - x) * (rpz + x), 0.0);
      return x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w);
    } else if (x >= rpz) {
      return M_PI * r2;
    } else if (x <= rmz) {
      return M_PI * x2;
    }
    return 0.0;
  }

  template <typename T>
  TRANSIT_CUDA_CALLABLE
  inline T compute_delta (
      int                         grid_size,
      const T* __restrict__ const x,
      const T* __restrict__ const grid,
      int                         indmin,
      int                         indmax,
      T                           z,
      T                           r)
  {
    if (z - r >= 1.0) return 0.0;

    T delta = 0.0;
    T A1 = 0.0;
    T I1 = grid[indmin];
    T A2, I2;
    for (int ind = indmin+1; ind <= indmax; ++ind) {
      A2 = compute_area<T>(x[ind], z, r);
      I2 = grid[ind];
      delta += 0.5 * (I1 + I2) * (A2 - A1);
      A1 = A2;
      I1 = I2;
    }
    return delta;
  }

  template <typename T>
  TRANSIT_CUDA_CALLABLE
  inline T compute_area_fwd (T x, T z, T r, T* d_z, T* d_r)
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
      T u = 0.5 * (z2 + x2 - r2) / zx;
      T v = 0.5 * (z2 + r2 - x2) / zr;
      T w = (x + rmz) * (x - rmz) * (rpz - x) * (rpz + x);
      if (w < 0.0) w = 0.0;

      // Some shortcuts
      T acosv = acos(v);
      T sqrtw = sqrt(w);

      // Compute all the partials
      T dudz = 0.5 * (1.0 / x - (x2 - r2) / (z2 * x));
      T dudr = -r / zx;

      T dvdz = 0.5 * (1.0 / r - (r2 - x2) / (z2 * r));
      T dvdr = 0.5 * (1.0 / z - (z2 - x2) / (r2 * z));

      T dwdz = 4.0 * z * (r2 - z2 + x2);
      T dwdr = 4.0 * r * (z2 - r2 + x2);

      T dacosu = -1.0 / sqrt(1.0 - u * u);
      T dacosv = -1.0 / sqrt(1.0 - v * v);

      // Combine the partials as needed
      *d_z = x2 * dacosu * dudz + r2 * dacosv * dvdz - 0.25 * dwdz / sqrtw;
      *d_r = x2 * dacosu * dudr + 2.0 * r * acosv + r2 * dacosv * dvdr - 0.25 * dwdr / sqrtw;

      return x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w);
    } else if (x >= rpz) {
      *d_r = 2.0 * M_PI * r;
      return M_PI * r2;
    } else if (x <= rmz) {
      return M_PI * x2;
    }
    return 0.0;
  }

  template <typename T>
  TRANSIT_CUDA_CALLABLE
  inline void compute_delta_rev (
      int                         grid_size,
      const T* __restrict__ const x,
      const T* __restrict__ const grid,
      int                         indmin,
      int                         indmax,
      T                           z,
      T                           r,
      T                           b_delta,
      // Outputs
      T*                          b_grid,
      T*                          b_z,
      T*                          b_r)
  {
    if (z - r >= 1.0) return;

    T d_z, d_r;
    T A2 = compute_area_fwd(x[indmax], z, r, &d_z, &d_r);
    T I2 = grid[indmax];
    T A1, I1, I, b_I;
    for (int ind = indmax-1; ind >= indmin; --ind) {
      I1 = grid[ind];
      I = 0.5 * (I1 + I2);
      *b_z += b_delta * I * d_z;
      *b_r += b_delta * I * d_r;

      A1 = compute_area_fwd(x[ind], z, r, &d_z, &d_r);

      *b_z -= b_delta * I * d_z;
      *b_r -= b_delta * I * d_r;

      b_I = 0.5 * b_delta * (A2 - A1);
      b_grid[ind+1] += b_I;
      b_grid[ind]   += b_I;

      A2 = A1;
      I2 = I1;
    }
  }

#undef TRANSIT_CUDA_CALLABLE

};

#endif
