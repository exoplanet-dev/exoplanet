#include <iostream>

#include "transit/transit.h"
#include "transit/limb_darkening.h"

/*#ifdef __CUDACC__*/
/*#define TRANSIT_CUDA_CALLABLE __host__ __device__*/
/*#endif*/


/*template <typename T>*/
/*struct QuadraticLimbDarkening {*/

/*  public:*/
/*    QuadraticLimbDarkening (T c1, T c2) : c1_(c1), c2_(c2) {*/
/*      I0_ = M_PI * (1.0 - (2.0 * c1_ + c2) / 6.0);*/
/*    }*/

/*    TRANSIT_CUDA_CALLABLE*/
/*    T evaluate (T x) const*/
/*    {*/
/*      T mu = 1.0 - sqrt(1.0 - x*x);*/
/*      return (1.0 - c1_ * mu - c2_ * mu * mu) / I0_;*/
/*    }*/

/*  private:*/
/*    T c1_, c2_, I0_;*/

/*};*/

/*template <typename T>*/
/*TRANSIT_CUDA_CALLABLE*/
/*inline T index_to_coord (int size, int index)*/
/*{*/
/*  T val = 1.0 - index / (size - 1);*/
/*  return 1.0 - val * val;*/
/*}*/

/*template <typename T>*/
/*TRANSIT_CUDA_CALLABLE*/
/*inline T coord_to_index (int size, T coord)*/
/*{*/
/*  return (size - 1) * (1.0 + sqrt(1.0 - coord));*/
/*}*/

/*template <typename T>*/
/*TRANSIT_CUDA_CALLABLE*/
/*inline T compute_area (T x, T z, T r)*/
/*{*/
/*  T rmz = r - z,*/
/*    rpz = r + z,*/
/*    x2 = x*x,*/
/*    r2 = r*r;*/

/*  if (fabs(x - r) < z < x + r) {*/
/*    T z2 = z*z;*/
/*    T u = 0.5 * (z2 + x2 - r2) / ((z) * (x));*/
/*    T v = 0.5 * (z2 + r2 - x2) / ((z) * (r));*/
/*    T w = (x + rmz) * (x - rmz) * (rpz - x) * (rpz + x);*/
/*    if (w < 0.0) w = 0.0;*/
/*    return x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w);*/
/*  } else if (x >= rpz) {*/
/*    return M_PI * r2;*/
/*  } else if (x <= rmz) {*/
/*    return M_PI * x2;*/
/*  }*/
/*  return 0.0;*/
/*}*/

/*template <typename T>*/
/*TRANSIT_CUDA_CALLABLE*/
/*inline T compute_delta (int                    grid_size,*/
/*                        T*  __restrict__ const grid,*/
/*                        T                      z,*/
/*                        T                      r)*/
/*{*/
/*  int indmin = max(0,           int(floor(coord_to_index(grid_size, z - r))));*/
/*  int indmax = min(grid_size-1, int( ceil(coord_to_index(grid_size, z + r))));*/
/*  T delta = 0.0;*/
/*  T A1 = 0.0;*/
/*  T I1 = grid[indmin];*/
/*  T x2, A2, I2;*/
/*  for (int ind = indmin+1; ind <= indmax; ++ind) {*/
/*    x2 = index_to_coord<T>(grid_size, ind);*/
/*    A2 = compute_area<T>(x2, z, r);*/
/*    I2 = grid[ind];*/
/*    delta += 0.5 * (I1 + I2) * (A2 - A1);*/
/*    A1 = A2;*/
/*    I1 = I2;*/
/*  }*/
/*  return delta;*/
/*}*/

using namespace transit;

template <typename T>
__global__ void transit_kernel(int                    grid_size,
                               T*  __restrict__ const grid,
                               int                    size,
                               T*  __restrict__ const z,
                               T*  __restrict__ const r,
                               T*  __restrict__       delta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < size; i += stride) {
    delta[i] = compute_delta<T>(grid_size, grid, z[i], r[i]);
  }
}

int main(void)
{
  typedef double T;

  int N = 10*700000;
  int grid_size = 1000;
  T *z, *r, *delta, *grid;
  QuadraticLimbDarkening<T> ld(0.5, 0.1);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&z,     N*sizeof(T));
  cudaMallocManaged(&r,     N*sizeof(T));
  cudaMallocManaged(&delta, N*sizeof(T));
  cudaMallocManaged(&grid,  grid_size*sizeof(T));

  for (int i = 0; i < N; i++) {
    z[i] = std::abs(1.1 - 2.2*T(i) / (N-1));
    r[i] = 0.01;
  }

  for (int i = 0; i < grid_size; ++i) {
    grid[i] = ld.value(index_to_coord<T>(grid_size, i));
  }

  int blockSize = 1024;
  int numBlocks = (N + blockSize - 1) / blockSize;

  transit_kernel<<<numBlocks, blockSize>>>(grid_size, grid, N, z, r, delta);
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(z);
  cudaFree(r);
  cudaFree(delta);
  cudaFree(grid);

  return 0;
}
