#include <iostream>
#include <cmath>

template <typename T>
struct QuadraticLimbDarkening {

  public:
    QuadraticLimbDarkening (T c1, T c2) : c1_(c1), c2_(c2) {
      I0_ = 1.0 - (2.0 * c1_ + c2) / 6.0;
    }

    __host__ __device__
    T evaluate (T x) const
    {
      T mu = 1.0 - sqrt(1.0 - x*x);
      return (1.0 - c1_ * mu - c2_ * mu * mu) / I0_;
    }

  private:
    T c1_, c2_, I0_;

};

template <typename T>
__host__ __device__ inline
T step_advance (T step_scale, T x1)
{
  return x1 + step_scale * acos(x1);
}

template <typename T>
__host__ __device__ inline
T index_to_coord (int size, int index)
{
  T val = 1.0 - index / (size - 1);
  return 1.0 - val * val;
}

template <typename T>
__host__ __device__ inline
T coord_to_index (int size, T coord)
{
  return (size - 1) * (1.0 + sqrt(1.0 - coord));
}

template <typename T>
__host__ __device__ inline
T compute_area (T x, T z, T r)
{
  T rmz = r - z,
    rpz = r + z,
    x2 = x*x,
    r2 = r*r;

  if (x <= -rmz) {
    return 0.0;
  } if (x <= rmz) {
    return x2;
  } else if (x >= rpz) {
    return r2;
  }

  T z2 = z*z;
  T u = 0.5 * (z2 + x2 - r2) / ((z) * (x));
  T v = 0.5 * (z2 + r2 - x2) / ((z) * (r));
  T w = (x + rmz) * (x - rmz) * (rpz - x) * (rpz + x);
  if (w < 0.0) w = 0.0;
  return (x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w)) / M_PI;
}

template <typename T>
__global__ void transit_kernel(int                               grid_size,
                               T*             __restrict__ const grid,
                               int                               size,
                               T*             __restrict__ const z,
                               T*             __restrict__ const r,
                               T*             __restrict__       delta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  T xmin, xmax, x1, x2, x, A1, A2, I1, I2;
  int indmin, indmax;
  for (int i = index; i < size; i += stride) {
    indmin = max(0,           int(floor(coord_to_index(grid_size, z[i] - r[i]))));
    indmax = min(grid_size-1, int( ceil(coord_to_index(grid_size, z[i] + r[i]))));
    delta[i] = 0.0;
    A1 = 0.0;
    I1 = grid[indmin];
    for (int ind = indmin+1; ind <= indmax; ++ind) {
      x2 = index_to_coord<T>(grid_size, ind);
      A2 = compute_area<T>(x2, z[i], r[i]);
      I2 = grid[ind];
      delta[i] += 0.5 * (I1 + I2) * (A2 - A1);
      A1 = A2;
      I1 = I2;
    }
  }
}

int main(void)
{
  typedef float T;

  T step_scale = 1e-3;
  int N = 10*700000;
  /*int N = 100;*/
  int grid_size = 1000;
  T *z, *r, *delta, *grid;
  QuadraticLimbDarkening<T> ld(0.5, 0.1);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&z,     N*sizeof(T));
  cudaMallocManaged(&r,     N*sizeof(T));
  cudaMallocManaged(&delta, N*sizeof(T));
  cudaMallocManaged(&grid,  grid_size*sizeof(T));

  /*cudaMallocManaged(&ld,    sizeof(QuadraticLimbDarkening<T>));*/
  /*ld[0] = QuadraticLimbDarkening<T>(0.5, 0.1);*/

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    z[i] = std::abs(1.1 - 2.2*T(i) / (N-1));
    r[i] = 0.01;
  }

  for (int i = 0; i < grid_size; ++i) {
    grid[i] = ld.evaluate(index_to_coord<T>(grid_size, i));
  }

  int blockSize = 128;
  int numBlocks = (N + blockSize - 1) / blockSize;

  transit_kernel<<<numBlocks, blockSize>>>(grid_size, grid, N, z, r, delta);
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(z);
  cudaFree(r);
  cudaFree(delta);
  cudaFree(grid);
  /*cudaFree(ld);*/

  return 0;
}
