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
T compute_area (T x, T z, T r)
{
  T rmz = r - z,
    rpz = r + z,
    x2 = (x)*(x),
    r2 = (r)*(r);

  if (x <= -rmz) {
    return 0.0;
  } if (x <= rmz) {
    return x2;
  } else if (x >= rpz) {
    return r2;
  }

  T z2 = (z)*(z);
  T u = 0.5 * (z2 + x2 - r2) / ((z) * (x));
  T v = 0.5 * (z2 + r2 - x2) / ((z) * (r));
  T w = (x + rmz) * (x - rmz) * (rpz - x) * (rpz + x);
  if (w < 0.0) w = 0.0;
  return (x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w)) / M_PI;
}

template <typename T, typename LimbDarkening>
__global__ void transit_kernel(LimbDarkening ld,
                               T                                 step_scale,
                               int                               size,
                               T*             __restrict__ const z,
                               T*             __restrict__ const r,
                               T*             __restrict__       delta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  T xmin, xmax, x1, x2, x, A1, A2, I;
  for (int i = index; i < size; i += stride) {
    xmin = fmax(0.0, z[i] - r[i]);
    xmax = fmin(1.0, z[i] + r[i]);
    delta[i] = 0.0;
    if (xmax > xmin) {
      x1 = xmin;
      A1 = 0.0;
      while (x1 < xmax) {
        x2 = step_advance<T>(step_scale, x1);
        x = 0.5 * (x1 + x2);
        A2 = compute_area<T>(x2, z[i], r[i]);
        I = ld.evaluate(x);
        delta[i] += (A2 - A1) * I;
        x1 = x2;
        A1 = A2;
      }
    }
  }
}

int main(void)
{
  typedef float T;

  T step_scale = 1e-3;
  int N = 10*700000;
  /*int N = 100;*/
  T *z, *r, *delta;
  QuadraticLimbDarkening<T> ld(0.5, 0.1);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&z,     N*sizeof(T));
  cudaMallocManaged(&r,     N*sizeof(T));
  cudaMallocManaged(&delta, N*sizeof(T));

  /*cudaMallocManaged(&ld,    sizeof(QuadraticLimbDarkening<T>));*/
  /*ld[0] = QuadraticLimbDarkening<T>(0.5, 0.1);*/

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    z[i] = std::abs(1.1 - 2.2*T(i) / (N-1));
    r[i] = 0.01;
  }

  int blockSize = 128;
  int numBlocks = (N + blockSize - 1) / blockSize;

  transit_kernel<<<numBlocks, blockSize>>>(ld, step_scale, N, z, r, delta);
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  /*for (int i = 0; i < N; i++)*/
  /*  std::cout << z[i] << " " << delta[i] << "\n";*/

  // Free memory
  cudaFree(z);
  cudaFree(r);
  cudaFree(delta);
  /*cudaFree(ld);*/

  return 0;
}
