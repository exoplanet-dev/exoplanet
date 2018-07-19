#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include "interp_op.h"

using namespace tensorflow;
using namespace exoplanet;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Interp")
  .Attr("T: {float, double}")
  .Input("t: T")
  .Input("x: T")
  .Input("y: T")
  .Output("v: T")
  .Output("inds: int32")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle t, x, y;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &t));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &y));

    TF_RETURN_IF_ERROR(c->Merge(x, y, &x));

    TF_RETURN_IF_ERROR(c->Subshape(t, 0, -1, &t));
    TF_RETURN_IF_ERROR(c->Subshape(x, 0, -1, &x));
    TF_RETURN_IF_ERROR(c->Merge(t, x, &t));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(0));
    return Status::OK();
  });

template <typename T>
class InterpOpBase : public OpKernel {
 public:
  explicit InterpOpBase (OpKernelConstruction* context) : OpKernel(context) {}

  virtual void DoCompute (OpKernelContext* context,
    int64 size,
    int M, const T* const x, const T* const y,
    int N, const T* const t, T* v, int* inds) = 0;

  void Compute (OpKernelContext* context) override {
    // Inputs
    const Tensor& t_tensor = context->input(0);
    const Tensor& x_tensor = context->input(1);
    const Tensor& y_tensor = context->input(2);

    // Check that the dimensions are consistent
    int64 ndim = t_tensor.dims();
    OP_REQUIRES(context, ndim >= 1,
        errors::InvalidArgument("t must be at least 1D"));
    OP_REQUIRES(context, x_tensor.dims() == ndim,
        errors::InvalidArgument("x and t must have the same number of dimensions"));
    OP_REQUIRES(context, y_tensor.shape() == x_tensor.shape(),
        errors::InvalidArgument("x and y must be the same shape"));

    // Compute the full size of the inner dimensions
    int64 size = 1;
    for (int64 k = 0; k < ndim - 1; ++k) {
      int64 dim = t_tensor.dim_size(k);
      size *= dim;
      OP_REQUIRES(context, x_tensor.dim_size(k) == dim,
          errors::InvalidArgument("incompatible dimensions"));
    }

    // The outer dimensions
    const int64 N = t_tensor.dim_size(ndim - 1);
    const int64 M = x_tensor.dim_size(ndim - 1);
    OP_REQUIRES(context, N <= tensorflow::kint32max,
        errors::InvalidArgument("too many elements in tensor"));
    OP_REQUIRES(context, M <= tensorflow::kint32max,
        errors::InvalidArgument("too many elements in tensor"));

    // Output
    Tensor* v_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, t_tensor.shape(), &v_tensor));
    Tensor* inds_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, t_tensor.shape(), &inds_tensor));

    // Access the data
    const auto t = t_tensor.template flat_inner_dims<T, 2>();
    const auto x = x_tensor.template flat_inner_dims<T, 2>();
    const auto y = y_tensor.template flat_inner_dims<T, 2>();
    auto v       = v_tensor->template flat_inner_dims<T, 2>();
    auto inds    = inds_tensor->flat_inner_dims<int, 2>();

    DoCompute(context,
      size,
      M, x.data(), y.data(),
      N, t.data(), v.data(), inds.data()
    );
  }
};



template <class Device, typename T>
class InterpOp;

template <typename T>
class InterpOp<CPUDevice, T> : public InterpOpBase<T> {

  public:
    explicit InterpOp (OpKernelConstruction* context) : InterpOpBase<T>(context) {}

    void DoCompute (OpKernelContext* ctx,
      int64 size,
      int M, const T* const x, const T* const y,
      int N, const T* const t, T* v, int* inds
    ) override {

      auto work = [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
          int k = i / N;
          int off_m = k * M;
          inds[i] = interp::interp1d(M, x + off_m, y + off_m, t[i], v + i);
        }
      };

      auto worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
      int64 cost = 5*M;
      Shard(worker_threads.num_threads, worker_threads.workers, N * size, cost, work);
    }

};


#define REGISTER_CPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Interp").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      InterpOp<CPUDevice, type>)

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

#ifdef GOOGLE_CUDA

template <typename T>
class InterpOp<GPUDevice, T> : public InterpOpBase<T> {

  public:
    explicit InterpOp (OpKernelConstruction* context) : InterpOpBase<T>(context) {}

    void DoCompute (OpKernelContext* ctx,
      int64 size,
      int M, const T* const x, const T* const y,
      int N, const T* const t, T* v, int* inds
    ) override {
      OP_REQUIRES(ctx, (size * N) <= tensorflow::kint32max,
          errors::InvalidArgument("too many elements in tensor"));
      InterpCUDAFunctor<T>()(ctx->eigen_device<GPUDevice>(), static_cast<int>(size * N), M, x, y, N, t, v, inds);
    }

};

#define REGISTER_GPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Interp").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      InterpOp<GPUDevice, type>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

#endif
