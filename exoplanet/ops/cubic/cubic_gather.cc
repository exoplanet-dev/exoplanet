#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

REGISTER_OP("CubicGather")
  .Attr("T: {float, double}")
  .Input("t: T")
  .Input("x: T")
  .Input("y: T")
  .Input("b: T")
  .Input("c: T")
  .Input("d: T")
  .Output("ak: T")
  .Output("bk: T")
  .Output("ck: T")
  .Output("dk: T")
  .Output("xk: T")
  .Output("inds: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle t, x, y, b, c_, d;

    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &t));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &y));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &b));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(4), 1, &c_));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(5), 1, &d));

    // x any y must be the same shape
    TF_RETURN_IF_ERROR(c->Merge(x, y, &x));

    // b, c, and d must have one entry less than x in the last dimension
    shape_inference::DimensionHandle dim = c->Dim(x, -1);
    TF_RETURN_IF_ERROR(c->Subtract(dim, 1, &dim));
    TF_RETURN_IF_ERROR(c->ReplaceDim(x, -1, dim, &x));
    TF_RETURN_IF_ERROR(c->Merge(x, b, &x));
    TF_RETURN_IF_ERROR(c->Merge(x, c_, &x));
    TF_RETURN_IF_ERROR(c->Merge(x, d, &x));

    // The first dimensions of t must match the others
    TF_RETURN_IF_ERROR(c->Subshape(x, 0, -1, &x));
    TF_RETURN_IF_ERROR(c->MergePrefix(t, x, &t, &x));

    c->set_output(0, t);
    c->set_output(1, t);
    c->set_output(2, t);
    c->set_output(3, t);
    c->set_output(4, t);
    c->set_output(5, t);

    return Status::OK();
  });

template <typename T>
class CubicGatherOp : public OpKernel {
 public:
  explicit CubicGatherOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& t_tensor = context->input(0);
    const Tensor& x_tensor = context->input(1);
    const Tensor& y_tensor = context->input(2);
    const Tensor& b_tensor = context->input(3);
    const Tensor& c_tensor = context->input(4);
    const Tensor& d_tensor = context->input(5);

    OP_REQUIRES(context, y_tensor.shape() == x_tensor.shape(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, x_tensor.dims() == t_tensor.dims(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, x_tensor.dims() == b_tensor.dims(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, x_tensor.dims() == c_tensor.dims(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, x_tensor.dims() == d_tensor.dims(), errors::InvalidArgument("dimension mismatch"));

    const int64 N = x_tensor.dim_size(x_tensor.dims()-1);
    const int64 K = t_tensor.dim_size(t_tensor.dims()-1);
    int64 n_in = 1;
    for (int64 n = 0; n < x_tensor.dims() - 1; ++n) {
      n_in *= x_tensor.dim_size(n);
      OP_REQUIRES(context, x_tensor.dim_size(n) == t_tensor.dim_size(n), errors::InvalidArgument("dimension mismatch"));
      OP_REQUIRES(context, x_tensor.dim_size(n) == b_tensor.dim_size(n), errors::InvalidArgument("dimension mismatch"));
      OP_REQUIRES(context, x_tensor.dim_size(n) == c_tensor.dim_size(n), errors::InvalidArgument("dimension mismatch"));
      OP_REQUIRES(context, x_tensor.dim_size(n) == d_tensor.dim_size(n), errors::InvalidArgument("dimension mismatch"));
    }
    OP_REQUIRES(context, b_tensor.dim_size(x_tensor.dims()-1) == N-1, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, c_tensor.dim_size(x_tensor.dims()-1) == N-1, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, d_tensor.dim_size(x_tensor.dims()-1) == N-1, errors::InvalidArgument("dimension mismatch"));

    // Output
    Tensor* ak_tensor = NULL;
    Tensor* bk_tensor = NULL;
    Tensor* ck_tensor = NULL;
    Tensor* dk_tensor = NULL;
    Tensor* xk_tensor = NULL;
    Tensor* inds_tensor = NULL;
    auto shape = t_tensor.shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &ak_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, shape, &bk_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, shape, &ck_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, shape, &dk_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(4, shape, &xk_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(5, shape, &inds_tensor));

    const auto t = t_tensor.template flat_inner_dims<T, 2>();
    const auto x = x_tensor.template flat_inner_dims<T, 2>();
    const auto y = y_tensor.template flat_inner_dims<T, 2>();
    const auto b = b_tensor.template flat_inner_dims<T, 2>();
    const auto c = c_tensor.template flat_inner_dims<T, 2>();
    const auto d = d_tensor.template flat_inner_dims<T, 2>();
    auto ak = ak_tensor->template flat_inner_dims<T, 2>();
    auto bk = bk_tensor->template flat_inner_dims<T, 2>();
    auto ck = ck_tensor->template flat_inner_dims<T, 2>();
    auto dk = dk_tensor->template flat_inner_dims<T, 2>();
    auto xk = xk_tensor->template flat_inner_dims<T, 2>();
    auto inds = inds_tensor->flat_inner_dims<int64, 2>();

    for (int64 n = 0; n < n_in; ++n) {
      for (int64 k = 0; k < K; ++k) {
        if (t(n, k) <= x(n, 0)) {
          inds(n, k) = -1;
          ak(n, k) = y(n, 0);
          bk(n, k) = T(0);
          ck(n, k) = T(0);
          dk(n, k) = T(0);
          xk(n, k) = x(n, 0);
        } else if (t(n, k) >= x(n, N-1)) {
          inds(n, k) = N - 1;
          ak(n, k) = y(n, N-1);
          bk(n, k) = T(0);
          ck(n, k) = T(0);
          dk(n, k) = T(0);
          xk(n, k) = x(n, N-1);
        } else {
          const T* bound = std::upper_bound(&(x(n, 0)), &(x(n, N-1)), t(n, k));
          int64 ind = bound - &(x(n, 0)) - 1;
          inds(n, k) = ind;
          ak(n, k) = y(n, ind);
          bk(n, k) = b(n, ind);
          ck(n, k) = c(n, ind);
          dk(n, k) = d(n, ind);
          xk(n, k) = x(n, ind);
        }
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CubicGather").Device(DEVICE_CPU).TypeConstraint<type>("T"),    \
      CubicGatherOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
