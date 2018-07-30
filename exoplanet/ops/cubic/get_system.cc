#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

REGISTER_OP("CubicInterpSystem")
  .Attr("T: {float, double}")
  .Input("x: T")
  .Input("y: T")
  .Output("diag: T")
  .Output("upper: T")
  .Output("lower: T")
  .Output("a: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle x, y;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &y));

    TF_RETURN_IF_ERROR(c->Merge(x, y, &x));

    shape_inference::DimensionHandle dim = c->Dim(x, -1);
    TF_RETURN_IF_ERROR(c->Subtract(dim, 1, &dim));

    c->set_output(0, x);
    c->set_output(3, x);
    TF_RETURN_IF_ERROR(c->ReplaceDim(x, -1, dim, &x));
    c->set_output(1, x);
    c->set_output(2, x);

    return Status::OK();
  });

template <typename T>
class CubicInterpSystemOp : public OpKernel {
 public:
  explicit CubicInterpSystemOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& x_tensor = context->input(0);
    const Tensor& y_tensor = context->input(1);

    OP_REQUIRES(context, y_tensor.shape() == x_tensor.shape(), errors::InvalidArgument("dimension mismatch"));

    // Check the dimensions
    const int64 n_inner = x_tensor.dim_size(x_tensor.dims()-1);
    int64 n_in = 1;
    for (int64 n = 0; n < x_tensor.dims() - 1; ++n) {
      n_in *= x_tensor.dim_size(n);
    }

    // Output
    Tensor* diag_tensor = NULL;
    Tensor* upper_tensor = NULL;
    Tensor* lower_tensor = NULL;
    Tensor* a_tensor = NULL;
    auto shape = x_tensor.shape();
    shape.set_dim(x_tensor.dims()-1, n_inner);
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &diag_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, shape, &a_tensor));
    shape.set_dim(x_tensor.dims()-1, n_inner-1);
    OP_REQUIRES_OK(context, context->allocate_output(1, shape, &upper_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, shape, &lower_tensor));

    const auto x = x_tensor.template flat_inner_dims<T, 2>();
    const auto y = y_tensor.template flat_inner_dims<T, 2>();
    auto diag = diag_tensor->template flat_inner_dims<T, 2>();
    auto upper = upper_tensor->template flat_inner_dims<T, 2>();
    auto lower = lower_tensor->template flat_inner_dims<T, 2>();
    auto a = a_tensor->template flat_inner_dims<T, 2>();

    for (int64 n = 0; n < n_in; ++n) {
      T dx = x(n, 1) - x(n, 0);
      T dy = y(n, 1) - y(n, 0);
      diag(n, 0) = 2 * dx;
      upper(n, 0) = dx;
      lower(n, 0) = dx;
      a(n, 0) = 3 * dy / dx;
      for (int64 k = 1; k < n_inner-1; ++k) {
        T dx_tmp = x(n, k+1) - x(n, k);
        T dy_tmp = y(n, k+1) - y(n, k);
        diag(n, k) = 2*(dx_tmp + dx);
        a(n, k) = 3 * (dy_tmp / dx_tmp - dy / dx);
        dx = dx_tmp;
        dy = dy_tmp;
        upper(n, k) = dx;
        lower(n, k) = dx;
      }
      diag(n, n_inner-1) = 2 * dx;
      a(n, n_inner-1) = -3 * dy / dx;
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CubicInterpSystem").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      CubicInterpSystemOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
