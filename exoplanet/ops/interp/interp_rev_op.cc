#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

REGISTER_OP("InterpRev")
  .Attr("T: {float, double}")
  .Input("t: T")
  .Input("p: T")
  .Input("x: T")
  .Input("y: T")
  .Input("a: T")
  .Input("inds: int64")
  .Input("bv: T")
  .Output("bt: T")
  .Output("bp: T")
  .Output("by: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle t, p, x, y, a, inds, bv;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &t));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &p));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &y));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(4), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(5), 1, &inds));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(6), 1, &bv));

    TF_RETURN_IF_ERROR(c->Merge(x, y, &x));
    TF_RETURN_IF_ERROR(c->Merge(t, a, &t));
    TF_RETURN_IF_ERROR(c->Merge(t, inds, &t));
    TF_RETURN_IF_ERROR(c->Merge(t, bv, &t));

    TF_RETURN_IF_ERROR(c->Subshape(t, 0, -1, &t));
    TF_RETURN_IF_ERROR(c->Subshape(x, 0, -1, &x));
    TF_RETURN_IF_ERROR(c->Merge(t, x, &t));
    TF_RETURN_IF_ERROR(c->Merge(t, p, &t));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    c->set_output(2, c->input(2));
    return Status::OK();
  });

template <typename T>
class InterpRevOp : public OpKernel {
 public:
  explicit InterpRevOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& t_tensor    = context->input(0);
    const Tensor& p_tensor    = context->input(1);
    const Tensor& x_tensor    = context->input(2);
    const Tensor& y_tensor    = context->input(3);
    const Tensor& a_tensor    = context->input(4);
    const Tensor& inds_tensor = context->input(5);
    const Tensor& bv_tensor   = context->input(6);

    // Check that the dimensions are consistent
    int64 ndim = t_tensor.dims();
    OP_REQUIRES(context, ndim >= 1,
        errors::InvalidArgument("t must be at least 1D"));
    OP_REQUIRES(context, p_tensor.dims() == ndim - 1,
        errors::InvalidArgument("p must have the dimension len(t.shape) - 1"));
    OP_REQUIRES(context, x_tensor.dims() == ndim,
        errors::InvalidArgument("x and t must have the same number of dimensions"));
    OP_REQUIRES(context, y_tensor.shape() == x_tensor.shape(),
        errors::InvalidArgument("x and y must be the same shape"));
    OP_REQUIRES(context, t_tensor.shape() == inds_tensor.shape(),
        errors::InvalidArgument("t and inds must be the same shape"));
    OP_REQUIRES(context, t_tensor.shape() == bv_tensor.shape(),
        errors::InvalidArgument("t and bv must be the same shape"));

    // Compute the full size of the inner dimensions
    int64 size = 1;
    for (int64 k = 0; k < ndim - 1; ++k) {
      int64 dim = t_tensor.dim_size(k);
      size *= dim;
      OP_REQUIRES(context, (x_tensor.dim_size(k) == dim && p_tensor.dim_size(k) == dim),
          errors::InvalidArgument("incompatible dimensions"));
    }

    // The outer dimensions
    const int64 N = t_tensor.dim_size(ndim - 1);
    const int64 M = x_tensor.dim_size(ndim - 1);

    // Output
    Tensor* bt_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, t_tensor.shape(), &bt_tensor));
    Tensor* bp_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, p_tensor.shape(), &bp_tensor));
    Tensor* by_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, y_tensor.shape(), &by_tensor));

    // Access the data
    const auto t    = t_tensor.template flat_inner_dims<T, 2>();
    const auto p    = p_tensor.template flat_inner_dims<T, 1>();
    const auto x    = x_tensor.template flat_inner_dims<T, 2>();
    const auto y    = y_tensor.template flat_inner_dims<T, 2>();
    const auto a    = a_tensor.template flat_inner_dims<T, 2>();
    const auto inds = inds_tensor.flat_inner_dims<int64, 2>();
    const auto bv   = bv_tensor.template flat_inner_dims<T, 2>();
    auto bt         = bt_tensor->template flat_inner_dims<T, 2>();
    auto bp         = bp_tensor->template flat_inner_dims<T, 1>();
    auto by         = by_tensor->template flat_inner_dims<T, 2>();

    bt.setConstant(0.0);
    bp.setConstant(0.0);
    by.setConstant(0.0);

    for (int64 k = 0; k < size; ++k) {
      T period = p(k);
      bool flag = (period > T(0));

      for (int64 n = 0; n < N; ++n) {
        int64 ind = inds(k, n);
        if (ind == 0) {
          by(k, 0) += bv(k, n);
        } else if (ind == M+1) {
          by(k, M-1) += bv(k, n);
        } else {
          T bvalue = bv(k, n) * (y(k, ind) - y(k, ind-1)) / (x(k, ind) - x(k, ind-1));
          bt(k, n)     += bvalue;
          by(k, ind)   += bv(k, n) * a(k, n);
          by(k, ind-1) += bv(k, n) * (1.0 - a(k, n));
          if (flag) bp(k) -= bvalue * std::floor(t(k, n) / period);
        }
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("InterpRev").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      InterpRevOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
