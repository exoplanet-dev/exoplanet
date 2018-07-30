#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

REGISTER_OP("CubicGatherRev")
  .Attr("T: {float, double}")
  .Input("x: T")
  .Input("inds: int64")
  .Input("bak: T")
  .Input("bbk: T")
  .Input("bck: T")
  .Input("bdk: T")
  .Input("bxk: T")
  .Output("bx: T")
  .Output("by: T")
  .Output("bb: T")
  .Output("bc: T")
  .Output("bd: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle x, inds;

    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &x));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &inds));

    TF_RETURN_IF_ERROR(c->Subshape(inds, 0, -1, &inds));
    TF_RETURN_IF_ERROR(c->MergePrefix(x, inds, &x, &inds));
    TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &inds));
    TF_RETURN_IF_ERROR(c->Merge(inds, c->input(3), &inds));
    TF_RETURN_IF_ERROR(c->Merge(inds, c->input(4), &inds));
    TF_RETURN_IF_ERROR(c->Merge(inds, c->input(5), &inds));
    TF_RETURN_IF_ERROR(c->Merge(inds, c->input(6), &inds));

    c->set_output(0, x);
    c->set_output(1, x);

    shape_inference::DimensionHandle dim = c->Dim(x, -1);
    TF_RETURN_IF_ERROR(c->Subtract(dim, 1, &dim));
    TF_RETURN_IF_ERROR(c->ReplaceDim(x, -1, dim, &x));
    c->set_output(2, x);
    c->set_output(3, x);
    c->set_output(4, x);

    return Status::OK();
  });

template <typename T>
class CubicGatherRevOp : public OpKernel {
 public:
  explicit CubicGatherRevOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& x_tensor    = context->input(0);
    const Tensor& inds_tensor = context->input(1);
    const Tensor& bak_tensor  = context->input(2);
    const Tensor& bbk_tensor  = context->input(3);
    const Tensor& bck_tensor  = context->input(4);
    const Tensor& bdk_tensor  = context->input(5);
    const Tensor& bxk_tensor  = context->input(6);

    OP_REQUIRES(context, inds_tensor.shape() == bak_tensor.shape(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, inds_tensor.shape() == bbk_tensor.shape(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, inds_tensor.shape() == bck_tensor.shape(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, inds_tensor.shape() == bdk_tensor.shape(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, inds_tensor.shape() == bxk_tensor.shape(), errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, inds_tensor.dims() == x_tensor.dims(), errors::InvalidArgument("dimension mismatch"));

    const int64 N = x_tensor.dim_size(x_tensor.dims()-1);
    const int64 K = inds_tensor.dim_size(inds_tensor.dims()-1);
    int64 n_in = 1;
    for (int64 n = 0; n < x_tensor.dims() - 1; ++n) {
      n_in *= x_tensor.dim_size(n);
      OP_REQUIRES(context, x_tensor.dim_size(n) == inds_tensor.dim_size(n), errors::InvalidArgument("dimension mismatch"));
    }

    // Output
    Tensor* bx_tensor = NULL;
    Tensor* by_tensor = NULL;
    Tensor* bb_tensor = NULL;
    Tensor* bc_tensor = NULL;
    Tensor* bd_tensor = NULL;
    auto shape = x_tensor.shape();
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &bx_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, shape, &by_tensor));
    shape.set_dim(x_tensor.dims()-1, N-1);
    OP_REQUIRES_OK(context, context->allocate_output(2, shape, &bb_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, shape, &bc_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(4, shape, &bd_tensor));

    const auto inds = inds_tensor.flat_inner_dims<int64, 2>();
    const auto bak = bak_tensor.template flat_inner_dims<T, 2>();
    const auto bbk = bbk_tensor.template flat_inner_dims<T, 2>();
    const auto bck = bck_tensor.template flat_inner_dims<T, 2>();
    const auto bdk = bdk_tensor.template flat_inner_dims<T, 2>();
    const auto bxk = bxk_tensor.template flat_inner_dims<T, 2>();
    auto bx = bx_tensor->template flat_inner_dims<T, 2>();
    auto by = by_tensor->template flat_inner_dims<T, 2>();
    auto bb = bb_tensor->template flat_inner_dims<T, 2>();
    auto bc = bc_tensor->template flat_inner_dims<T, 2>();
    auto bd = bd_tensor->template flat_inner_dims<T, 2>();

    bx.setZero();
    by.setZero();
    bb.setZero();
    bc.setZero();
    bd.setZero();

    for (int64 n = 0; n < n_in; ++n) {
      for (int64 k = 0; k < K; ++k) {
        if (inds(n, k) == -1) {
          by(n, 0) += bak(n, k);
          bx(n, 0) += bxk(n, k);
        } else if (inds(n, k) == N-1) {
          by(n, N-1) += bak(n, k);
          bx(n, N-1) += bxk(n, k);
        } else {
          int64 ind = inds(n, k);
          by(n, ind) += bak(n, k);
          bb(n, ind) += bbk(n, k);
          bc(n, ind) += bck(n, k);
          bd(n, ind) += bdk(n, k);
          bx(n, ind) += bxk(n, k);
        }
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("CubicGatherRev").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CubicGatherRevOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
