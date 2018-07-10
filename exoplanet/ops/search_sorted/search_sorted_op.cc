#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

REGISTER_OP("SearchSorted")
  .Attr("T: {float, double}")
  .Input("a: T")
  .Input("v: T")
  .Output("inds: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle a, v, tmp;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &v));

    // Must have the same first dimension
    TF_RETURN_IF_ERROR(c->Subshape(a, 0, 1, &a));
    TF_RETURN_IF_ERROR(c->Subshape(v, 0, 1, &tmp));
    TF_RETURN_IF_ERROR(c->Merge(a, tmp, &tmp));

    TF_RETURN_IF_ERROR(c->Concatenate(v, tmp, &tmp));
    TF_RETURN_IF_ERROR(c->ReplaceDim(tmp, 2, c->MakeDim(2), &tmp));

    c->set_output(0, tmp);
    return Status::OK();
  });

template <typename T>
class SearchSortedOp : public OpKernel {
 public:
  explicit SearchSortedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& a_tensor = context->input(0);
    const Tensor& v_tensor = context->input(1);

    // Dimensions
    const int64 K = v_tensor.dim_size(0);
    const int64 N = a_tensor.dim_size(1);
    const int64 M = v_tensor.dim_size(1);
    OP_REQUIRES(context, a_tensor.dim_size(0) == K,
        errors::InvalidArgument("a and v must have the same first dimension"));

    // Output
    Tensor* inds_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({K, M, 2}), &inds_tensor));

    // Access the data
    const auto a = a_tensor.template matrix<T>();
    const auto v = v_tensor.template matrix<T>();
    auto inds = inds_tensor->tensor<int64, 3>();

    for (int64 k = 0; k < K; ++k) {
      for (int64 m = 0; m < M; ++m) {
        inds(k, m, 0) = k;
        if (v(k, m) >= a(k, N-1)) {
          inds(k, m, 1) = N;
        } else {
          const T* bound = std::upper_bound(&(a(k, 0)), &(a(k, N-1)), v(k, m));
          int64 ind = bound - &(a(k, 0));
          inds(k, m, 1) = ind;
        }
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SearchSorted").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      SearchSortedOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
