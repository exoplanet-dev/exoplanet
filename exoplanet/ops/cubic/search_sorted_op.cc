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
    shape_inference::ShapeHandle a, v;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &v));

    // Must have the same first dimensions
    TF_RETURN_IF_ERROR(c->Subshape(a, 0, -1, &a));
    TF_RETURN_IF_ERROR(c->Subshape(v, 0, -1, &v));
    TF_RETURN_IF_ERROR(c->Merge(a, v, &v));

    c->set_output(0, c->input(1));
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
    const int64 dim = a_tensor.dims();
    OP_REQUIRES(context, dim == v_tensor.dims(),
        errors::InvalidArgument("a and v must have the same number of dimensions"));
    int64 K = 1;
    for (int64 n = 0; n < dim-1; ++n) {
      K *= a_tensor.dim_size(n);
      OP_REQUIRES(context, a_tensor.dim_size(n) == v_tensor.dim_size(n),
          errors::InvalidArgument("a and v must have the same inner dimensions"));
    }
    const int64 N = a_tensor.dim_size(dim-1);
    const int64 M = v_tensor.dim_size(dim-1);

    // Output
    Tensor* inds_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_tensor.shape(), &inds_tensor));

    // Access the data
    const auto a = a_tensor.template flat_inner_dims<T, 2>();
    const auto v = v_tensor.template flat_inner_dims<T, 2>();
    auto inds = inds_tensor->flat_inner_dims<int64, 2>();

    for (int64 k = 0; k < K; ++k) {
      for (int64 m = 0; m < M; ++m) {
        if (v(k, m) >= a(k, N-1)) {
          inds(k, m) = N;
        } else {
          const T* bound = std::upper_bound(&(a(k, 0)), &(a(k, N-1)), v(k, m));
          int64 ind = bound - &(a(k, 0));
          inds(k, m) = ind;
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
