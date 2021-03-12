# -*- coding: utf-8 -*-

__all__ = ["RegularGridOp"]

import sys

import aesara_theano_fallback.tensor as tt
import pkg_resources
from aesara_theano_fallback import aesara as theano

from ..build_utils import get_cache_version
from ..compat import Apply, COp, ParamsType


class RegularGridOp(COp):
    params_type = ParamsType(
        ndim=theano.scalar.int64,
        nout=theano.scalar.int64,
        check_sorted=theano.scalar.bool,
        bounds_error=theano.scalar.bool,
    )
    __props__ = ("ndim", "nout", "check_sorted", "bounds_error")
    func_file = "./regular_grid.cc"
    func_name = "APPLY_SPECIFIC(regular_grid)"

    def __init__(
        self, ndim, nout=-1, check_sorted=True, bounds_error=True, **kwargs
    ):
        self.ndim = int(ndim)
        if not 0 < self.ndim <= 5:
            raise ValueError("ndim must be less than or equal to 5")
        self.nout = int(nout)
        self.check_sorted = bool(check_sorted)
        self.bounds_error = bool(bounds_error)
        super(RegularGridOp, self).__init__(self.func_file, self.func_name)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("Only the C version is implemented")

    def c_code_cache_version(self, *args, **kwargs):
        return get_cache_version()

    def c_headers(self, *args, **kwargs):
        return ["theano_helpers.h"]

    def c_header_dirs(self, *args, **kwargs):
        return [
            pkg_resources.resource_filename(__name__, "include"),
            pkg_resources.resource_filename(__name__, "../lib/vendor/eigen"),
        ]

    def c_compile_args(self, *args, **kwargs):
        args = ["-std=c++11", "-DNDEBUG"]
        if sys.platform == "darwin":
            args += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        if sys.platform.startswith("win"):
            args += ["-D_USE_MATH_DEFINES", "-D_hypot=hypot"]
        else:
            args += ["-O2"]

        args.append("-DREGULAR_GRID_NDIM={0}".format(self.ndim))
        if self.ndim == 1:
            args.append("-DREGULAR_GRID_NDIM_ORDER=Eigen::ColMajor")
        else:
            args.append("-DREGULAR_GRID_NDIM_ORDER=Eigen::RowMajor")

        if 0 < self.nout <= 16:
            args.append("-DREGULAR_GRID_NOUT={0}".format(self.nout))
        if self.nout == 1:
            args.append("-DREGULAR_GRID_NOUT_ORDER=Eigen::ColMajor")
        else:
            args.append("-DREGULAR_GRID_NOUT_ORDER=Eigen::RowMajor")

        ndim_nout = self.ndim * self.nout
        if 0 < ndim_nout <= 16:
            args.append("-DREGULAR_GRID_NDIM_NOUT={0}".format(ndim_nout))
        if ndim_nout == 1:
            args.append("-DREGULAR_GRID_NDIM_NOUT_ORDER=Eigen::ColMajor")
        else:
            args.append("-DREGULAR_GRID_NDIM_NOUT_ORDER=Eigen::RowMajor")

        for i in range(self.ndim):
            args.append("-DREGULAR_GRID_{0}".format(i))

        return args

    def make_node(self, *args):
        ndim = len(args) - 2
        if ndim != self.ndim:
            raise ValueError("number of input grids does not match ndim")

        dtype = theano.config.floatX
        in_args = []
        for a in args:
            try:
                a = tt.as_tensor_variable(a)
            except tt.AsTensorError:
                pass
            else:
                dtype = theano.scalar.upcast(dtype, a.dtype)
            in_args.append(a)
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False, False])(),
            tt.TensorType(dtype=dtype, broadcastable=[False, False, False])(),
        ]
        return Apply(self, in_args, out_args)

    def grad(self, inputs, gradients):
        xi = inputs[0]
        zi, dz = self(*inputs)
        bz = gradients[0]

        bx = tt.sum(
            tt.reshape(bz, (xi.shape[0], 1, zi.shape[1])) * dz, axis=-1
        )
        return tuple([bx] + [tt.zeros_like(i) for i in inputs[1:]])

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
