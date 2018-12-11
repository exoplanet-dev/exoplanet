# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RegularGridOp"]

import pkg_resources

import theano
from theano import gof
import theano.tensor as tt

from ..build_utils import get_compile_args


class RegularGridOp(gof.COp):
    params_type = gof.ParamsType(
        ndim=theano.scalar.int64,
        nout=theano.scalar.int64,
        check_sorted=theano.scalar.bool,
        bounds_error=theano.scalar.bool,
    )
    __props__ = ("ndim", "nout", "check_sorted", "bounds_error")
    func_file = "./regular_grid.cc"
    func_name = "APPLY_SPECIFIC(regular_grid)"

    def __init__(self, ndim, nout=-1, check_sorted=True, bounds_error=True,
                 **kwargs):
        self.ndim = int(ndim)
        if not 0 < self.ndim <= 5:
            raise ValueError("ndim must be less than or equal to 5")
        self.nout = int(nout)
        self.check_sorted = bool(check_sorted)
        self.bounds_error = bool(bounds_error)
        super(RegularGridOp, self).__init__(self.func_file, self.func_name)

    # def c_code_cache_version(self):
    #     return (0, 0, 1)

    def c_headers(self, compiler):
        return ["theano_helpers.h"]

    def c_header_dirs(self, compiler):
        return [
            pkg_resources.resource_filename(__name__, "include"),
            pkg_resources.resource_filename("exoplanet.theano_ops.starry",
                                            "starry/lib/eigen_3.3.3")
        ]

    def c_compile_args(self, compiler):
        args = get_compile_args(compiler)
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
            tt.TensorType(dtype=dtype,
                          broadcastable=[False, False])(),
            tt.TensorType(dtype=dtype,
                          broadcastable=[False] * (self.ndim + 1))(),
        ]
        return gof.Apply(self, in_args, out_args)

    # def infer_shape(self, node, shapes):
    #     return shapes[-1], [self.ndim + 1]

    # def grad(self, inputs, gradients):
    #     M, e = inputs
    #     E, f = self(M, e)

    #     bM = tt.zeros_like(M)
    #     be = tt.zeros_like(M)
    #     ecosE = e * tt.cos(E)

    #     if not isinstance(gradients[0].type, theano.gradient.DisconnectedType):
    #         # Backpropagate E_bar
    #         bM = gradients[0] / (1 - ecosE)
    #         be = tt.sin(E) * bM

    #     if not isinstance(gradients[1].type, theano.gradient.DisconnectedType):
    #         # Backpropagate f_bar
    #         sinf2 = tt.sin(0.5*f)
    #         cosf2 = tt.cos(0.5*f)
    #         tanf2 = sinf2 / cosf2
    #         e2 = e**2
    #         ome2 = 1 - e2
    #         ome = 1 - e
    #         ope = 1 + e
    #         cosf22 = cosf2**2
    #         twoecosf22 = 2 * e * cosf22
    #         factor = tt.sqrt(ope/ome)
    #         inner = (twoecosf22+ome) * tt.as_tensor_variable(gradients[1])

    #         bM += factor*(ome*tanf2**2+ope)*inner*cosf22/(ope*ome2)
    #         be += -2*cosf22*tanf2/ome2**2*inner*(ecosE-2+e2)

    #     return [bM, be]

    # def R_op(self, inputs, eval_points):
    #     if eval_points[0] is None:
    #         return eval_points
    #     return self.grad(inputs, eval_points)
