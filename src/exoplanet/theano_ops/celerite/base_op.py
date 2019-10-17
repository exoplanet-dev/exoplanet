# -*- coding: utf-8 -*-

__all__ = ["CeleriteBaseOp"]

from theano import gof

from ..build_utils import get_cache_version, get_compile_args, get_header_dirs


class CeleriteBaseOp(gof.COp):

    __props__ = ("J", "n_rhs")
    func_file = None
    func_name = None

    def __init__(self, J=-1, n_rhs=-1):
        self.J = int(J)
        self.n_rhs = int(n_rhs)
        super(CeleriteBaseOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        return get_cache_version()

    def c_headers(self, compiler):
        return ["exoplanet/theano_helpers.h", "exoplanet/celerite.h"]

    def c_header_dirs(self, compiler):
        return get_header_dirs(eigen=True)

    def c_compile_args(self, compiler):
        args = get_compile_args(compiler)
        if 0 < self.J <= 32:
            args.append("-DCELERITE_J={0}".format(self.J))
            args.append("-DCELERITE_J2={0}".format(self.J * self.J))
            if self.J == 1:
                args.append("-DCELERITE_J_ORDER=Eigen::ColMajor")
            else:
                args.append("-DCELERITE_J_ORDER=Eigen::RowMajor")
        if 0 < self.n_rhs <= 32:
            args.append("-DCELERITE_NRHS={0}".format(self.n_rhs))
            args.append("-DCELERITE_JNRHS={0}".format(self.n_rhs * self.J))
            if self.n_rhs == 1:
                args.append("-DCELERITE_NRHS_ORDER=Eigen::ColMajor")
            else:
                args.append("-DCELERITE_NRHS_ORDER=Eigen::RowMajor")
            if self.n_rhs == 1 and self.J == 1:
                args.append("-DCELERITE_JNRHS_ORDER=Eigen::ColMajor")
            else:
                args.append("-DCELERITE_JNRHS_ORDER=Eigen::RowMajor")
        return args
