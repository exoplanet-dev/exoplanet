# -*- coding: utf-8 -*-

__all__ = ["StarryBaseOp"]

from theano import gof

from ..build_utils import get_cache_version, get_compile_args, get_header_dirs


class StarryBaseOp(gof.COp):

    __props__ = ()
    func_file = None
    func_name = None

    def __init__(self):
        super(StarryBaseOp, self).__init__(self.func_file, self.func_name)

    def c_code_cache_version(self):
        return get_cache_version()

    def c_headers(self, compiler):
        xo = lambda f: "exoplanet/" + f  # NOQA
        vc = lambda f: "exoplanet/integrate/" + f  # NOQA
        st = lambda f: "exoplanet/starry/" + f  # NOQA
        return [
            xo("theano_helpers.h"),
            vc("integrators.h"),
            vc("functors.h"),
            st("ellip.h"),
            st("limbdark.h"),
            st("utils.h"),
            "vector",
        ]

    def c_header_dirs(self, compiler):
        return get_header_dirs(eigen=True)

    def c_compile_args(self, compiler):
        return get_compile_args(compiler)
