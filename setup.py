#!/usr/bin/env python

import os
import sys
import tensorflow as tf
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    sup = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', "nvcc")
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        sup(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(build_ext):

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


compile_flags = tf.sysconfig.get_compile_flags()
link_flags = tf.sysconfig.get_link_flags()

sources = [
    os.path.join("transit", "transit_op", "transit_op.cc"),
    os.path.join("transit", "transit_op", "transit_rev_op.cc"),
]
include_dirs = [".", "transit", os.path.join("transit", "transit_op")]

# Check for flag and nvcc
if "--with-cuda" in sys.argv:
    index = sys.argv.index("--with-cuda")
    base_dir = sys.argv.pop(index+1)
    sys.argv.pop(index)
    compile_flags += ["-DGOOGLE_CUDA=1"]
    sources += [os.path.join("transit", "transit_op", "transit_op.cc.cu")]
    include_dirs += [os.path.join(base_dir, "include")]
    link_flags += ["-L" + os.path.join(base_dir, "lib64")]


gcc_flags = compile_flags + ["-std=c++11", "-O2", "-march=native", "-fPIC"]
nvcc_flags = compile_flags + ["-std=c++11", "-shared", "-Xcompiler", "-fPIC",
                              "-x", "cu",
                              "--expt-relaxed-constexpr"]
if sys.platform == "darwin":
    gcc_flags += ["-mmacosx-version-min=10.9"]

extensions = [
    Extension(
        "transit.transit_op",
        sources=sources,
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=dict(
            nvcc=nvcc_flags,
            gcc=gcc_flags,
        ),
        extra_link_args=link_flags,
    ),
]

setup(
    name="transit",
    license="MIT",
    packages=["transit"],
    ext_modules=extensions,
    cmdclass={"build_ext": custom_build_ext},
    zip_safe=True,
)
