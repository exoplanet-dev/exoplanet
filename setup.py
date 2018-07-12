#!/usr/bin/env python

import os
import sys
import glob
import tensorflow as tf
from setuptools import setup

from exoplanet.cpp_extension import BuildExtension, CppExtension


def src_path(name, filename):
    return os.path.join("exoplanet", "ops", name, filename)


def construct_op(name):
    path = os.path.join("exoplanet", "ops", name)
    cpp_files = glob.glob(os.path.join(path, "*.cc"))
    cuda_files = glob.glob(os.path.join(path, "*.cu"))

    flags = []
    if sys.platform == "darwin":
        flags += ["-mmacosx-version-min=10.9"]

    if tf.test.is_built_with_cuda() and len(cuda_files):
        assert 0
    return CppExtension(
        "exoplanet." + name + "_op",
        cpp_files,
        include_dirs=[path, "include"],
        extra_compile_args=flags,
        extra_link_args=flags,
    )


extensions = [
    construct_op("transit"),
    construct_op("interp"),
    construct_op("kepler"),
]


# def customize_compiler_for_nvcc(self):
#     self.src_extensions.append('.cu')
#     default_compiler_so = self.compiler_so
#     sup = self._compile

#     def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
#         if os.path.splitext(src)[1] == '.cu':
#             self.set_executable('compiler_so', "nvcc")
#             postargs = extra_postargs["nvcc"]
#         else:
#             postargs = extra_postargs["gcc"]

#         sup(obj, src, ext, cc_args, postargs, pp_opts)
#         self.compiler_so = default_compiler_so

#     self._compile = _compile


# class custom_build_ext(build_ext):

#     def build_extensions(self):
#         customize_compiler_for_nvcc(self.compiler)
#         build_ext.build_extensions(self)


# compile_flags = tf.sysconfig.get_compile_flags()
# link_flags = tf.sysconfig.get_link_flags()

# transit_sources = [
#     os.path.join("exoplanet", "ops", "transit", "transit_op.cc"),
#     os.path.join("exoplanet", "ops", "transit", "transit_rev_op.cc"),
#     os.path.join("exoplanet", "ops", "transit", "occulted_area_op.cc"),
#     os.path.join("exoplanet", "ops", "transit", "occulted_area_rev_op.cc"),
# ]
# kepler_sources = [
#     os.path.join("exoplanet", "ops", "kepler", "kepler_op.cc"),
# ]
# search_sorted_sources = [
#     os.path.join("exoplanet", "ops", "search_sorted", "search_sorted_op.cc"),
# ]
# interp_sources = [
#     os.path.join("exoplanet", "ops", "interp", "interp_op.cc"),
#     os.path.join("exoplanet", "ops", "interp", "interp_rev_op.cc"),
# ]
# include_dirs = [
#     ".", "include", "exoplanet",
#     os.path.join("exoplanet", "ops", "transit"),
#     os.path.join("exoplanet", "ops", "kepler"),
# ]

# # Check for flag and nvcc
# if "--with-cuda" in sys.argv:
#     index = sys.argv.index("--with-cuda")
#     base_dir = sys.argv.pop(index+1)
#     sys.argv.pop(index)
#     compile_flags += ["-DGOOGLE_CUDA=1"]
#     transit_sources += [
#         os.path.join("exoplanet", "transit_op", "transit_op.cc.cu")]
#     include_dirs += [os.path.join(base_dir, "include")]
#     link_flags += ["-L" + os.path.join(base_dir, "lib64")]


# gcc_flags = compile_flags + ["-std=c++11", "-O2", "-march=native", "-fPIC"]
# nvcc_flags = compile_flags + ["-std=c++11", "-shared", "-Xcompiler", "-fPIC",
#                               "-x", "cu",
#                               "--expt-relaxed-constexpr"]
# if sys.platform == "darwin":
#     gcc_flags += ["-mmacosx-version-min=10.9"]

# extensions = [
#     Extension(
#         "exoplanet.transit_op",
#         sources=transit_sources,
#         language="c++",
#         include_dirs=include_dirs,
#         extra_compile_args=dict(
#             nvcc=nvcc_flags,
#             gcc=gcc_flags,
#         ),
#         extra_link_args=link_flags,
#     ),
#     Extension(
#         "exoplanet.kepler_op",
#         sources=kepler_sources,
#         language="c++",
#         include_dirs=include_dirs,
#         extra_compile_args=dict(
#             nvcc=nvcc_flags,
#             gcc=gcc_flags,
#         ),
#         extra_link_args=link_flags,
#     ),
#     Extension(
#         "exoplanet.search_sorted_op",
#         sources=search_sorted_sources,
#         language="c++",
#         include_dirs=include_dirs,
#         extra_compile_args=dict(
#             nvcc=nvcc_flags,
#             gcc=gcc_flags,
#         ),
#         extra_link_args=link_flags,
#     ),
#     Extension(
#         "exoplanet.interp_op",
#         sources=interp_sources,
#         language="c++",
#         include_dirs=include_dirs,
#         extra_compile_args=dict(
#             nvcc=nvcc_flags,
#             gcc=gcc_flags,
#         ),
#         extra_link_args=link_flags,
#     ),
# ]

setup(
    name="exoplanet",
    license="MIT",
    packages=["exoplanet"],
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=True,
)
