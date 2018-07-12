# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["BuildExtension"]

import os
import re
import sys
import glob
import copy
import subprocess

import setuptools
from setuptools.command.build_ext import build_ext

import tensorflow as tf


def _find_cuda_home():
    '''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        if sys.platform == 'win32':
            cuda_home = glob.glob(
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
        else:
            cuda_home = '/usr/local/cuda'
        if not os.path.exists(cuda_home):
            # Guess #3
            try:
                which = 'where' if sys.platform == 'win32' else 'which'
                nvcc = subprocess.check_output(
                    [which, 'nvcc']).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = None
    if cuda_home and not tf.test.is_built_with_cuda():
        print("No CUDA runtime is found, using CUDA_HOME='{}'"
              .format(cuda_home))
    return cuda_home


CUDA_HOME = _find_cuda_home()


class BuildExtension(build_ext):

    def build_extensions(self):
        # Register .cu and .cuh as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def unix_wrap_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = _join_cuda_home('bin', 'nvcc')
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags += ['--compiler-options', "'-fPIC'"]
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                # NVCC does not allow multiple -std to be passed, so we avoid
                # overriding the option if the user explicitly passed it.
                if not any(flag.startswith('-std=') for flag in cflags):
                    cflags.append('-std=c++11')

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def win_wrap_compile(sources,
                             output_dir=None,
                             macros=None,
                             include_dirs=None,
                             debug=0,
                             extra_preargs=None,
                             extra_postargs=None,
                             depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files

                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []
                        cmd = [
                            nvcc, '-c', src, '-o', obj, '-Xcompiler',
                            '/wd4819', '-Xcompiler', '/MD'
                        ] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = self.cflags
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        # Monkey-patch the _compile method.
        if self.compiler.compiler_type == 'msvc':
            self.compiler.compile = win_wrap_compile
        else:
            self.compiler._compile = unix_wrap_compile

        build_ext.build_extensions(self)


def CppExtension(name, sources, *args, **kwargs):
    kwargs['include_dirs'] = kwargs.get('include_dirs', []) + include_paths()
    kwargs = add_tf_flags(kwargs)
    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    kwargs['include_dirs'] = kwargs.get('include_dirs', []) + include_paths()
    kwargs['library_dirs'] = kwargs.get('library_dirs', []) + library_paths()
    kwargs['libraries'] = kwargs.get('libraries', []) + ['cudart']

    kwargs = add_tf_flags(kwargs)
    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def add_tf_flags(kwargs):
    flags = list(kwargs.get('extra_compile_args', []))
    if isinstance(flags, dict):
        for k in flags:
            flags[k] += tf.sysconfig.get_compile_flags()
    else:
        flags += tf.sysconfig.get_compile_flags()
    kwargs['extra_compile_args'] = flags

    flags = list(kwargs.get('extra_link_args', []))
    if isinstance(flags, dict):
        for k in flags:
            flags[k] += tf.sysconfig.get_link_flags()
    else:
        flags += tf.sysconfig.get_link_flags()
    kwargs['extra_link_args'] = flags

    return kwargs


def include_paths(cuda=False):
    here = os.path.abspath(__file__)
    torch_path = os.path.dirname(os.path.dirname(here))
    lib_include = os.path.join(torch_path, 'lib', 'include')
    paths = [lib_include]
    if cuda:
        paths.append(_join_cuda_home('include'))
    return paths


def library_paths(cuda=False):
    paths = []
    if sys.platform == 'win32':
        here = os.path.abspath(__file__)
        torch_path = os.path.dirname(os.path.dirname(here))
        lib_path = os.path.join(torch_path, 'lib')
        paths.append(lib_path)
    if cuda:
        lib_dir = 'lib/x64' if sys.platform == 'win32' else 'lib64'
        paths.append(_join_cuda_home(lib_dir))
    return paths


def _join_cuda_home(*paths):
    '''
    Joins paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.
    This is basically a lazy way of raising an error for missing $CUDA_HOME
    only once we need to get any CUDA-specific path.
    '''
    if CUDA_HOME is None:
        raise EnvironmentError('CUDA_HOME environment variable is not set. '
                               'Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)


def _is_cuda_file(path):
    return os.path.splitext(path)[1] in ['.cu', '.cuh']
