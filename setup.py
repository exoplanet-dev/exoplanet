#!/usr/bin/env python

# Inspired by:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

import codecs
import os
import re
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# PROJECT SPECIFIC

NAME = "exoplanet"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "exoplanet", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]
SETUP_REQUIRES = ["setuptools>=40.6.0", "setuptools_scm"]
INSTALL_REQUIRES = [
    "pybind11>=2.4",
    "theano>=1.0.4",
    "numpy>=1.13.0",
    "pymc3>=3.5",
    "astropy>=3.1",
    "pymc3-ext>=0.0.1",
]
EXTRA_REQUIRE = {
    "test": [
        "scipy",
        "nose",
        "parameterized",
        "arviz",
        "pytest",
        "pytest-cov>=2.6.1",
        "pytest-env",
        "coveralls",
        "pybind11",
        "batman-package",
        "rebound; sys_platform != 'win32'",
        "starry; sys_platform != 'win32'",
    ],
    "docs": [
        "sphinx>=1.7.5",
        "pandoc",
        "jupyter",
        "ipywidgets",
        "sphinx-typlog-theme",
        "nbformat",
        "nbconvert",
        "corner",
        "lightkurve",
        "jupytext",
        "rtds_action",
        "nbsphinx",
    ],
    "tutorials": [
        "jupytext",
        "jupyter",
        "nbconvert",
        "matplotlib",
        "corner",
        "lightkurve",
    ],
    "nbody": [
        "rebound; sys_platform != 'win32'",
        "rebound_pymc3>=0.0.3; sys_platform != 'win32'",
    ],
}
EXTRA_REQUIRE["dev"] = (
    EXTRA_REQUIRE["test"]
    + EXTRA_REQUIRE["docs"]
    + EXTRA_REQUIRE["nbody"]
    + [
        "pre-commit",
        "black",
        "black_nbconvert",
        "isort",
        "toml",
        "flake8",
        "nbstripout",
        "jupytext",
        "radvel",
        "jupyterlab",
        "lightkurve",
        "pep517",
        "twine",
    ]
)

# END PROJECT SPECIFIC

# PYBIND11


class get_pybind_include:
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


class get_numpy_include:
    def __str__(self):
        import numpy

        return numpy.get_include()


class custom_build_ext(build_ext):
    c_opts = {"msvc": ["/EHsc"], "unix": []}
    l_opts = {"msvc": [], "unix": []}

    if sys.platform == "darwin":
        darwin_opts = [
            "-stdlib=libc++",
            "-mmacosx-version-min=10.14",
            "-march=native",
        ]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def has_flag(self, flagname):
        import tempfile

        import setuptools

        with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
            f.write("int main (int argc, char **argv) { return 0; }")
            try:
                self.compiler.compile([f.name], extra_postargs=[flagname])
            except setuptools.distutils.errors.CompileError:
                return False
        return True

    def cpp_flag(self):
        flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

        for flag in flags:
            if self.has_flag(flag):
                return flag

        raise RuntimeError(
            "Unsupported compiler. At least C++11 support is needed."
        )

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(
                '-DVERSION_INFO="%s"' % self.distribution.get_version()
            )
            opts.append(self.cpp_flag())
            if self.has_flag("-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append(
                '/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version()
            )
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


include_dirs = [
    "src/exoplanet/theano_ops/lib/include",
    "src/exoplanet/theano_ops/lib/vendor/eigen",
    get_numpy_include(),
    get_pybind_include(),
    get_pybind_include(user=True),
]
ext_modules = [
    Extension(
        "exoplanet.theano_ops.driver",
        ["src/exoplanet/theano_ops/driver.cpp"],
        include_dirs=include_dirs,
        language="c++",
    )
]

# END PYBIND11

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        use_scm_version={
            "write_to": os.path.join(
                "src", NAME, "{0}_version.py".format(NAME)
            ),
            "write_to_template": '__version__ = "{version}"\n',
        },
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_dir={"": "src"},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        classifiers=CLASSIFIERS,
        setup_requires=SETUP_REQUIRES,
        zip_safe=False,
        ext_modules=ext_modules,
        cmdclass={"build_ext": custom_build_ext},
    )
