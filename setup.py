#!/usr/bin/env python

import sys
from setuptools import setup

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__EXOPLANET_SETUP__ = True
import exoplanet  # NOQA


setup(
    name="exoplanet",
    version=exoplanet.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/exoplanet",
    license="MIT",
    packages=[
        "exoplanet",
        "exoplanet.theano_ops",
        "exoplanet.theano_ops.starry",
    ],
    description="Scalable 1D Gaussian Processes",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE", "CITATION"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=True,
)
