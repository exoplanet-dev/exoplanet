#!/usr/bin/env python

import os

from setuptools import setup

dirname = os.path.dirname(os.path.realpath(__file__))


def get_requirements():
    with open(os.path.join(dirname, "requirements.txt"), "r") as f:
        return f.read().splitlines()


def get_long_description():
    with open(os.path.join(dirname, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    name="exoplanet",
    use_scm_version={
        "write_to": "exoplanet/exoplanet_version.py",
        "write_to_template": '__version__ = "{version}"\n',
    },
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/exoplanet",
    license="MIT",
    packages=[
        "exoplanet",
        "exoplanet.gp",
        "exoplanet.orbits",
        "exoplanet.theano_ops",
        "exoplanet.distributions",
        "exoplanet.theano_ops.starry",
        "exoplanet.theano_ops.kepler",
        "exoplanet.theano_ops.contact",
        "exoplanet.theano_ops.celerite",
        "exoplanet.theano_ops.interp",
        "exoplanet.theano_ops.rebound",
    ],
    description="Fast & scalable MCMC for all your exoplanet needs",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    install_requires=get_requirements(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=False,
)
