#!/usr/bin/env python

# Inspired by:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

import codecs
import os
import re

from setuptools import find_packages, setup

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
# TODO: Update exoplanet-core version for pymc5
INSTALL_REQUIRES = ["exoplanet-core>=0.2.0", "astropy>=3.1"]
EXTRA_REQUIRE = {
    "pymc3": [
        "pymc3>=3.9",
        "numpy<1.22",
        "aesara-theano-fallback>=0.0.2",
        "xarray<2023.10.0",
    ],
    "pymc": ["pymc>=5.0.0"],
    # TODO: Update pymc-ext and celerite2 versions for pymc5
    "extras": ["pymc>=5.0.0", "pymc-ext>=1.0.0", "celerite2>=0.2.0"],
    "test": ["scipy", "pytest", "pytest-env"],
    # TODO: Update pymc-ext version for pymc5
    "docs": [
        "pymc>=5.0.0",
        "pymc-ext>=1.0.0",
        "sphinx-book-theme",
        "myst-nb",
        "ipython",
        "matplotlib",
        "corner",
        "emcee",
    ],
}

# END PROJECT SPECIFIC


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
        python_requires=">=3.6",
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        classifiers=CLASSIFIERS,
        zip_safe=False,
    )
