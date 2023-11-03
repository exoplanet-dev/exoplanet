import tempfile
from pathlib import Path

import nox

ALL_PYTHON_VS = ["3.8", "3.9", "3.10"]


@nox.session(python=ALL_PYTHON_VS)
def test_pymc3(session):
    session.install(".[test,pymc3]")
    session.install("batman-package", "starry>=0.9.9", "tqdm")
    session.run("python", "-c", "import theano")
    session.run("python", "-c", "import exoplanet_core.pymc3.ops")
    session.run("python", "-c", "import starry")
    session.run("pytest", "-v", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def test_pymc(session):
    session.install(".[test,pymc]")
    session.install("batman-package")
    session.run("python", "-c", "import pytensor")
    session.run("python", "-c", "import exoplanet_core.pymc.ops")
    session.run("pytest", "-v", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def docs(session):
    session.install(".[docs]")
    session.install("sphinx")
    session.run(
        "sphinx-build",
        "--color",
        "-W",
        "-b",
        "dirhtml",
        "-d",
        str(session.cache_dir / "doctrees"),
        "docs",
        str(session.cache_dir / "dirhtml"),
        *session.posargs
    )
    session.run("open", session.cache_dir)
