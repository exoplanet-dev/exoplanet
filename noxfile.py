import nox

ALL_PYTHON_VS = ["3.8", "3.9", "3.10"]


@nox.session(python=ALL_PYTHON_VS)
def test_pymc3(session):
    session.install(".[test,pymc3]")
    session.install("batman-package", "starry")
    session.run("python", "-c", "import theano")
    session.run("python", "-c", "import exoplanet_core.pymc3.ops")
    session.run("pytest", "-v", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def test_pymc(session):
    session.install(".[test,pymc]")
    session.install("batman-package")
    session.run("python", "-c", "import aesara")
    session.run("python", "-c", "import exoplanet_core.pymc4.ops")
    session.run("pytest", "-v", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)
