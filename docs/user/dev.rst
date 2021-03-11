.. _dev:

Developer documentation
=======================

Development of ``exoplanet`` is actively happening `on GitHub
<https://github.com/exoplanet-dev/exoplanet>`_ and we would love your
contributions. There are a few different methods of contributing to
``exoplanet`` and the details are discussed below.

Reporting an issue
------------------

If you run into issues, bugs, or anything else, it is very useful if you can
`post an issue on the GitHub repository
<https://github.com/exoplanet-dev/exoplanet/issues>`_. When you post an issue,
please provide the details to reproduce the issue. For example, if you find a
bug, please provide a *standalone* and *executable* snippet of code that
demonstrates the issue. It's also useful to include details about your platform
and the versions of key packages that your using.


Contributing code or documentation
----------------------------------

If you're not familiar with the workflow for contributing code to a GitHub
repository, an excellent place to start is the `AstroPy developer docs
<https://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_.


Set up your development environment
+++++++++++++++++++++++++++++++++++

After getting some familiarity with the workflow, you should fork `the exoplanet
repository <https://github.com/exoplanet-dev/exoplanet>`_ and clone it to your
local machine:

.. code-block:: bash

    git clone --recursive https://github.com/YOURUSERNAME/exoplanet.git
    cd exoplanet
    git checkout -b BRANCHNAME

for some name ``BRANCHNAME`` describing your contribution.

Then you should set up an isolated environment for development using a `Conda
environment
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_,
`virtualenv <https://virtualenv.pypa.io/>`_, `venv
<https://docs.python.org/3/library/venv.html>`_, or similar.

If you're using a ``pip`` based environment, you can install the developer
dependencies as follows:

.. code-block:: bash

    python -m pip install -U pip
    python -m pip install -U -e ".[dev]"


Finding your way around the codebase
++++++++++++++++++++++++++++++++++++

``exoplanet`` is mostly arranged as a typical Python project with the module
root in the ``src/exoplanet`` directory. But there are a few directions that can
be useful before diving in:

1. **Tutorials**: The tutorials are written using the `jupytext
<https://github.com/mwouts/jupytext>`_ tool to keep the size of the repository
from blowing up. The tutorial files are saved (using jupytext) in the
``docs/tutorials`` directory as ``.py`` files in the ``py:light`` format. You
can edit these files as if they are Jupyter notebooks by using the extension:

.. code-block:: bash

    jupyter nbextension install --py jupytext
    jupyter nbextension enable --py jupytext

If you are contributing a new tutorial, you should copy one of the existing ones
and try to follow roughly the same format.

2. **Case studies**: These are more in depth tutorials that require more
computational run time. These can be found in the `case studies
<https://github.com/exoplanet-dev/case-studies>`_ repo and there is more
information there about how to contribute.

3. **Theano/Aesara ops**: ``exoplanet`` comes bundled with a set of custom ops
that are implemented in ``src/exoplanet/theano_ops``. As a user, you'll rarely
interact with these directly and we haven't put a lot of work into making them
user friendly, but if you are interested in diving in, feel free to ask
questions on GitHub or via email.


Testing your contribution
+++++++++++++++++++++++++

If you're contributing a change to the code (either a new feature or bug fix),
make sure that you implement at least one test that checks the behavior of your
code. Then, you should run all of the unit tests before submitting a pull
request using the following command:

.. code-block:: bash

    python -m pytest -v tests
