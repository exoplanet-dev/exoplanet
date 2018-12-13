Steps to perform when doing a release:

1. Bump version number and update changelog date in `HISTORY.rst`
2. `python setup.py sdist` then `pip install dist/whatever` to test
3. From another directory run `py.test -v --pyargs exoplanet` to make sure that everything is good
4. Tag a GitHub release
5. Update the code with the new Zenodo DOI
6. `python setup.py sdist` `twine upload dist/whatever`
