Steps to perform when doing a release:

1. Bump version number and update changelog date in `HISTORY.rst`
2. `python setup.py sdist` then `pip install dist/whatever` to test
3. From another directory run `py.test -v --pyargs exoplanet` to make sure that everything is good
4. Run `run_notebooks.sh` to make sure that the tutorials all run (push the changes to GitHub)
5. Tag a GitHub release
6. Update the code with the new Zenodo DOI: in `exoplanet/exoplanet_version.py`.
7. Run `update_zenodo.py` to update all instances of the DOI
8. `python setup.py sdist` `twine upload dist/whatever`
