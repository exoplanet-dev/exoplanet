Steps to perform when doing a release:

1. Update changelog date in `HISTORY.rst`
2. Update citation date in `exoplanet/citations.py`.
3. Run `run_notebooks.sh` to make sure that the tutorials all run (push the changes to GitHub)
4. Tag a GitHub release
5. `python setup.py sdist` `twine upload dist/whatever`
