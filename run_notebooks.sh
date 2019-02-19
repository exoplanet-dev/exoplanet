#!/bin/bash

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

cd /mnt/home/dforeman/research/projects/dfm/exoplanet_auto
conda activate autoexoplanet

git checkout master
git pull origin master
python setup.py develop

git branch -D auto_notebooks
git checkout -b auto_notebooks master

cd docs
conda env export > auto_environment.yml

python run_notebooks.py $*

git -c user.name='exoplanetbot' -c user.email='exoplanetbot' commit -am "updating notebooks [ci skip]"
git push -q -f https://dfm:`cat .github_api_key`@github.com/dfm/exoplanet.git auto_notebooks

cd ..
git checkout master
