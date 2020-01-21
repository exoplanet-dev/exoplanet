#!/bin/bash

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

cd /mnt/home/dforeman/research/projects/exoplanet-dev/exoplanet_auto
conda activate autoexoplanet

CACHEDIR=/mnt/home/dforeman/research/projects/exoplanet-dev/exoplanet_auto/theano_cache
rm -rf $CACHEDIR
export THEANO_FLAGS=base_compiledir=$CACHEDIR

git checkout master
git pull origin master
python setup.py develop

git branch -D auto_notebooks
git checkout -b auto_notebooks master

cd docs
conda env export > auto_environment.yml

python run_notebooks.py $*

cp notebooks/notebook_setup.py _static/notebooks/notebook_setup.py
git add _static/notebooks/notebook_setup.py
git add _static/notebooks/*.ipynb

git -c user.name='exoplanetbot' -c user.email='exoplanetbot' commit -am "updating notebooks [ci skip]"
git push -q -f https://dfm:`cat .github_api_key`@github.com/exoplanet-dev/exoplanet.git auto_notebooks

cd ..
git checkout master

mail -s "autoexoplanet finished" "foreman.mackey@gmail.com" <<EOF
run_notebooks finished running
EOF
