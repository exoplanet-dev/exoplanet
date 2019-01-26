#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/mnt/home/dforeman/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/mnt/home/dforeman/miniconda3/etc/profile.d/conda.sh" ]; then
#        . "/mnt/home/dforeman/miniconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/mnt/home/dforeman/miniconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<

conda shell.bash hook

cd /mnt/home/dforeman/research/projects/dfm/exoplanet_auto
conda activate autoexoplanet

git checkout master
git pull origin master
python setup.py develop

git branch -D auto_notebooks
git checkout -b auto_notebooks master

cd docs
conda env export > auto_environment.yml

python run_notebooks.py

git -c user.name='exoplanetbot' -c user.email='exoplanetbot' commit -am "updating notebooks [ci skip]"
git push -q -f https://dfm:`cat .github_api_key`@github.com/dfm/exoplanet.git auto_notebooks

cd ..
git checkout master
