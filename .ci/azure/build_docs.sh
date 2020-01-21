#!/bin/bash -x

set -e

. $CONDA/etc/profile.d/conda.sh
conda activate ./env

# We need pandoc to convert the notebooks
conda install -y -q pandoc

# Build the docs
cd docs
make dirhtml

# Rename master to latest because of RTDs
if [[ "$SOURCE_BRANCH_NAME" = "master" ]]; then
    SOURCE_BRANCH_NAME="latest"
fi

# Clone the existing snapshot of the docs
mkdir _render
cd _render
git clone -b gh-pages --single-branch https://github.com/exoplanet-dev/exoplanet.git .

# Reset git and copy over the docs
rm -rf .git
rm -rf en/$SOURCE_BRANCH_NAME/*
mkdir -p en/$SOURCE_BRANCH_NAME
mv ../_build/dirhtml/* en/$SOURCE_BRANCH_NAME/

# Deal with releases
if [[ "$SOURCE_BRANCH_NAME" =~ ^v[0-9].*  ]]; then
    echo "This is a release: $SOURCE_BRANCH_NAME"
    cd en
    rm -rf stable
    ln -s $SOURCE_BRANCH_NAME stable
    cd ..
fi

python ../../.ci/azure/update_versions.py

# Push back to Github
git init
touch .nojekyll
git add .nojekyll
git add -f *
git -c user.name='exoplanet-doc-bot' -c user.email='exoplanet-doc-bot@azure' \
    commit -m "rebuild gh-pages for ${SOURCE_BRANCH_NAME}"
git push -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/exoplanet-dev/exoplanet.git \
    HEAD:gh-pages >/dev/null 2>&1 -q
