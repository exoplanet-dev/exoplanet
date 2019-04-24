#!/bin/bash -x
set -e

# Are there changes in the tex directory?
if [ "$TRAVIS_PULL_REQUEST" = "false" ]
then

    # Build the paper
    echo "Building the paper..."

    cd $TRAVIS_BUILD_DIR/paper
    make

    cd $TRAVIS_BUILD_DIR
    git checkout --orphan $TRAVIS_BRANCH-pdf
    git rm -rf .
    git add -f paper/exoplanet.pdf
    git add -f paper/notebooks/*.pdf
    git add -f paper/notebooks/notebook_errors.log
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf >/dev/null 2>&1

fi
