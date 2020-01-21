#!/bin/bash -x

set -e

if [[ -n "$GITHUB_API_KEY" && "$TRAVIS_PULL_REQUEST" = "false" ]]
then
  if [ -n "$TRAVIS_TAG" ]
  then
    git clone --recursive --depth=50 https://github.com/exoplanet-dev/exoplanet-docs.git
  else
    git clone --recursive --depth=50 --branch=$TRAVIS_BRANCH https://github.com/exoplanet-dev/exoplanet-docs.git
  fi

  cd exoplanet-docs/exoplanet
  git checkout $TRAVIS_COMMIT
  cd ..
  git add exoplanet
  git -c user.name='travis' -c user.email='travis' commit -m "updating exoplanet [ci skip]"

  if [ -n "$TRAVIS_TAG" ]
  then
    git tag -a $TRAVIS_TAG -m "exoplanet $TRAVIS_TAG"
  fi

  git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/exoplanet-dev/exoplanet-docs.git $TRAVIS_BRANCH
fi
