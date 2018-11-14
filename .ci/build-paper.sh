#!/bin/bash -x
set -e

# Are there changes in the tex directory?
if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'tex/'
then

    # Build the paper
    echo "Building the paper..."

    # Download HD189733b MCMC chains
    wget https://www.dropbox.com/s/hfi5tvjziu9d37u/map_chain.npz
    mv map_chain.npz $TRAVIS_BUILD_DIR/tex/figures/

    cd $TRAVIS_BUILD_DIR/tex && make

    # If `proofs.rst` changed, let's clone the repo in a temporary
    # directory and commit & push the changes
    if git diff --name-only | grep 'proofs.rst'
    then
        mkdir -p tmp && cd tmp
        git clone https://github.com/rodluger/starry.git && cd starry
        cp $TRAVIS_BUILD_DIR/docs/proofs.rst docs/proofs.rst
        git add docs/proofs.rst
        git commit -m "updating proofs.rst [skip ci]"
        git push https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH || echo "Failed to push `proofs.rst`"
        cd $TRAVIS_BUILD_DIR/tex && rm -rf tmp
    fi

    # Force push the paper to GitHub
    mkdir -p travis && cd travis
    mkdir -p tex && mv ../starry.pdf tex/
    git init
    git add -f tex/starry.pdf
    # If we're publishing the paper, commit the figures as well
    if [[ $TRAVIS_COMMIT_MESSAGE = *"[publish]"* ]]
    then
        mv ../figures tex/
        git add -f tex/figures/*.pdf
    fi
    git status
    git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
    git status
    git push -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG HEAD:$TRAVIS_BRANCH-pdf
    cd $TRAVIS_BUILD_DIR

fi
