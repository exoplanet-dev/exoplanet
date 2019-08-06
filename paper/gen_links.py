import os
import subprocess

hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")[
    :-1
]
slug = "dfm/exoplanet"
with open("gitlinks.tex", "w") as f:
    print(
        r"\newcommand{\codelink}[1]{\href{https://github.com/%s/blob/%s/paper/figures/#1.ipynb}{\codeicon}\,\,}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\animlink}[1]{\href{https://github.com/%s/blob/%s/paper/figures/#1.gif}{\animicon}\,\,}"
        % (slug, hash),
        file=f,
    )
    print(
        r"\newcommand{\prooflink}[1]{\href{https://github.com/%s/blob/%s/paper/proofs/#1.ipynb}{\raisebox{-0.1em}{\prooficon}}}"
        % (slug, hash),
        file=f,
    )
