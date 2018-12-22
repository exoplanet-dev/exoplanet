TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun ms.log

default: exoplanet.pdf

exoplanet.pdf: exoplanet.tex
	#tectonic exoplanet.tex
	pdflatex exoplanet
	bibtex   exoplanet
	pdflatex exoplanet
	pdflatex exoplanet

clean:
	$(RM_TMP)
