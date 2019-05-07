TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun exoplanet.log
TECTONIC    = $(shell command -v tectonic >/dev/null && echo true || echo false )

default: exoplanet.pdf

exoplanet.pdf: exoplanet.tex exoplanet.bib xostyle.tex
	# Generate links to current git commit
	python gen_links.py
	if [ "${TECTONIC}" = "true" ]; then\
		tectonic exoplanet.tex --print --keep-logs;\
	else\
		${LATEX} exoplanet;\
		( ${CHECK_RERUN} && ${LATEX} exoplanet ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} exoplanet ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} exoplanet ) || echo "Done.";\
	fi

clean:
	$(RM_TMP)

.PHONY: clean