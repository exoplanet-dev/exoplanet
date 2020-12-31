# -*- coding: utf-8 -*-

__all__ = ["add_citations_to_model", "get_citations_for_model"]

import logging

import pymc3 as pm

from exoplanet.citations import CITATIONS, format_citations


def add_citations_to_model(citations, model=None):
    try:
        model = pm.modelcontext(model)
        if not hasattr(model, "__citations__"):
            model.__citations__ = dict()
        for k in citations:
            model.__citations__[k] = CITATIONS[k]

    except TypeError:
        pass


def get_citations_for_model(model=None, width=79):
    """Get the citations for the components used an exoplanet PyMC3

    Returns: The acknowledgement text for exoplanet and its dependencies and a
    string containing the BibTeX entries for the citations in the
    acknowledgement.

    """
    model = pm.modelcontext(model)
    if not hasattr(model, "__citations__"):
        logging.warning("no citations registered with model")
        return "", ""
    return format_citations(model.__citations__, width=width)
