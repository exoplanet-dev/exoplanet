# -*- coding: utf-8 -*-

__all__ = ["add_citations_to_model", "get_citations_for_model"]

from exoplanet.citations import format_citations


def add_citations_to_model(citations, model=None):
    pass


def get_citations_for_model(model=None, width=79):
    return format_citations({}, width=width)
