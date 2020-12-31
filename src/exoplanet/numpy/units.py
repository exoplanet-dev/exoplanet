# -*- coding: utf-8 -*-

__all__ = ["with_unit", "has_unit", "to_unit"]

import astropy.units as u


def with_unit(obj, unit):
    """Decorate a numpy array with Astropy units

    Args:
        obj: The array
        unit (astropy.Unit): The units for this object

    """
    return u.Quantity(obj, unit)


def has_unit(obj):
    """Does an object have units as defined by exoplanet?"""
    return hasattr(obj, "unit")


def to_unit(obj, target):
    """Convert a numpy array with units to a target set of units

    Args:
        obj: The array
        target (astropy.Unit): The target units

    Returns:
        An array in the right units

    """
    return u.Quantity(obj, target).value
