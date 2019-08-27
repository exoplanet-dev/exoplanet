# -*- coding: utf-8 -*-

__all__ = ["with_unit", "has_unit", "to_unit"]

import theano.tensor as tt

UNIT_ATTR_NAME = "__exoplanet_unit__"


def with_unit(obj, unit):
    """Decorate a Theano tensor with Astropy units

    Args:
        obj: The Theano tensor
        unit (astropy.Unit): The units for this object

    Raises:
        TypeError: If the tensor already has units

    """
    if hasattr(obj, UNIT_ATTR_NAME):
        raise TypeError("{0} already has units".format(repr(obj)))
    obj = tt.as_tensor_variable(obj)
    setattr(obj, UNIT_ATTR_NAME, unit)
    return obj


def has_unit(obj):
    """Does an object have units as defined by exoplanet?"""
    return hasattr(obj, UNIT_ATTR_NAME)


def to_unit(obj, target):
    """Convert a Theano tensor with units to a target set of units

    Args:
        obj: The Theano tensor
        target (astropy.Unit): The target units

    Returns:
        A Theano tensor in the right units

    """
    if not has_unit(obj):
        return obj
    base = getattr(obj, UNIT_ATTR_NAME)
    return obj * base.to(target)
