# -*- coding: utf-8 -*-

__all__ = ["Apply", "COp", "Op", "Params", "ParamsType"]

# Yep...
try:
    from theano.graph.basic import Apply
    from theano.graph.op import ExternalCOp as COp
    from theano.graph.op import Op
    from theano.graph.params_type import Params, ParamsType
except ImportError:
    from theano.gof.graph import Apply
    from theano.gof.op import COp, Op
    from theano.gof.params_type import Params, ParamsType
