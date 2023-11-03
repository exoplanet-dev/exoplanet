import os

if os.environ.get("EXOPLANET_FORCE_PYMC3", "0") == "1":
    import pymc3 as pm

    USING_PYMC3 = True

else:
    try:
        import pymc as pm

        USING_PYMC3 = False

    except ImportError:
        try:
            import pymc3 as pm

            USING_PYMC3 = True

        except ImportError:
            raise ImportError(
                "No compatible version of PyMC found. "
                "Please install either pymc3 or pymc>=4."
            )

if USING_PYMC3:
    from exoplanet_core.pymc3 import ops as ops
    from theano import function as function
    from theano import grad as grad
    from theano import tensor as tensor
    from theano.ifelse import ifelse as ifelse

    tensor.abs = tensor.abs_

    try:
        from theano.assert_op import Assert as Assert
    except ImportError:
        from theano.tensor.opt import Assert as Assert

    from theano.gradient import verify_grad as verify_grad

    try:
        import theano

        change_flags = theano.config.change_flags
    except (ImportError, AttributeError):
        from theano.configparser import change_flags as change_flags

else:
    import pytensor
    from pytensor import function as function
    from pytensor import grad as grad
    from pytensor import tensor as tensor

    change_flags = pytensor.config.change_flags
    from exoplanet_core.pymc import ops as ops
    from pytensor.gradient import verify_grad as verify_grad
    from pytensor.ifelse import ifelse as ifelse
    from pytensor.raise_op import Assert as Assert
