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
    from aesara import function as function
    from aesara import grad as grad
    from aesara import tensor as tensor
    from aesara.configparser import change_flags
    from aesara.gradient import verify_grad as verify_grad
    from aesara.ifelse import ifelse as ifelse
    from aesara.raise_op import Assert as Assert
    from exoplanet_core.pymc4 import ops as ops
