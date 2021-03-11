# -*- coding: utf-8 -*-

__all__ = ["ReboundOp"]

import warnings

import aesara_theano_fallback.tensor as tt
import numpy as np
from aesara_theano_fallback import aesara as theano

from ..utils import as_tensor_variable
from .compat import Apply, Op


class ReboundOp(Op):

    __props__ = ()

    def __init__(self, **rebound_args):
        warnings.warn("For better performance, install 'rebound_pymc3'")
        self.rebound_args = rebound_args
        super(ReboundOp, self).__init__()

    def make_node(self, masses, initial_coords, times):
        in_args = [
            as_tensor_variable(masses),
            as_tensor_variable(initial_coords),
            as_tensor_variable(times),
        ]
        dtype = theano.config.floatX
        out_args = [
            tt.TensorType(dtype=dtype, broadcastable=[False] * 3)(),
            tt.TensorType(dtype=dtype, broadcastable=[False] * 5)(),
        ]
        return Apply(self, in_args, out_args)

    def infer_shape(self, *args):
        shapes = args[-1]
        return (
            list(shapes[2]) + list(shapes[0]) + [6],
            list(shapes[2]) + list(shapes[0]) + [6] + list(shapes[0]) + [7],
        )

    def perform(self, node, inputs, outputs):
        # NOTE: Units should be AU, M_sun, year/2pi
        import rebound

        masses, initial_coords, times = inputs
        masses = np.atleast_1d(masses)
        initial_coords = np.atleast_2d(initial_coords)
        times = np.atleast_1d(times)

        if len(np.shape(masses)) != 1:
            raise ValueError("the array of masses must be 1D")
        num_bodies = len(masses)

        if np.shape(initial_coords) != (num_bodies, 6):
            raise ValueError(
                "the initial coordinates must have shape {0}".format(
                    (num_bodies, 6)
                )
            )

        if len(np.shape(times)) != 1:
            raise ValueError("the array of times must be 1D")
        num_times = len(times)
        time_inds = np.argsort(times)
        sorted_times = times[time_inds]

        # Set up the simulation
        sim = rebound.Simulation()
        gr_sources = []
        for k, v in self.rebound_args.items():
            if "gr" in k and "force" not in k:
                gr_sources += [v]
                continue
            setattr(sim, k, v)

        if "gr_force" in self.rebound_args.keys():
            force = self.rebound_args["gr_force"]
        else:
            force = "gr"  # default to the gr force

        for i in range(num_bodies):
            sim.add(
                m=masses[i],
                x=initial_coords[i, 0],
                y=initial_coords[i, 1],
                z=initial_coords[i, 2],
                vx=initial_coords[i, 3],
                vy=initial_coords[i, 4],
                vz=initial_coords[i, 5],
            )

        ps = sim.particles

        if len(gr_sources) != 0:  # if gr_sources have been added
            try:
                import reboundx
                from reboundx import constants
            except ImportError:
                raise ImportError(
                    """Please install REBOUNDx to include
                                relativistic effects."""
                )
            rebx = reboundx.Extras(sim)
            gr = rebx.load_force(force)
            gr.params["c"] = constants.C
            for i, particle in enumerate(ps):
                particle.params["gr_source"] = gr_sources[i]
            rebx.add_force(gr)
        # Add the variational particles to track the derivatives
        var_systems = np.empty((num_bodies, 7), dtype=object)
        for i in range(num_bodies):
            for j, coord in enumerate("m x y z vx vy vz".split()):
                var = sim.add_variation()
                setattr(var.particles[i], coord, 1.0)
                var_systems[i, j] = var

        # Integrate the system
        coords = np.empty((num_times, num_bodies, 6))
        jac = np.empty((num_times, num_bodies, 6, num_bodies, 7))
        for ind in range(num_times):
            sim.integrate(sorted_times[ind])

            # Save the coordinates at this time
            for i in range(num_bodies):
                for j, coord in enumerate("x y z vx vy vz".split()):
                    coords[ind, i, j] = getattr(sim.particles[i], coord)

            # Save the jacobian at this time
            for i in range(num_bodies):
                for j, coord in enumerate("x y z vx vy vz".split()):
                    for k in range(num_bodies):
                        for l in range(7):
                            jac[ind, i, j, k, l] = getattr(
                                var_systems[k, l].particles[i], coord
                            )

        # Save the results
        outputs[0][0] = np.ascontiguousarray(coords[time_inds])
        outputs[1][0] = np.ascontiguousarray(jac[time_inds])

    def grad(self, inputs, gradients):
        masses, initial_coords, times = inputs
        coords, jac = self(*inputs)
        bcoords = gradients[0]
        if not isinstance(gradients[1].type, theano.gradient.DisconnectedType):
            raise ValueError(
                "can't propagate gradients with respect to Jacobian"
            )

        # (6, time, num) * (6, time, num, num, 7) -> (num, 7)
        grad = tt.sum(bcoords[:, :, :, None, None] * jac, axis=(0, 1, 2))
        return grad[:, 0], grad[:, 1:], tt.zeros_like(times)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
