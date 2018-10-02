# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "LinearInterpolator", "CubicInterpolator",
    "RegularGridInterpolator",
]

import tensorflow as tf

from .tf_utils import load_op_library
from .tri_diag_solve import tri_diag_solve


linear_op = load_op_library("linear_op")
cubic_op = load_op_library("cubic_op")
regular_op = load_op_library("regular_op")


class LinearInterpolator(object):
    """Linear interpolation for a scalar function in one dimension

    Args:
        x (..., N): The independent coordinates of the training points.
        y (..., N): The dependent coordinates of the training points. This
            must be the same shape as ``x`` and the interpolation is always
            performed along the last axis.

    """

    def __init__(self, x, y, name=None):
        self.name = name
        self.x = x
        self.y = y

    def evaluate(self, t, name=None):
        """Interpolate the training points linearly

        Args:
            t (..., M): The independent coordinates where the model should be
                evaluated. The dimensions of all but the last axis must match
                the dimensions of ``x``. The interpolation is performed
                in the last dimension independently for each of the earlier
                dimensions.

        """
        with tf.name_scope(self.name, "LinearInterpolator"):
            return linear_op.linear_interp(t, self.x, self.y, name=name)[0]


class CubicInterpolator(object):
    """Cubic spline interpolation for a scalar function in one dimension

    Args:
        x (..., N): The independent coordinates of the training points.
        y (..., N): The dependent coordinates of the training points. This
            must be the same shape as ``x`` and the interpolation is always
            performed along the last axis.
        fpa: The value of the derivative of the function at the first data
            point. By default this is zero.
        fpb: The value of the derivative of the function at the last data
            point. By default this is zero.

    """

    def __init__(self, x, y, fpa=None, fpb=None, name=None):
        self.name = name
        with tf.name_scope(name, "CubicInterpolator"):
            # Compute the deltas
            size = tf.shape(x)[-1]
            axis = tf.rank(x) - 1
            dx = tf.gather(x, tf.range(1, size), axis=axis) \
                - tf.gather(x, tf.range(size-1), axis=axis)
            dy = tf.gather(y, tf.range(1, size), axis=axis) \
                - tf.gather(y, tf.range(size-1), axis=axis)

            # Compute the slices
            upper_inds = tf.range(1, size-1)
            lower_inds = tf.range(size-2)
            s_up = lambda a: tf.gather(a, upper_inds, axis=axis)  # NOQA
            s_lo = lambda a: tf.gather(a, lower_inds, axis=axis)  # NOQA
            dx_up = s_up(dx)
            dx_lo = s_lo(dx)
            dy_up = s_up(dy)
            dy_lo = s_lo(dy)

            first = lambda a: tf.gather(a, tf.zeros(1, dtype=tf.int64),  # NOQA
                                        axis=axis)
            last = lambda a: tf.gather(a, [size-2], axis=axis)  # NOQA

            fpa_ = fpa if fpa is not None else tf.constant(0, x.dtype)
            fpb_ = fpb if fpb is not None else tf.constant(0, x.dtype)

            diag = 2*tf.concat((first(dx), dx_up+dx_lo, last(dx)), axis)
            upper = dx
            lower = dx
            Y = 3*tf.concat((first(dy)/first(dx) - fpa_,
                             dy_up/dx_up - dy_lo/dx_lo,
                             fpb_ - last(dy)/last(dx)), axis)

            # Solve the tri-diagonal system
            c = tri_diag_solve(diag, upper, lower, Y)
            c_up = tf.gather(c, tf.range(1, size), axis=axis)
            c_lo = tf.gather(c, tf.range(size-1), axis=axis)
            b = dy / dx - dx * (c_up + 2*c_lo) / 3
            d = (c_up - c_lo) / (3*dx)

            self.x = x
            self.y = y
            self.b = b
            self.c = c_lo
            self.d = d

    def evaluate(self, t, name=None):
        """Interpolate the training points using a cubic spline

        Args:
            t (..., M): The independent coordinates where the model should be
                evaluated. The dimensions of all but the last axis must match
                the dimensions of ``x``. The interpolation is performed
                in the last dimension independently for each of the earlier
                dimensions.

        """
        with tf.name_scope(self.name, "CubicInterpolator"):
            with tf.name_scope(name, "evaluate"):
                res = cubic_op.cubic_gather(t, self.x, self.y, self.b, self.c,
                                            self.d)
                tau = t - res.xk
                mod = res.ak + res.bk * tau + res.ck * tau**2 + res.dk * tau**3
                return mod


class RegularGridInterpolator(object):
    """Linear interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a filled regular grid, but the spacing may be
    uneven in any of the dimensions.

    Args:
        points: A list of ``Tensor`` objects with shapes ``(m1,), ... (mn,)``.
            These tensors define the grid points in each dimension.
        values: A ``Tensor`` defining the values at each point in the grid
            defined by ``points``. This must have the shape
            ``(m1, ... mn, ...)``.
        xi: A ``Tensor`` defining the coordinates where the interpolation
            should be evaluated.
        check_sorted: If ``True`` (default), check that the tensors in
            ``points`` are all sorted in ascending order. This can be set to
            ``False`` if the axes are known to be sorted, but the results will
            be unpredictable if this ends up being wrong.
        bounds_error: If ``False`` (default) extrapolate beyond the edges of
            the grid. Otherwise raise an exception.
        name: A name for the operation (optional).

    """

    def __init__(self, points, values, check_sorted=True, bounds_error=False,
                 name=None):
        self.points = points
        self.values = values
        self.check_sorted = check_sorted
        self.bounds_error = bounds_error
        self.name = name

    def evaluate(self, t, name=None):
        with tf.name_scope(self.name, "RegularGridInterpolator"):
            return regular_op.regular_interp(
                self.points, self.values, t,
                check_sorted=self.check_sorted,
                bounds_error=self.bounds_error,
                name=self.name)[0]


@tf.RegisterGradient("LinearInterp")
def _linear_interp_rev(op, *grads):
    t, x, y = op.inputs
    v, inds = op.outputs
    bv = grads[0]
    bt, by = linear_op.linear_interp_rev(t, x, y, inds, bv)
    return [bt, None, by]


@tf.RegisterGradient("CubicGather")
def _cubic_gather_rev(op, *grads):
    x = op.inputs[1]
    inds = op.outputs[-1]
    args = [x, inds] + list(grads)
    results = cubic_op.cubic_gather_rev(*args)
    return [tf.zeros_like(op.inputs[0])] + list(results)


@tf.RegisterGradient("RegularInterp")
def _regular_interp_rev(op, *grads):
    xi = op.inputs[-1]
    Z = op.outputs[0]
    dZ = op.outputs[1]
    bZ = grads[0]
    nx = tf.size(tf.shape(xi))
    axes = tf.range(nx, tf.size(tf.shape(Z))+1)
    bxi = tf.reduce_sum(dZ * tf.expand_dims(bZ, nx-1), axis=axes)
    return tuple([None for i in range(len(op.inputs)-1)] + [bxi])
