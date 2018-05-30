# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["regular_nd"]

import os
import sysconfig
import tensorflow as tf


def _load_library(name):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    dirname = os.path.dirname(os.path.abspath(__file__))
    libfile = os.path.join(dirname, name)
    if suffix is not None:
        libfile += suffix
    else:
        libfile += ".so"
    return tf.load_op_library(libfile)


interp_ops = _load_library("interp_ops")


def regular_nd(*args, **kwargs):
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
    return interp_ops.interp_regular(*args, **kwargs)[0]


@tf.RegisterGradient("InterpRegular")
def _interp_regular_rev(op, *grads):
    xi = op.inputs[-1]
    Z = op.outputs[0]
    dZ = op.outputs[1]
    bZ = grads[0]
    nx = tf.size(tf.shape(xi))
    axes = tf.range(nx, tf.size(tf.shape(Z))+1)
    bxi = tf.reduce_sum(dZ * tf.expand_dims(bZ, nx-1), axis=axes)
    return tuple([None for i in range(len(op.inputs)-1)] + [bxi])
