# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

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


regular_op = _load_library("regular_op")


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
