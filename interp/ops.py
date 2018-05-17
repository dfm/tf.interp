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
