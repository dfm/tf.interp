#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension

import tensorflow as tf

args = ["-O2", "-std=c++14", "-stdlib=libc++"]
ext_modules = [
    Extension(
        "interp.regular_op",
        [os.path.join("interp", "regular", "regular_interp.cc")],
        # include_dirs=["dr25", ],
        language="c++",
        extra_compile_args=args+tf.sysconfig.get_compile_flags(),
        extra_link_args=tf.sysconfig.get_link_flags(),
    ),
]

setup(
    name="interp",
    version="0.0.0",
    author="Dan Foreman-Mackey",
    ext_modules=ext_modules,
    install_requires=["tensorflow", "numpy"],
    zip_safe=False,
)
