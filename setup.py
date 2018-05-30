#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension

import tensorflow as tf

link_args = ["-march=native", "-mmacosx-version-min=10.9"]
args = ["-O2", "-std=c++14", "-stdlib=libc++"] + link_args
ext_modules = [
    Extension(
        "interp.interp_ops",
        [os.path.join("interp", "regular_interp.cc")],
        language="c++",
        extra_compile_args=args+tf.sysconfig.get_compile_flags(),
        extra_link_args=link_args + tf.sysconfig.get_link_flags(),
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
