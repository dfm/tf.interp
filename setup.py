#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def get_ext():
    return [
        Extension(
            "tfinterp.interp_ops",
            [os.path.join("tfinterp", "regular_interp.cc")],
            language="c++",
        ),
    ]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def has_library(compiler, libname):
    """Return a boolean indicating whether a library is found."""
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as srcfile:
        srcfile.write("int main (int argc, char **argv) { return 0; }")
        srcfile.flush()
        outfn = srcfile.name + ".so"
        try:
            compiler.link_executable(
                [srcfile.name],
                outfn,
                libraries=[libname],
            )
        except setuptools.distutils.errors.LinkError:
            return False
        if not os.path.exists(outfn):
            return False
        os.remove(outfn)
    return True


def cpp_flag(compiler):
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class build_ext(_build_ext):

    def build_extensions(self):
        import tensorflow as tf
        ct = self.compiler.compiler_type
        compile_flags = tf.sysconfig.get_compile_flags()
        link_flags = tf.sysconfig.get_link_flags()
        if ct == "unix":
            print("testing C++14/C++11 support")
            compile_flags.append(cpp_flag(self.compiler))
            flags = ["-O2", "-stdlib=libc++", "-fvisibility=hidden",
                     "-Wno-unused-function", "-Wno-uninitialized",
                     "-Wno-unused-local-typedefs", "-funroll-loops"]

            # Mac specific flags and libraries
            if sys.platform == "darwin":
                flags += ["-march=native", "-mmacosx-version-min=10.9"]
                for lib in ["m", "c++"]:
                    for ext in self.extensions:
                        ext.libraries.append(lib)
                for ext in self.extensions:
                    link_flags += ["-mmacosx-version-min=10.9"]
            else:
                libraries = ["m", "stdc++", "c++"]
                for lib in libraries:
                    if not has_library(self.compiler, lib):
                        continue
                    for ext in self.extensions:
                        ext.libraries.append(lib)

            # Check the flags
            print("testing compiler flags")
            for flag in flags:
                if has_flag(self.compiler, flag):
                    compile_flags.append(flag)

        for ext in self.extensions:
            ext.extra_compile_args = compile_flags
            ext.extra_link_args = link_flags

        # Run the standard build procedure.
        _build_ext.build_extensions(self)


setup(
    name="tfinterp",
    version="0.0.0",
    author="Dan Foreman-Mackey",
    ext_modules=get_ext(),
    install_requires=["tensorflow", "numpy"],
    cmdclass=dict(build_ext=build_ext),
    zip_safe=True,
)
