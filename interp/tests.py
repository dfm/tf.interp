#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator

from .ops import regular_op


class RegularTest(tf.test.TestCase):

    def test_value(self):
        np.random.seed(42)
        with self.test_session():
            shape_in = []

            for s_in in [10, 5, 6, 8]:
                shape_in.append(s_in)
                shape_out = []
                for s_out in [5, 3, 1]:
                    shape_out.append(s_out)

                    points = [np.linspace(1, 2, s) for s in shape_in]
                    values = np.random.randn(*(shape_in + [1, 10, 2]))
                    xi = np.random.uniform(1, 2, shape_out + [len(shape_in)])
                    Z, _ = regular_op.interp_regular(points, values, xi)

                    interp = RegularGridInterpolator(points, values)
                    Z_ref = interp(xi)

                    assert np.allclose(Z.eval(), Z_ref)

    def test_gradient(self):
        np.random.seed(42)
        with self.test_session():
            shape_in = []

            for s_in in [9, 7, 5]:
                shape_in.append(s_in)
                shape_out = []
                for s_out in [8, 6, 4]:
                    shape_out.append(s_out)

                    points = [np.linspace(1, 2, s) for s in shape_in]
                    values = np.random.randn(*(shape_in + [3, 2, 1]))
                    xi = tf.constant(
                        np.random.uniform(1, 2, shape_out + [len(shape_in)]))
                    Z, _ = regular_op.interp_regular(points, values, xi)

                    xi_val = xi.eval()
                    shape = xi_val.shape

                    self.assertAllCloseAccordingToType(
                        tf.test.compute_gradient_error(
                            [xi], [shape], Z, Z.eval().shape, [xi_val], 1e-8),
                        0.0)


if __name__ == "__main__":
    tf.test.main()
