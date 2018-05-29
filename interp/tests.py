#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator

from .ops import regular_op


class RegularTest(tf.test.TestCase):

    def test_regular(self):
        np.random.seed(42)
        with self.test_session():
            shape_in = []

            for s_in in [10, 5, 6, 8]:
                shape_in.append(s_in)
                shape_out = []
                for s_out in [5, 3, 1]:
                    shape_out.append(s_out)

                    points = [np.linspace(1, 2, s) for s in shape_in]
                    values = np.prod([np.sin(x)
                                      for x in np.meshgrid(*points,
                                                           indexing="ij")],
                                     axis=0)
                    xi = np.random.uniform(1, 2, shape_out + [len(shape_in)])
                    Z, _ = regular_op.interp_regular(points, values, xi)

                    interp = RegularGridInterpolator(points, values)
                    Z_ref = interp(xi)

                    assert np.allclose(Z.eval(), Z_ref)


if __name__ == "__main__":
    tf.test.main()
