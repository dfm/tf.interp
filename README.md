**Multivariate linear interpolation in TensorFlow**

This module implements a custom TensorFlow operation that replicates the
[RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html)
from SciPy.

To insall, run:

```bash
python setup.py install
```

or

```bash
python setup.py develop
```

And then use:

```python
from tfinterp import regular_nd
zi = regular_nd(points, values, xi)
```

in your code. See the docstring for `regular_nd` for more information.
