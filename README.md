**Multivariate linear interpolation in TensorFlow**

This module implements a custom TensorFlow operation that replicates the
[RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html)
from SciPy.

To install, run:

```bash
python setup.py install
```

or

```bash
python setup.py develop
```

And then use:

```python
from tfinterp.interp import RegularGridInterpolator as RGI

rgi = RGI(points, values)
zi = rgi.evaluate(xi)
```

in your code. See the docstring for `RegularGridInterpolator` for more information.
