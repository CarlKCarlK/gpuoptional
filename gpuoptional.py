# License: Apache 2.0
# Carl Kadie
# https://fastlmm.github.io/

import os
import numpy as np
import inspect
import logging
from types import ModuleType

_warn_array_module_once = False


def array_module(xp=None):
    """
    Find the array module to use, for example **numpy** or **cupy**.

    :param xp: The array module to use, for example, 'numpy'
               (normal CPU-based module) or 'cupy' (GPU-based module).
               If not given, will try to read
               from the ARRAY_MODULE environment variable. If not given and
               ARRAY_MODULE is not set,
               will use numpy. If 'cupy' is requested, will
               try to 'import cupy'. If that import fails, will
               revert to numpy.
    :type xp: optional, string or Python module
    :rtype: Python module

    >>> from pysnptools.util import array_module
    >>> xp = array_module() # will look at environment variable
    >>> print(xp.zeros((3)))
    [0. 0. 0.]
    >>> xp = array_module('cupy') # will try to import 'cupy'
    >>> print(xp.zeros((3)))
    [0. 0. 0.]
    """
    xp = xp or os.environ.get("ARRAY_MODULE", "numpy")

    if isinstance(xp, ModuleType):
        return xp

    if xp == "numpy":
        return np

    if xp == "cupy":
        try:
            import cupy as cp

            return cp
        except ModuleNotFoundError as e:
            global _warn_array_module_once
            if not _warn_array_module_once:
                logging.warning(f"Using numpy. ({e})")
                _warn_array_module_once = True
            return np

    raise ValueError(f"Don't know ARRAY_MODULE '{xp}'")


def asnumpy(a):
    """
    Given an array created with any array module, return the equivalent
    numpy array. (Returns a numpy array unchanged.)

    >>> from pysnptools.util import asnumpy, array_module
    >>> xp = array_module('cupy')
    >>> zeros_xp = xp.zeros((3)) # will be cupy if available
    >>> zeros_np = asnumpy(zeros_xp) # will be numpy
    >>> zeros_np
    array([0., 0., 0.])
    """
    if isinstance(a, np.ndarray):
        return a
    return a.get()


def get_array_module(a):
    """
    Given an array, returns the array's
    module, for example, **numpy** or **cupy**.
    Works for numpy even when cupy is not available.

    >>> import numpy as np
    >>> zeros_np = np.zeros((3))
    >>> xp = get_array_module(zeros_np)
    >>> xp.ones((3))
    array([1., 1., 1.])
    """
    submodule = inspect.getmodule(type(a))
    module_name = submodule.__name__.split(".")[0]
    xp = array_module(module_name)
    return xp


if __name__ == "__main__":

    def gen_data(size, seed=1, xp=None):
        xp = array_module(xp)
        rng = xp.random.RandomState(seed=seed)
        a = rng.choice([0.0, 1.0, 2.0, xp.nan], size=size)
        return a

    a = gen_data((1_000, 100_000))  # Python 3.6+ allows _ in numbers
    print(type(a))
    print(a[:3, :3])  # print 1st 3 rows & cols

    a = gen_data((1_000, 100_000), xp="cupy")
    print(type(a))
    print(a[:3, :3])  # print 1st 3 rows & cols

    # 'patch' is a nice built-in Python function that can temporarily
    # add an item to a dictionary, including os.environ.
    from unittest.mock import patch

    with patch.dict("os.environ", {"ARRAY_MODULE": "cupy"}) as _:
        a = gen_data((5, 5))
        print(type(a))

    def unit_standardize(a):
        """
        Standardize array to zero-mean and unit standard deviation.
        """
        xp = get_array_module(a)

        assert a.dtype in [
            np.float64,
            np.float32,
        ], "a must be a float in order to standardize in place."

        imissX = xp.isnan(a)
        snp_std = xp.nanstd(a, axis=0)
        snp_mean = xp.nanmean(a, axis=0)
        # avoid div by 0 when standardizing
        snp_std[snp_std == 0.0] = xp.inf

        a -= snp_mean
        a /= snp_std
        a[imissX] = 0

    a = gen_data((1_000, 100_000))
    unit_standardize(a)
    print(type(a))
    print(a[:3, :3])  # 1st 3 rows and cols

    a = gen_data((1_000, 100_000), xp="cupy")
    unit_standardize(a)
    print(type(a))
    print(a[:3, :3])  # 1st 3 rows and cols

    a = gen_data((1_000, 100_000))
    print(type(a))  # numpy
    xp = array_module(xp="cupy")
    a = xp.asarray(a)
    print(type(a))  # cupy
    unit_standardize(a)
    print(type(a))  # still, cupy
    a = asnumpy(a)
    print(type(a))  # numpy
    print(a[:3, :3])  # print 1st 3 rows and cols
