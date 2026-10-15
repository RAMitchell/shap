"""Load the Rust module directly to test it without importing other SHAP extensions.

Usage: python3.14t scripts/check_rust_cutils_free_threading.py /path/to/_rust_cutils.so
"""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

assert not sys._is_gil_enabled(), "Run this check with free-threaded Python"
spec = importlib.util.spec_from_file_location("_rust_cutils", sys.argv[1])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not sys._is_gil_enabled(), "Importing the Rust module enabled the GIL"

weights = np.array([0.25, 0.75])
y = np.arange(12, dtype=np.float64).reshape(4, 3)
expected = (y.reshape(2, 2, 3) * weights[None, :, None]).sum(axis=1)


def evaluate(_):
    output = np.zeros((2, 3))
    assert module.compute_exp_val(0, 2, 3, 2, weights, y, output) == 2
    np.testing.assert_array_equal(output, expected)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(evaluate, range(100)))
assert not sys._is_gil_enabled()
print("Rust extension: 100 concurrent calls passed with the GIL disabled")
