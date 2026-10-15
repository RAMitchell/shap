"""Parity and buffer-contract checks for the opt-in Rust utility."""

import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest

rust = pytest.importorskip("shap._rust_cutils")
cpp = pytest.importorskip("shap._cutils")


@pytest.mark.parametrize(
    "start,added,n,d", [(0, 5, 7, 3), (2, 5, 7, 3), (5, 5, 7, 3), (7, 5, 7, 3), (0, 3, 0, 2), (0, 0, 3, 2)]
)
@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed"])
def test_matches_cpp(start, added, n, d, layout):
    rng = np.random.default_rng(0)
    weights = rng.random(n)
    y = rng.normal(size=(added * n, d))
    expected = np.full((added, d), -17.0)
    if layout == "F":
        y = np.asfortranarray(y)
        expected = np.asfortranarray(expected)
    elif layout == "strided":
        y_view = np.zeros((added * n, d * 2))[:, ::2]
        y_view[:] = y
        y = y_view
        expected = np.full((added, d * 2), -17.0)[:, ::2]
    elif layout == "reversed":
        y = y[::-1, ::-1]
        weights = weights[::-1]
        expected = expected[::-1, ::-1]
    actual_base = np.full((added, d * 2), -17.0)
    actual = actual_base[:, ::2]
    if layout == "reversed":
        actual = actual[::-1, ::-1]
    result = cpp.compute_exp_val(start, added, d, n, weights, y, expected)
    y.flags.writeable = False
    weights.flags.writeable = False
    assert rust.compute_exp_val(start, added, d, n, weights, y, actual) == result
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_base[:, 1::2], -17.0)


def test_kernel_explainer_uses_rust():
    from shap.explainers import _kernel

    assert _kernel.compute_exp_val is rust.compute_exp_val


@pytest.mark.parametrize("argument", ["weights", "y", "ey"])
def test_reject_dtype_conversion(argument):
    args = {"weights": np.ones(2), "y": np.ones((4, 3)), "ey": np.zeros((2, 3))}
    args[argument] = args[argument].astype(np.float32)
    with pytest.raises(TypeError):
        rust.compute_exp_val(0, 2, 3, 2, **args)


def test_reject_readonly_output():
    ey = np.zeros((2, 3))
    ey.flags.writeable = False
    with pytest.raises((TypeError, ValueError)):
        rust.compute_exp_val(0, 2, 3, 2, np.ones(2), np.ones((4, 3)), ey)


@pytest.mark.parametrize(
    "argument,shape", [("weights", (1,)), ("y", (3, 3)), ("y", (4, 2)), ("ey", (1, 3)), ("ey", (2, 4))]
)
def test_reject_incompatible_shapes(argument, shape):
    args = {"weights": np.ones(2), "y": np.ones((4, 3)), "ey": np.zeros((2, 3))}
    args[argument] = np.zeros(shape)
    with pytest.raises(ValueError, match="shape"):
        rust.compute_exp_val(0, 2, 3, 2, **args)


def test_reject_aliasing():
    y = np.ones((4, 3))
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        rust.compute_exp_val(0, 2, 3, 2, np.ones(2), y, y[:2])


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="requires free-threaded Python")
def test_free_threaded_execution():
    script = Path(__file__).resolve().parents[2] / "scripts/check_rust_cutils_free_threading.py"
    subprocess.run([sys.executable, str(script), rust.__file__], check=True)
