"""Compare the opt-in Rust utility with its retained nanobind implementation.

Run against an installed Rust-enabled wheel: python -P scripts/benchmark_rust_cutils.py
"""

import json
import platform
import statistics
import timeit

import numpy as np

from shap import _cutils, _rust_cutils


def main():
    rng = np.random.default_rng(0)
    results = []
    for label, samples, n, d, start, strided in [
        ("binding_only", 1, 10, 1, 1, False),
        ("small", 1, 10, 1, 0, False),
        ("medium", 128, 100, 4, 0, False),
        ("large", 512, 100, 8, 0, False),
        ("strided", 128, 100, 4, 0, True),
    ]:
        weights = rng.random(n)
        y = rng.normal(size=(samples * n, d * (2 if strided else 1)))
        if strided:
            y = y[:, ::2]
        output_cpp = np.zeros((samples, d))
        output_rust = np.zeros_like(output_cpp)
        args = (start, samples, d, n, weights, y)
        _cutils.compute_exp_val(*args, output_cpp)
        _rust_cutils.compute_exp_val(*args, output_rust)
        np.testing.assert_array_equal(output_cpp, output_rust)
        loops = max(10, min(100_000, 5_000_000 // max(1, (samples - start) * n * d)))
        timings = {}
        for name, function, output in [
            ("cpp", _cutils.compute_exp_val, output_cpp),
            ("rust", _rust_cutils.compute_exp_val, output_rust),
        ]:
            measurements = timeit.repeat(lambda: function(*args, output), repeat=7, number=loops)
            timings[name] = statistics.median(measurements) / loops
        results.append(
            {
                "case": label,
                "samples": samples,
                "background_rows": n,
                "outputs": d,
                "seconds_per_call": timings,
                "rust_over_cpp": timings["rust"] / timings["cpp"],
            }
        )
    print(
        json.dumps({"python": platform.python_version(), "platform": platform.platform(), "cases": results}, indent=2)
    )


if __name__ == "__main__":
    main()
