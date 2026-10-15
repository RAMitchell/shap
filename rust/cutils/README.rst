Rust cutils experiment
======================

This is an opt-in PyO3 + rust-numpy implementation of ``compute_exp_val``,
KernelExplainer's weighted background reduction. The default build continues to
use ``shap._cutils``; an enabled build installs ``shap._rust_cutils`` and routes
KernelExplainer's call to Rust. The C++ function remains available for comparison.
This is independent of the tree-binding nanobind PR.

Build and check
---------------

Install Rust/Cargo (the locked dependencies require Rust >= 1.83), then use the
existing scikit-build-core backend::

    python -m pip install . -Ccmake.define.SHAP_BUILD_RUST_CUTILS=ON
    python -P -m pytest tests/explainers/test_rust_cutils.py tests/explainers/test_kernel.py --import-mode=append
    python -P scripts/benchmark_rust_cutils.py

For uv, scope the option to SHAP so it does not reach dependency build backends::

    uv pip install . --config-settings-package shap:cmake.define.SHAP_BUILD_RUST_CUTILS=ON

The additional dependencies are Cargo crates, not Python runtime packages:
``numpy`` 0.29.0, ``pyo3`` 0.29.3 and the build dependency
``pyo3-build-config`` 0.29.3. Cargo.lock locks 24 external crates, including
ndarray and PyO3's macro/build dependencies. Neither maturin nor setuptools-rust
is needed. Cargo downloads the locked crates on the first source build unless
they are already cached; --locked does not make builds offline.

The binding accesses NumPy buffers without copies, supports arbitrary strides,
checks relevant dimensions and returns Python errors for read-only or aliased
outputs. The numerical loops use safe Rust; there is no application-level unsafe
code. PyO3/rust-numpy necessarily use unsafe code internally at the Python/NumPy
boundary. Rust-numpy's automatic writable-array extraction can panic on a failed
borrow, so the prototype explicitly uses ``try_readwrite`` instead.

Measurements and scope
----------------------

``measurements.json`` records local Linux results, including source line counts,
crate/download/toolchain sizes, wheel sizes, build timings and the microbenchmark.
Those are measurements of this small prototype on one machine, not language-wide
performance claims. Build times exclude initial toolchain and crate downloads.
The extra shared-library cost is for a second extension while retaining C++.

The dedicated CI workflow proposes native Linux, macOS and Windows builds, plus
an isolated free-threaded 3.14 test. Only Linux execution has been verified locally;
this branch does not claim macOS/Windows or hosted CI have passed. Existing
workflows are unchanged because the Rust option defaults to OFF.

Free-threaded Python 3.14 builds use their interpreter-specific extension suffix;
normal CPython builds retain cp312/abi3 wheels. The module was also loaded directly
and exercised concurrently with the GIL disabled, independently of the other SHAP
extensions. A full free-threaded SHAP application is outside this prototype:
existing extensions and dependencies may re-enable the GIL. This PyO3 version
rejects 3.13t. The free-threaded CI check pins a known NumPy wheel and avoids
building unrelated scientific/plotting dependencies from source.

What making Rust mandatory would require
-----------------------------------------

* Install Rust in every job that builds SHAP from source, including test, typing,
  docs/notebooks, data-preparation and GPU build jobs.
* Install it inside cibuildwheel's manylinux/musllinux containers, not just on the
  host runner; add Cargo registry/target caches keyed by toolchain, OS, architecture,
  Python flavor and Cargo.lock. Normal pip/scikit-build cache keys need review too.
* Map target architectures, linkers and PyO3 cross-Python configuration for cross
  builds. The current CMake adapter intentionally rejects cross/Pyodide and macOS
  universal builds. Native platform branches are implemented but unverified off Linux.
* Add wasm32-unknown-emscripten support and validate the Python/NumPy boundary for
  Pyodide, or retain the C++ backend there. Native ARM wheel jobs need validation.
* Keep Rust formatting, clippy, Cargo.lock and dependency/MSRV updates in CI.
* Keep scikit-build-core/CMake for the remaining C++/CUDA code. Replacing it with
  maturin would be a separate and more disruptive packaging experiment.

This function shows that a Rust kernel is compact, while integrating a second
binding/toolchain ecosystem carries most of the one-function migration cost.
