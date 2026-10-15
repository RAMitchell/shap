use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute weighted background averages, updating only newly evaluated samples.
#[pyfunction]
#[allow(non_snake_case)]
fn compute_exp_val(
    nsamples_run: usize,
    nsamples_added: usize,
    D: usize,
    N: usize,
    weights: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    ey: &Bound<'_, PyArray2<f64>>,
) -> PyResult<usize> {
    let weights = weights.as_array();
    let y = y.as_array();
    let mut output = ey
        .try_readwrite()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut ey = output.as_array_mut();
    let required_rows = nsamples_added
        .checked_mul(N)
        .ok_or_else(|| PyValueError::new_err("nsamples_added * N overflows"))?;
    if weights.len() < N
        || y.nrows() < required_rows
        || y.ncols() < D
        || ey.nrows() < nsamples_added
        || ey.ncols() > D
    {
        return Err(PyValueError::new_err(
            "incompatible weights, y, or ey shape",
        ));
    }

    let mut sums = vec![0.0; D];
    for i in nsamples_run..nsamples_added {
        sums.fill(0.0);
        for j in 0..N {
            for k in 0..D {
                sums[k] += y[[i * N + j, k]] * weights[j];
            }
        }
        for k in 0..ey.ncols() {
            ey[[i, k]] = sums[k];
        }
    }
    Ok(nsamples_run.max(nsamples_added))
}

#[pymodule(gil_used = false)]
fn _rust_cutils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_exp_val, m)?)
}
