import numpy as np
import numpy.typing as npt

def compute_exp_val(
    nsamples_run: int,
    nsamples_added: int,
    D: int,
    N: int,
    weights: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    ey: npt.NDArray[np.float64],
) -> int: ...
