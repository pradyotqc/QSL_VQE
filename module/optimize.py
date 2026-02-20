from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from scipy.optimize import minimize


OPTIMIZER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "COBYLA": {"maxiter": 150, "rhobeg": 0.5, "tol": 1e-8},
    "Nelder-Mead": {"maxiter": 200, "xatol": 1e-8, "fatol": 1e-8},
    "L-BFGS-B": {"maxiter": 200, "ftol": 1e-12},
}


def run_minimize(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    method: str,
    callback: Optional[Callable[[np.ndarray], None]] = None,
    options: Optional[Dict[str, float]] = None,
):
    default_opts = OPTIMIZER_DEFAULTS.get(method, {})
    opts = {**default_opts, **(options or {})}
    return minimize(objective, x0, method=method, callback=callback, options=opts)
