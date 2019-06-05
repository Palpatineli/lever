from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

def score(params: np.ndarray, paths: np.ndarray, response: np.ndarray) -> Tuple[float, np.ndarray]:
    """target function for a 1d gaussian neuron rf
    Args:
        params: [μ, ρ, b] for one neuron, b = baseline as in exp(b) = amplitude/coefficient
        paths: trajectory, 1D array for time series
        response: neuron activity at each path point, 1-d
    """
    μ, ρ, baseline = params
    lnλ = baseline - 0.5 * (((paths - μ) / ρ) ** 2)
    λ = np.exp(lnλ)
    score = (response * lnλ - λ).mean()
    db = response - λ
    dμ = (paths - μ) / ρ ** 2 * db  # db stading in for response - λ
    dρ = db * (paths - μ) / ρ
    return -score, np.array([-dμ.mean(), -dρ.mean(), -db.mean()])

def estimate_neurons(paths, responses, x0=(0.0, 1.0, 0),
                     bounds=((-1.5, 1.5), (-2, 2), (-10.0, 10.0))) -> np.ndarray:
    """
    Args:
        paths: 1D array for the lever trajectory, need to be resampled to be same as responses
        responses: response from one neuron per row
    """
    bounds = (bounds[0], np.exp(np.subtract(bounds[1], 1)), bounds[2])
    results = [minimize(score, x0=x0, args=(paths, responses[idx, :]), jac=True,
                        method="l-bfgs-b", options={'disp': False, 'ftol': 1E-6}, bounds=bounds).x
               for idx in tqdm(range(responses.shape[0]))]
    return np.array(results)
