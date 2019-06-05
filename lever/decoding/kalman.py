## kalman decoder in 1D
from typing import Tuple
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from .gaussian_neuron import estimate_neurons
from .utils import Decoder, Bounds

__all__ = ['decode', 'decoder_factory']

def _get_posterior(x_hat, y, neurons, x_hat_prev, w_hat_inv):
    """score function for update x_hat
    Returns:
        score, jacobian
    """
    centers, rhos, baselines = neurons.T
    log_prob = baselines - 0.5 * (((x_hat - centers) / rhos) ** 2)
    prob = np.exp(log_prob)
    score = (prob - y * log_prob).sum() + 0.5 * (x_hat - x_hat_prev) ** 2 * w_hat_inv
    log_prob_diff = -(x_hat - centers) / rhos ** 2
    jacobian = w_hat_inv * (x_hat - x_hat_prev) + ((prob - y) * log_prob_diff).sum(axis=0)
    return score, jacobian

def _get_posterior_hess(x_hat, y, neurons):
    """get the Hessians for the scoreing function
    Returns:
    """
    centers, rhos, baselines = neurons.T
    prob = np.exp(baselines - 0.5 * (((x_hat - centers) / rhos) ** 2))
    log_prob_diff = -(x_hat - centers) / rhos ** 2
    return ((prob - y) / rhos ** 2).sum() + (log_prob_diff ** 2 * prob).sum()

def decode(y_mat: np.ndarray, neurons: np.ndarray, w_mat: float,
           w_hat: float = 5, x_hat: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        y_mat: 2D float neuron firing
        neurons: 2D float neuron RFs, [neuron_no, params], where params: μ, ρ, b (b=baseline)
        w_mat: random factor between steps
        w_hat: initial w_hat value
        x_hat: initial x_hat value
    Returns:
        trace: recontructed
        w_series: w value at each step
    """
    x_series = list()
    w_series = list()
    for y_step in tqdm(y_mat.T):
        w_hat += w_mat  # update p(x(t) | y(0) ... y(t))
        w_hat_inv = 1 / w_hat
        x_hat = minimize(_get_posterior, x_hat, jac=True, method="l-bfgs-b",
                         options={"disp": False, "ftol": 1E-6},  # "gtol": 1E-4},
                         args=(y_step, neurons, x_hat, w_hat_inv)).x[0]
        w_hat = 1 / (_get_posterior_hess(x_hat, y_step, neurons) + w_hat_inv)
        x_series.append(x_hat)
        w_series.append(w_hat)
    return x_series, w_series

def decoder_factory(bounds: Bounds) -> Decoder:
    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        neurons_hat = estimate_neurons(y_learn, X_learn, bounds=bounds)
        return decode(X_test, neurons_hat, 0.01)[0]
    return _decoder

def predictor_factory(y: np.ndarray) -> Decoder:
    bounds: Bounds = (tuple(np.quantile(y, [0.005, 0.995])), (-2, 1), (-5, 5))  # type: ignore
    w_hat = np.var(y) / 4

    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        neurons_hat = estimate_neurons(y_learn, X_learn, bounds=bounds)
        return decode(X_test, neurons_hat, w_hat, 1000)[0]
    return _decoder
##
