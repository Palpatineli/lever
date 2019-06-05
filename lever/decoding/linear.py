from typing import Tuple, Callable
import numpy as np
from .gaussian_neuron import estimate_neurons
from .utils import Decoder, Bounds

def decode(y_mat, neurons) -> np.ndarray:
    """Use the mean of linear combination of gaussian neurons
    Args:
        y_mat: neuron firing, row of neurons and columns of time points
        neurons: 2D matrix, np.c_[μ, σ, baseline], where each is one column
    Returns:
        predicted: 1D predicted trajectory
    """
    return (y_mat * neurons[:, 0: 1] / y_mat.mean(axis=0, keepdims=True)).mean(axis=0)

def decoder_factory(bounds: Tuple[Tuple[float, float], Tuple[int, int], Tuple[int, int]])\
        -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        neuron_hat = estimate_neurons(y_learn, X_learn, bounds=bounds)
        return decode(X_test, neuron_hat)
    return _decoder

def predictor_factory(y: np.ndarray) -> Decoder:
    bounds: Bounds = (tuple(np.quantile(y, [0.005, 0.995])), (-2, 1), (-5, 5))  # type: ignore

    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        neurons_hat = estimate_neurons(y_learn, X_learn, bounds=bounds)
        return decode(X_test, neurons_hat)
    return _decoder
