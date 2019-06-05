##
from typing import Tuple, List
from functools import reduce
import numpy as np
from numpy import newaxis
from tqdm import tqdm
from .gaussian_neuron import estimate_neurons
from .utils import Decoder, Bounds

__all__ = ['decode', 'decoder_factory']

def scale(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.max(axis=-1, keepdims=True) - x.min(axis=-1, keepdims=True)) * 2

def _isbad(*data: List[np.ndarray], axis=None):
    if axis is None:
        masks = ((np.isnan(datum) | np.isinf(datum)) for datum in data)
    else:
        masks = ((np.any(np.isnan(datum), axis=axis) | np.any(np.isinf(datum), axis=axis)) for datum in data)
    return reduce(np.logical_or, masks)

def log_likelihood(particles: np.ndarray, y: np.ndarray, neurons: np.ndarray) -> np.ndarray:
    """
    Args:
        partiles: 1-D array of spatial partiles
        y: firing of neurons
        neurons: 2D array with rows of neurons, each neuron has μ, ρ, b (baseline)
    Returns:
        1D array of probability of each partile
    """
    inf_mask = ~(_isbad(*neurons.T))
    μ, ρ, b = neurons[inf_mask, :].T
    diff = ((particles[:, newaxis] - μ[newaxis, inf_mask]) / ρ[newaxis, inf_mask]) ** 2
    log_prob = b[inf_mask] - diff * 0.5  # diff[particle_no, neuron_no]
    return log_prob.dot(y[inf_mask, newaxis]).ravel() - np.exp(log_prob).sum(axis=1)

def decode(y_mat, neurons, w_mat, particle_no: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        y_mat: 2D float neuron firing
        neurons: 2D float neuron RFs, [neuron_no, params], where params: μ, ρ, b (b=baseline)
        w_mat: random factor between steps
        particle_no: number of particles
    Returns:
        trace: recontructed
        w_series: w value at each step
        particle_series: particles at each step
    """
    def normalize(mat):
        mat = np.exp(mat - mat.max())
        return mat / mat.sum()

    particles = np.random.randn(particle_no)
    particle_series = list()
    x_series = list()
    w_series = list()
    w_sqrt = np.sqrt(w_mat)
    for firing in tqdm(y_mat.T):
        weights = normalize(log_likelihood(particles, firing, neurons))
        particle_sample = np.sort(np.random.choice(np.arange(particle_no), particle_no, True, weights))
        particles = particles[particle_sample]
        particle_series.append(particles)
        x_series.append(particles.mean())
        w_series.append(np.var(particles))
        particles += w_sqrt * (np.random.randn(particle_no))
    return np.asarray(x_series), np.asarray(w_series), np.asarray(particle_series)

def decoder_factory(bounds: Bounds) -> Decoder:
    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        neurons_hat = estimate_neurons(y_learn, X_learn, bounds=bounds)
        return decode(X_test, neurons_hat, 0.01, 1000)[0]
    return _decoder

def predictor_factory(y: np.ndarray) -> Decoder:
    bounds: Bounds = (tuple(np.quantile(y, [0.01, 0.99])), (-2, 1), (-5, 5))  # type: ignore
    w_hat = 0.001  # np.var(y) / 4

    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        neurons_hat = estimate_neurons(y_learn, X_learn, bounds=bounds)
        return decode(X_test, neurons_hat, w_hat, 1000)[0]
    return _decoder
##
