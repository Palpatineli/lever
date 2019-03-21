"""analysis types done on lever push and related neuron activity"""
from typing import Tuple
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.stats import pearsonr
from algorithm.time_series import SparseRec
from algorithm.time_series.utils import bool2index, splice
from algorithm.array.main import DataFrame
from .cluster.main import threshold_big_cluster
from .utils import neuron_lever, MotionParams

def motion_corr(lever_data: SparseRec, neuron_data: DataFrame, neuron_rate: float, threshold: float,
                motion_params: MotionParams) -> np.ndarray:
    """Calculate the inter-neuronal correlation for motion correlated/uncorrelated neurons.
    Args:
        lever_data: recording of lever push, unfolded
        neuron_data: unfolded neuronal data in a DataFrame/Recording/SparseRec
        neuron_rate: sample rate of neuronal data, in case it's in DataFrame format
        threshold: threshold for main cluster, for threshold_big_cluster
    Returns:
        correlation between neuron and lever_data
    """
    neuron, lever = neuron_lever(lever_data, neuron_data, neuron_rate, motion_params)
    typical_trials = threshold_big_cluster(lever, threshold)[0]
    lever_comb = lever[typical_trials].ravel()
    neuron_comb = neuron[:, typical_trials, :].reshape(neuron.shape[0], -1)
    return np.vstack([pearsonr(lever_comb, x) for x in neuron_comb])

def classify_cells(corr: np.ndarray, p_value: float = 0.001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the inter-neuronal correlation for motion correlated/uncorrelated neurons.
    Args:
        corr: [r, p] where each row is one neuron
        p_value: if p < p_level, the cell is considered correlated
    Returns:
        [correlated, anticorrelated, uncorrelated] mask for neuron, assuming same ids as neuron_data
    """
    unrelated = corr[:, 1] >= p_value
    positive = corr[:, 0] > 0
    return ~unrelated & positive, unrelated, ~unrelated & ~positive

def quiet_period(lever_data: SparseRec, threshold: float = 50.) -> np.ndarray:
    """Extract quiet period from lever data. Where sample diff (change in 1 / 256 ) is smaller than
    threshold for longer than 1 sec.
    Args:
        neuron_lever: SparseRec where lever trajectory is [1, sample_no] array
        threshold: change per second (averaged to sample) must be smaller than threshold
    Returns:
        [period_no, 2] start and end of the quiet periods
    """
    fluctuation = np.diff(lever_data.values[0, :])
    threshold /= lever_data.sample_rate
    quiet_regions = bool2index((fluctuation < threshold) & (fluctuation > -threshold))
    return quiet_regions[np.diff(quiet_regions).ravel() > lever_data.sample_rate, :]

def noise_autocorrelation(lever_data: SparseRec, neuron_data: DataFrame, neuron_sample_rate: float) -> np.ndarray:
    quiet_regions = np.rint(quiet_period(lever_data) * (neuron_sample_rate / lever_data.sample_rate)).astype(np.int)
    mask = (quiet_regions[:, 1] <= neuron_data.shape[1])
    baseline_activity = splice(neuron_data.values, quiet_regions[mask, :], axis=1)
    return np.corrcoef(baseline_activity)

def noise_correlation(lever_data: SparseRec, neuron_data_1: DataFrame, neuron_data_2: DataFrame,
                      neuron_sample_rate: float) -> np.ndarray:
    quiet_regions = np.rint(quiet_period(lever_data) * (neuron_sample_rate / lever_data.sample_rate)).astype(np.int)
    mask1, mask2 = (quiet_regions[:, 1] <= neuron_data_1.shape[1]), (quiet_regions[:, 1] <= neuron_data_2.shape[1])
    activity_1 = splice(neuron_data_1.values, quiet_regions[mask1, :], axis=1)
    activity_2 = splice(neuron_data_2.values, quiet_regions[mask2, :], axis=1)
    len_1, len_2 = activity_1.shape[0], activity_1.shape[0] + activity_2.shape[0]
    return np.corrcoef(activity_1, activity_2)[0:len_1, len_1:len_2]

def get_shift(trace0: np.ndarray, trace1: np.ndarray) -> int:
    return np.argmax(np.abs(ifft(fft(trace1) * fft(trace0).conjugate())))

def reliability(traces: np.ndarray) -> float:
    """Calculate the reliability between repeated observations.
    Args:
        traces: 2D array where rows are observations and one row is the time series during one observation.
    Return:
        reliability: :math:`\\frac{2}{T^2 -T} \\sum_{i=1}^T{\\sum_{j=i+1}^T{\\rho(R_i, R_j)}}`
    """
    trial_no = traces.shape[0]
    coef = 2 / (trial_no ** 2 - trial_no)
    return np.triu(np.corrcoef(traces, rowvar=False)).sum() * coef
