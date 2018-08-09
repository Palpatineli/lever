"""analysis types done on lever push and related neuron activity"""
from itertools import combinations
from typing import List, Tuple, TypeVar
import numpy as np
from scipy.cluster.hierarchy import fcluster, single
from scipy.stats import pearsonr
from fastdtw import fastdtw
from algorithm.time_series import SparseRec, fold_by, resample
from algorithm.time_series.utils import bool2index, splice
from algorithm.array.main import DataFrame
from .template import TraceTemplate

def threshold_big_cluster(motion_traces: List[np.ndarray], threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Find largest cluster and trials belonging to it
    Args:
        motion_traces: list of trajectory
        threshold: distance threshold for clusters
    Returns:
        (bool array of (trial belongs to largest cluster), cluster_ids of all trials)
    """
    cluster_ids = fcluster(single(trace_cluster(motion_traces)), threshold, criterion='distance')
    ids, counts = np.unique(cluster_ids, return_counts=True)
    biggest_cluster = ids[np.argmax(counts)]
    return cluster_ids == biggest_cluster, cluster_ids

def trace_cluster(motion_traces: List[np.ndarray]) -> np.ndarray:
    """Calculate distance matrix between traces using fastdtw
    Args:
        motion_traces: list of 1d traces
    Returns:
        2d distance matrix
    """
    trace_count = len(motion_traces)
    dtw_mat = np.zeros((trace_count, trace_count))
    dtw_mat[np.triu_indices(trace_count, 1)] = [fastdtw(x, y)[0] for x, y in combinations(motion_traces, 2)]
    return dtw_mat

def create_push_template(lever_data: SparseRec, alpha: float=0.05, threshold: float=16000) -> TraceTemplate:
    lever_data.set_trials()
    traces = lever_data.trials(lever_data.motion_stamp)
    typical_trials = threshold_big_cluster(traces, threshold)[0]
    expert_pushes = traces[typical_trials]
    return TraceTemplate(expert_pushes, alpha)

def classify_cells(lever_data: SparseRec, neuron_data: DataFrame, neuron_sample_rate: float, threshold: float) -> dict:
    """based on last day performance, classify cells by their correlation with typical pushes"""
    # find the typical pushes on the last day
    lever_data.center_on('motion')
    typical_trials = threshold_big_cluster(lever_data.fold_trials(), threshold)[0]

    # classify cells by their correlation to the typical pushes
    good, neg, bad = correlated_cells(lever_data, neuron_data, neuron_sample_rate, typical_trials)
    unrelated = sorted(list(set(neg) | set(bad)))
    return {'good': list(good), 'unrelated': unrelated}

def correlated_cells(lever_data: SparseRec, neuron_data: DataFrame, neuron_sample_rate: float,
                     typical_trials: np.ndarray, p_level: float=0.001)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """extract lever movement and neuron activity around typical movements and find cells with positive significant
    correlation"""
    lever_data.center_on('motion')
    neuron_timeseries = fold_by(neuron_data, lever_data, neuron_sample_rate, True)
    lever_timeseries = resample(lever_data.fold_trials().values, lever_data.sample_rate, neuron_sample_rate, axis=1)
    min_length = min(lever_timeseries.shape[1], neuron_timeseries.shape[2])
    lever_timeseries = lever_timeseries[:, 0: min_length].ravel()
    neuron_timeseries = neuron_timeseries[:, :, 0: min_length].reshape(neuron_timeseries.shape[0], -1)
    corr = np.vstack([pearsonr(lever_timeseries, x) for x in neuron_timeseries])
    return (neuron_data.axes[0][np.logical_and(corr[1, :] < p_level, corr[0, :] > 0)],
            neuron_data.axes[0][np.logical_and(corr[1, :] < p_level, corr[0, :] < 0)],
            neuron_data.axes[0][corr[1, :] >= p_level])

def quiet_period(lever_data: SparseRec, threshold: float=0.2) -> np.ndarray:
    fluctuation = np.diff(lever_data.values)
    quiet_regions = bool2index(np.logical_and(fluctuation < threshold, fluctuation > -threshold))
    return quiet_regions[np.diff(quiet_regions).ravel() > 256, :]

def noise_autocorrelation(lever_data: SparseRec, neuron_data: DataFrame, neuron_sample_rate: float) -> np.ndarray:
    quiet_regions = np.rint(quiet_period(lever_data) * (neuron_sample_rate / lever_data.sample_rate))
    baseline_activity = splice(neuron_data.values, quiet_regions, axis=1)
    return baseline_activity.corr().as_matrix()

def noise_correlation(lever_data: SparseRec, neuron_data_1: DataFrame, neuron_data_2: DataFrame,
                      neuron_sample_rate: float) -> np.ndarray:
    quiet_regions = np.rint(quiet_period(lever_data) * (neuron_sample_rate / lever_data.sample_rate))
    activity_1 = splice(neuron_data_1, quiet_regions, axis=1)
    activity_2 = splice(neuron_data_2, quiet_regions, axis=1)
    len_1, len_2 = activity_1.shape[0], activity_1.shape[0] + activity_2.shape[0]
    return np.corrcoef(activity_1, activity_2)[0: len_1, len_1: len_2]
