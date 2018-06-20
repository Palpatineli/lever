"""analysis types done on lever push and related neuron activity"""
from itertools import combinations
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, single
from scipy.stats import pearsonr
from fastdtw import fastdtw
from .reader.mat_reader import LeverData
from .template import TraceTemplate
from .timeseries import peri_motion


TP_FREQ = 10.039


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


def create_push_template(lever_data: LeverData, alpha: float=0.05, threshold: float=16000) -> TraceTemplate:
    traces = lever_data.trials(lever_data.motion_stamp)
    typical_trials = threshold_big_cluster(traces, threshold)[0]
    expert_pushes = traces[typical_trials]
    return TraceTemplate(expert_pushes, alpha)


def classify_cells(lever_data: LeverData, neuron_data: pd.DataFrame, threshold: bool) -> dict:
    """based on last day performance, classify cells by their correlation with typical pushes"""
    # find the typical pushes on the last day
    typical_trials = threshold_big_cluster(lever_data.trials(lever_data.motion_stamp), threshold)[0]

    # classify cells by their correlation to the typical pushes
    good, neg, bad = correlated_cells(lever_data, neuron_data, typical_trials)
    unrelated = sorted(list(set(neg) | set(bad)))
    return {'good': list(good), 'unrelated': unrelated}


def correlated_cells(lever_data: LeverData, neuron_data: pd.DataFrame, typical_trials: np.ndarray,
                     p_level: float=0.001):
    """extract lever movement and neuron activity around typical movements and find cells with positive significant
    correlation"""
    segment = lever_data.resampled(peri_motion(lever_data)[typical_trials, :], TP_FREQ)
    lever_timeseries = nag.splice(lever_data.resampled(lever_data.full_trace, TP_FREQ), segment)
    neuron_timeseries = nag.splice(neuron_data, segment)
    lever_timeseries, neuron_timeseries = nag.trim_to_min(lever_timeseries, neuron_timeseries)
    corr = neuron_timeseries.apply(lambda x: pd.Series(pearsonr(x, lever_timeseries))).values
    return (neuron_data.columns[np.logical_and(corr[1, :] < p_level, corr[0, :] > 0)],
            neuron_data.columns[np.logical_and(corr[1, :] < p_level, corr[0, :] < 0)],
            neuron_data.columns[corr[1, :] >= p_level])


def quiet_period(lever_data: LeverData, threshold: float=0.2) -> np.ndarray:
    fluctuation = np.diff(lever_data.full_trace)
    flat_regions = nag.contiguous_regions(np.logical_and(fluctuation < threshold, fluctuation > -threshold))
    return flat_regions[np.diff(flat_regions).ravel() > 256, :]


def noise_autocorrelation(lever_data: LeverData, neuron_data: pd.DataFrame) -> np.ndarray:
    baseline_activity = nag.splice(neuron_data, lever_data.resample_points(quiet_period(lever_data), TP_FREQ))
    return baseline_activity.corr().as_matrix()


def noise_correlation(lever_data: LeverData, neuron_data_1: pd.DataFrame, neuron_data_2: pd.DataFrame) -> np.ndarray:
    activity_1 = nag.splice(neuron_data_1, lever_data.resample_points(quiet_period(lever_data), TP_FREQ))
    activity_2 = nag.splice(neuron_data_2, lever_data.resample_points(quiet_period(lever_data), TP_FREQ))
    result = np.empty((len(neuron_data_1.columns), len(neuron_data_2.columns)), dtype=float)
    for idx, x in enumerate(neuron_data_1.columns):
        for idy, y in enumerate(neuron_data_2.columns):
            result[idx, idy] = pearsonr(activity_1[x], activity_2[y])[0]
    return result
