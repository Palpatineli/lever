from typing import Dict, List, Tuple
from itertools import combinations
import numpy as np
from scipy.cluster.hierarchy import fcluster, single, linkage
from sklearn.decomposition import PCA
from fastdtw import fastdtw
from noformat import File

from lever.reader import load_mat
from lever.filter import devibrate_trials

def dtw_cluster(motion_traces: List[np.ndarray]) -> np.ndarray:
    """Calculate distance matrix between traces using fastdtw
    Args:
        motion_traces: list of 1d traces
    Returns:
        2d distance matrix
    """
    return [fastdtw(x, y)[0] for x, y in combinations(motion_traces, 2)]

def corr_cluster(motion_traces: List[np.ndarray], beta: int = 2) -> np.ndarray:
    """Calculate distance matrix between traces using pearson correlation."""
    cross_corr = np.corrcoef(motion_traces)[np.triu_indices(len(motion_traces), 1)]
    return ((1 - cross_corr) / (1 + cross_corr)) ** beta

def get_l2(points: np.ndarray) -> np.ndarray:
    """calculate distance for n-d points in ascending order"""
    idx = np.arange(points.shape[0])
    mesh_x, mesh_y = np.meshgrid(idx, idx)
    return np.sqrt(np.sum((points[mesh_x, :] - points[mesh_y, :]) ** 2, 2))[np.triu_indices(points.shape[0], 1)]

trace_cluster = get_l2

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

def get_linkage(record_file: File, params: Dict[str, float]) -> np.ndarray:
    """Load record file and get linkage matrix between the trajectories of trials.
    Args:
        record_file: the record file with at least the lever file
        motion_params: param dict including window size and push threshold
    """
    lever = load_mat(record_file['response'])
    lever.center_on("motion", **params)
    lever.fold_trials()
    mask, lever_trials = devibrate_trials(lever.values[0], params['pre_time'], sample_rate=lever.sample_rate)
    dist_mat = trace_cluster(lever_trials)
    return linkage(dist_mat)

def get_cluster_labels(link_mat: np.ndarray, threshold: int) -> np.ndarray:
    clusters = fcluster(link_mat, threshold, 'distance')
    sort_idx = np.argsort(clusters)
    return np.vstack([sort_idx, clusters[sort_idx]])

def kNNdist(points: np.ndarray, k: int) -> np.ndarray:
    """calculate distance for k nearest-neighbor"""
    dist_mat = get_l2(points)
    return np.sqrt(dist_mat[:, 1: k + 1])

def min_points(points: np.ndarray, min_dist: float) -> int:
    return int(sum([np.searchsorted(x, min_dist ** 2) for x in get_l2(points)]) / points.shape[0])

def pca_bisect(trials: np.ndarray) -> np.ndarray:
    """Arbitrarily assign trials into two groups of equal size based on first principle component"""
    components = PCA(3).fit_transform(trials)
    return (components[:, 0] > np.median(components[:, 0])).astype(np.int)
