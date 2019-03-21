"""Determine the cluster number from a sample distribution"""
from typing import Callable, Tuple, List, Generator, Sequence, TypeVar
import numpy as np

T = TypeVar("T")

def combinations(x: Sequence[T]) -> Generator[Tuple[T, T], None, None]:
    for idx, item_a in enumerate(x):
        for item_b in x[idx + 1:]:
            yield item_a, item_b

def combinations2(x: Sequence[T], y: Sequence[T]) -> Generator[Tuple[T, T], None, None]:
    for item_a in x:
        for item_b in y:
            yield item_a, item_b

def l2(x, y):
    return np.sqrt(np.sum(x - y) ** 2)

def _get_range(data: np.ndarray) -> np.ndarray:
    return np.amin(data, 0), np.amax(data, 0)

def _unique(x: np.ndarray) -> List[np.ndarray]:
    numbers, indices = np.unique(x, return_inverse=True)
    return [np.flatnonzero(indices == idx) for idx in range(len(numbers))]

def _get_null_dist(data: np.ndarray, sample_no: int) -> np.ndarray:
    u, d, vh = np.linalg.svd(data, full_matrices=True)
    minimum, maximum = _get_range(np.dot(data, vh.T))
    random_sample = np.random.random_sample((sample_no, data.shape[1])) * (maximum - minimum) + minimum
    return np.dot(random_sample, vh)

def get_w_k(points: np.ndarray, clusters: np.ndarray,  # clusters as the indices from numpy unique return_inverse
            dst: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    w_k = 0.0
    for cluster in _unique(clusters):
        d_r = 0.0
        for idx_x, idx_y in combinations(cluster):
            d_r += dst(points[idx_x, :], points[idx_y, :])
        w_k += d_r / (2 * len(cluster))
    return w_k

def gap_stat(data: np.ndarray, cluster_func: Callable[[np.ndarray, int], np.ndarray],
             b: int, ks: Tuple[int, int],
             dst: Callable[[np.ndarray, np.ndarray], float] = l2) -> np.ndarray:
    """Gap statistic from Tibshirani2001, first above zero shows the optimal cluster number.
    Can detect samples with null/one cluster
    Args:
        data: 2d matrix with observation x features
        cluster_func: clustering function that takes the data and cluster number,
            returns a 1d label array
        b: number of sample sets for the null distribution
        ks: from cluster number ks[0] to ks[1]
        dst: distance metric for clustering result
    Returns:
        1d array of gap score for each k number
    """
    samples = np.split(_get_null_dist(data, data.shape[0] * b), b, 0)
    gap = list()
    s = list()
    for k in range(ks[0], ks[1] + 1):
        logwk = np.log(get_w_k(data, cluster_func(data, k)[1], l2))
        labels = [cluster_func(sample, k)[1] for sample in samples]
        logwke = np.log([get_w_k(sample, label, l2)
                        for sample, label in zip(samples, labels)])
        s.append(np.std(logwke) * np.sqrt(1 + 1 / b))
        gap.append(np.mean(logwke) - logwk)
    return -np.diff(gap) + s[1:]

