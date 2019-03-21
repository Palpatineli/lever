"""reusable plotting methods for lever push analysis"""
from typing import Callable, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
from noformat import File
from algorithm.array import DataFrame
from algorithm import time_series as ts

from lever.reader import load_mat
from lever.utils import filter_empty_trials, iter_groups

COLOR_TABLE = (('WT', '#B90803'), ('Het', '#0C3D79'))  # dark red and blue

def plot_daily(groups: dict, func_one_case: Callable[[str, int], float], ax: Axes) -> Axes:
    for group_id, color in COLOR_TABLE:
        result = pd.DataFrame(index=np.arange(1, 15))
        case_ids = groups[group_id]
        for case_id, fov_id in case_ids:
            result[case_id] = func_one_case(case_id, fov_id)
        result = result.interpolate()
        for col in result.columns:
            ax.plot(result[col], color)
        return ax

def plot_day_compiled(groups: dict, func_one_case: Callable[[str, int], pd.DataFrame], ax: Axes) -> Axes:
    maximum, minimum = list(), list()
    for group_id, color in COLOR_TABLE:
        case_ids = groups[group_id]
        result = pd.concat([func_one_case(*x) for x in case_ids], ignore_index=True)
        maximum.append(result['day_id'].max())
        minimum.append(result['day_id'].min())
        ax = sns.pointplot(x='day_id', y='values', data=result.groupby('day_id').mean().reset_index(),
                           color=color, ax=ax)
    ax.set_xlim((max(minimum) - 1.5, min(maximum) - 0.5))
    return ax

def update(x, y, comparison=np.greater):
    if x is None or comparison(y, x):
        return y
    else:
        return x

def show_correspondance(ax: Axes, record_file: File, motion_params: Dict[str, float]):
    def scale(x):
        x -= x.mean()
        x /= x.std()
        return x
    lever = load_mat(record_file['response'])
    lever.center_on("motion", **motion_params)
    activity = DataFrame.load(record_file["measurement"])
    neuron_rate = record_file.attrs['frame_rate']
    trials = filter_empty_trials(ts.fold_by(activity, lever, neuron_rate, True))
    slow_lever = ts.resample(lever.fold_trials().values, lever.sample_rate, neuron_rate, -1)[:, 0: trials.shape[1], :]
    ax.plot(scale(trials.values[0].reshape(-1)), 'green')
    ax.plot(scale(slow_lever.reshape(-1)), 'red')

try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    Axes3D = None
from sklearn.decomposition import PCA
_colors = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]

def show_cluster(data: np.ndarray, clusters: np.ndarray, colors: List[str] = _colors) -> None:
    """If we have mplot3d, show a 3d plot for the PCA of clustered points.
    Otherwise show a 3d plot
    """
    if Axes3D:
        _show_cluster_3d(data, clusters, colors)
    else:
        _show_cluster_2d(data, clusters, colors)

def _show_cluster_2d(data: np.ndarray, clusters: np.ndarray, colors: List[str] = _colors) -> None:
    points = PCA(2).fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for unique, color in zip(np.unique(clusters), colors):
        cluster = clusters == unique
        ax.scatter(points[cluster, 0], points[cluster, 1], c=color)
    plt.show()
    return points

def _show_cluster_3d(data: np.ndarray, clusters: np.ndarray, colors: List[str] = _colors) -> None:
    points = PCA(3).fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for unique, color in zip(np.unique(clusters), colors):
        cluster = clusters == unique
        ax.scatter(points[cluster, 0], points[cluster, 1], points[cluster, 2], c=color)
    plt.show()
    return points

def jitter(x, y, ax=None, **kwargs):
    return (plt if ax is None else ax).scatter(y, x * np.ones(len(y)) + (np.random.rand(len(y)) - 0.5) * 0.1, **kwargs)

def plot_scatter(data, colors, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    for (idx, name, group), color in zip(iter_groups(data), colors):
        jitter(idx, group, ax, color=color, label=name)
        ax.legend()
