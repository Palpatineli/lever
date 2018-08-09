"""reusable plotting methods for lever push analysis"""
from typing import Callable
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd


COLOR_TABLE = (('WT', '#B90803'),  # dark red
               ('Het', '#0C3D79'))  # dark blue


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

def fancy_dendrogram(*args, **kwargs) -> np.ndarray:
    """Plot a cluster matrix with threshold and annotation.
    Pass arguments to scipy.cluster.hierarchy.dendrogram"""
    annotate_above = kwargs.pop('annotate_above', 0)
    linked = linkage(args[0])
    threshold = kwargs.pop("color_threshold", kwargs.pop("threshold", kwargs.pop("max_d", None)))
    if threshold is not None:
        kwargs["color_threshold"] = threshold
    ddata = dendrogram(linked, *args[1:], **kwargs)
    if 'ax' in kwargs:
        ax = kwargs['ax']
        ax.set_title('Hierarchical Clustering Dendrogram (truncated)')
        ax.set_xlabel('sample index or (cluster size)')
        ax.set_ylabel('distance')
        for i, dendro, color in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = dendro[1]
            if y > annotate_above:
                ax.plot(x, y, 'o', c=color)
                ax.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points',
                            va='top', ha='center')
        ddata['ax'] = ax
    return linked
