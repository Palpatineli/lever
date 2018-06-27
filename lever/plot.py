"""reusable plotting methods for lever push analysis"""
from typing import Callable
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


COLOR_TABLE = (('WT', '#B90803'),  # dark red
               ('Het', '#0C3D79'))  # dark blue


def plot_daily(groups: dict, func_one_case: Callable[[str, int], float]):
    for group_id, color in COLOR_TABLE:
        result = pd.DataFrame(index=np.arange(1, 15))
        case_ids = groups[group_id]
        for case_id, fov_id in case_ids:
            result[case_id] = func_one_case(case_id, fov_id)
        result = result.interpolate()
        for col in result.columns:
            plt.plot(result[col], color)


def plot_day_compiled(groups: dict, func_one_case: Callable[[str, int], pd.DataFrame]):
    maximum, minimum = list(), list()
    for group_id, color in COLOR_TABLE:
        case_ids = groups[group_id]
        result = pd.concat([func_one_case(*x) for x in case_ids], ignore_index=True)
        maximum.append(result['day_id'].max())
        minimum.append(result['day_id'].min())
        sns.pointplot(x='day_id', y='values', data=result.groupby('day_id').mean().reset_index(), color=color)
    plt.xlim((max(minimum) - 1.5, min(maximum) - 0.5))


def update(x, y, comparison=np.greater):
    if x is None or comparison(y, x):
        return y
    else:
        return x


def fancy_dendrogram(*args, **kwargs):
    """plot a cluster matrix with threshold and annotation"""
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    ddata = dendrogram(*args, **kwargs)
    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, dendro, color in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = dendro[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=color)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
