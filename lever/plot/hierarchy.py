import numpy as np
from scipy.cluster.hierarchy import dendrogram
from mplplot import Figure
from mplplot.interact import AxhlineMover
import matplotlib.pyplot as plt

def fancy_dendrogram(*args, **kwargs):
    """Plot a cluster matrix with threshold and annotation.
    Pass arguments to scipy.cluster.hierarchy.dendrogram
    Args:
        args[0]: the linkage matrix
        args[1:]: pass to dendrogram
    """
    annotate_above = kwargs.pop('annotate_above', 0)
    annotate_above = annotate_above if annotate_above else 0
    linked = args[0]
    threshold = kwargs.pop("color_threshold", kwargs.pop("threshold", kwargs.pop("max_d", None)))
    if threshold is not None:
        kwargs["color_threshold"] = threshold
    ddata = dendrogram(linked, *args[1:], **kwargs)
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        _, ax = plt.subplots()
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
    return ddata

class DendrogramUpdater(object):
    def __init__(self, data, ax):
        self.data = data
        self.y = -1
        self.ax = ax

    def __call__(self, y_data):
        self.y = y_data
        self.ax.clear()
        dendrogram(self.data, color_threshold=y_data, ax=self.ax)
        return self.ax.axhline(y_data)

def get_threshold(data: np.ndarray, color_threshold: float = None) -> float:
    if color_threshold is None:
        ddata = dendrogram(data, no_plot=True)
        color_threshold = np.median([np.mean(x[1:]) for x in ddata['dcoord']])
    with Figure() as axes:
        ax = axes[0]
        dendrogram(data, color_threshold=color_threshold, ax=ax)
        line = ax.axhline(color_threshold)
        updater = DendrogramUpdater(data, ax)
        mover = AxhlineMover(line, updater)
    while mover.on:
        plt.pause(1)
    del mover
    return updater.y
