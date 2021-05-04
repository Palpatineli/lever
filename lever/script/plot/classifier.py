##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplplot import Figure, plots
from mplplot.pair import pair, density_plot
from mplplot.util import get_gradient_cmap
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import wilcoxon, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from algorithm.utils import quantize
from pypedream import get_result
from lever.plot import get_threshold
from lever.script.steps.trial_neuron import res_trial_neuron, proj_folder
from lever.script.steps.classifier import res_cluster, res_linkage, mice

fig_folder = proj_folder.joinpath("report", "fig", "classifier")

def scale(x, axis=-1):
    std = x.std(axis=axis, keepdims=True)
    return np.divide(x - x.mean(axis=axis, keepdims=True), std, out=np.zeros_like(x), where=std != 0)

##
def draw_threshold():
    case_idx = [idx for idx, case in enumerate(mice) if case.id == "14032" and case.fov == 1][0]
    case = mice[case_idx: case_idx + 1]
    linkage = get_result([x.name for x in case], [res_linkage])[0][0]
    threshold = get_threshold(linkage)
    with Figure(proj_folder.joinpath("report", "fig", "threshold-sample.svg"), (6, 6)) as axes:
        ax = axes[0]
        dendrogram(linkage, color_threshold=threshold, ax=ax)
        ax.axhline(threshold)
        ax.set_xlabel("Trials")
        ax.set_ylabel("Warp path length (a.u.)")

def draw_pca():
    case_idx = [idx for idx, case in enumerate(mice) if case.id == "14032" and case.fov == 1][0]
    case = mice[case_idx: case_idx + 1]
    spikes, clusters = get_result([x.name for x in case], [res_trial_neuron, res_cluster])
    spike, cluster = spikes[0], clusters[0]
    neuron = scale(np.swapaxes(spike.values, 0, 1).reshape(spike.shape[1], -1), axis=0)
    y = quantize(cluster)
    neuron = PCA(20).fit_transform(neuron)
    svc = SVC()
    svc.fit(neuron, y)

    idx0, idx1 = 0, 5
    x_max, x_min = neuron[:, idx0].max(), neuron[:, idx0].min()
    y_max, y_min = neuron[:, idx1].max(), neuron[:, idx1].min()
    XX, YY = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]  # type: ignore
    samples = np.zeros((10000, 20), dtype=np.float)
    samples[:, idx0] = XX.ravel()
    samples[:, idx1] = YY.ravel()
    Z = svc.decision_function(samples).reshape(XX.shape)
    primary_color, secondary_color = "#F8766D", "#619CFF"
    with Figure(fig_folder.joinpath("classifier-decision.png"), figsize=(12, 12),
                despine={'bottom': True, 'left': True}) as axes:
        ax = axes[0]
        mask = y > 0
        ax.pcolormesh(XX, YY, Z, cmap=get_gradient_cmap(secondary_color, primary_color))
        ax.contour(XX, YY, Z, colors=['k'], linestyles=['-'], levels=[.25])
        ax.scatter(neuron[mask, idx0], neuron[mask, idx1], color=primary_color, s=75, edgecolors='k')
        ax.scatter(neuron[~mask, idx0], neuron[~mask, idx1], color=secondary_color, s=75, edgecolors='k')

def draw_diagnal():
    data = pd.read_csv(proj_folder.joinpath("data", "analysis", "classifier_power_validated.csv"))
    means = data.groupby(["id", "session", "group", "type"]).mean()
    colors = {'wt': '#00272B', 'glt1': '#099963', 'dredd': '#F94040', "gcamp6f": "#619CFF"}
    xs, ys = zip(*[(means.query(f"type=='corr' and group=='{group}'")['precision'].values,
                    means.query(f"type=='mean' and group=='{group}'")['precision'].values)
                   for group in ('wt', 'gcamp6f', 'glt1', 'dredd')])
    # customize Figure generation
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot()
    pair(xs, ys, [colors[x] for x in ('wt', 'gcamp6f', 'glt1', 'dredd')], ax1,
         density_params={"bw": 0.5}, scatter_params={"alpha": 0.7})
    plt.savefig(fig_folder.joinpath("classifier-scatterplot.svg"))
    fig2 = plt.figure(figsize=(4, 9))
    ax2 = fig2.add_subplot(1, 1, 1, sharey=ax1)
    data_mean = data[data.type == "mean"]
    data_mean = data_mean.set_index(["group", "id", "session"])
    values_mean = [data_mean.loc[group_str, 'precision'].values for group_str in ('wt', 'glt1', 'dredd')]
    boxplot = plots.boxplot(ax2, values_mean, whis=(10., 90.), showfliers=False, colors=colors.values())
    plots.annotate_boxplot(ax2, boxplot, 16, 1.0, p_values=[((0, 1), 0.334), ((0, 2), 0.964)])
    ax2.set_xticklabels(["WT", 'gcamp6s', "GLT1", "Gq"])
    ax2.set_ylim(*ax1.get_ylim())
    plt.savefig(fig_folder.joinpath("classifier-edge.svg"))
    plt.show()

def draw_boxplot():
    data = pd.read_csv(proj_folder.joinpath("data", "analysis", "classifier_power_validated.csv"))
    data = data[data.type != "none"]
    means = data.groupby(["id", "session", "group", "type"]).mean().reset_index()
    width = 0.6
    with Figure(fig_folder.joinpath("classifier-compare.svg"), (10, 6)) as axes:
        sns.boxplot(x="group", y="precision", hue="type", data=data, notch=True, width=width, whis=1.0, ax=axes[0])
        for idx, group in enumerate(('wt', 'gcamp6f', 'glt1', 'dredd')):
            temp = pd.pivot_table(means[means.group == group], index=['id', 'session'], columns='type', values='precision')
            for value in np.fliplr(temp.values):
                axes[0].plot([idx - width / 4, idx + width / 4], value, color="#555753")
