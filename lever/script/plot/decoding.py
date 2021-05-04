##
from itertools import combinations
from typing import Dict, List, Tuple
from pickle import dump, load
from warnings import simplefilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon, skew
import pandas as pd
from algorithm.stats import scale_features
from algorithm.optimize.piece_lm import PieceLinear2, PieceLinear3
from pypedream import Task, get_result
from lever.script.steps.log import res_trial_log
from lever.script.steps.decoder import res_predict, res_align_xy, res_neuron_info, res_trial_xy
from lever.script.steps.decoder import res_predict_trial
from lever.script.steps.decoder import mice, proj_folder, grouping
from lever.script.utils import print_stats_factory
from mplplot import Figure, plots
from mplplot.accessory import add_scalebar

simplefilter(action='ignore', category=FutureWarning)
colors = ["#00272B", "#099963", "#f94040", "#1481BA", "#E0FF4F"]
Task.save_folder = proj_folder.joinpath("data", "interim")
fig_folder = proj_folder.joinpath("report", "fig", "decoder")
analysis_folder = proj_folder.joinpath("data", "analysis")
colors = ["#00272B", "#099963", "#f94040", "#1481BA", "#E0FF4F"]
print_stats = print_stats_factory(proj_folder.joinpath("report", "stat", "decoding.txt"))

##
def pop_decoder_power():
    def temp(data, ax, name):
        res = ["median: "] + str(data.groupby('group').median()).split('\n')[2:]
        group_names = ('wt', 'glt1', 'dredd', "gcamp6f")
        group_strs = ["WT", "GLT1", "Gq", "gcamp6f"]
        annotation = list()
        for (idx, x), (idy, y) in combinations(enumerate(group_names), 2):
            p = mannwhitneyu(data.loc[x, ], data.loc[y, ], False, 'two-sided').pvalue
            res.append(f"mann u, {x} v. {y} p={p}")
            if p < 0.05:
                annotation.append(((idx, idy), p))
        print_stats(name, res)
        values = [data.loc[x, "mutual_info"] for x in group_names]
        boxplots = plots.boxplot(ax, values, whis=(10., 90.), zorder=1, showfliers=False, colors=colors)
        plots.dots(ax, values, zorder=3, s=24, jitter=0.02)
        ax.set_xticklabels(group_strs)
        plots.annotate_boxplot(ax, boxplots, 24, 1.2, annotation)
    with Figure(fig_folder.joinpath("decoder-comp.svg"), figsize=(8, 9), grid=(1, 2)) as axes:
        axes[0].get_shared_y_axes().join(*axes)
        data: pd.DataFrame = pd.read_csv(analysis_folder.joinpath("decoder_power.csv"), index_col=[0])  # type: ignore
        data = data.set_index(["group", "session"], append=True).reorder_levels(["group", "id", "session"]).sort_index()  # type: ignore
        data = data.drop([('gcamp6f', 51551, 4)], axis='index')
        temp(data, axes[0], "pop-power by fov")
        temp(data.groupby(["group", "id"]).mean(), axes[1], "pop-power by case")

    data.loc['gcamp6f']

def pop_decoder_power_trial():
    def temp(data, ax, name):
        res = ["median: "] + str(data.groupby('group').median()).split('\n')[2:]
        group_names = ('wt', 'glt1', 'dredd')
        group_strs = ["WT", "GLT1", "Gq"]
        annotation = list()
        for (idx, x), (idy, y) in combinations(enumerate(group_names), 2):
            p = mannwhitneyu(data.loc[x, ], data.loc[y, ], True, 'two-sided').pvalue
            res.append(f"mann u, {x} v. {y} p={p}")
            if p < 0.05:
                annotation.append(((idx, idy), p))
        print_stats(name, res)
        values = [data.loc[x, "mutual_info"] for x in group_names]
        boxplots = plots.boxplot(ax, values, whis=(10., 90.), zorder=1, showfliers=False, colors=colors)
        plots.dots(ax, values, zorder=3, s=24, jitter=0.02)
        ax.set_xticklabels(group_strs)
        plots.annotate_boxplot(ax, boxplots, 24, 1.2, annotation)
    with Figure(fig_folder.joinpath("decoder-comp-trial.svg"), figsize=(8, 9), grid=(1, 2)) as axes:
        axes[0].get_shared_y_axes().join(*axes)
        data: pd.DataFrame = pd.read_csv(analysis_folder.joinpath("decoder_power_trial.csv"), index_col=[0])  # type: ignore
        data = data.set_index(["group", "session"], append=True).reorder_levels(["group", "id", "session"]).sort_index()  # type: ignore
        temp(data, axes[0], "pop-power by fov")
        temp(data.groupby(["group", "id"]).mean(), axes[1], "pop-power by case")

def plot_cno_saline():
    data: pd.DataFrame = pd.read_csv(analysis_folder.joinpath("decoder_cno.csv"), usecols=[1, 2, 3], index_col=[2, 1]).sort_index()  # type: ignore
    group_strs = ('saline', 'cno')
    paired_data = [data.loc[treat, 'mutual_info'] for treat in group_strs]
    p_value = wilcoxon(*paired_data).pvalue
    res = ["median: "] + str(data.groupby("treat").median()).split('\n')[2:]
    print_stats("saline vs. cno in Gq", res + [f"paired wilcox: p={p_value}"])
    with Figure(fig_folder.joinpath("decoder-pair-cno.svg"), figsize=(6, 9)) as axes:
        boxplots = plots.boxplot(axes[0], paired_data, whis=(10., 90.), zorder=1, showfliers=False,
                                 colors=colors[2: 4], widths=0.65)
        plots.dots(axes[0], paired_data, zorder=3, s=24)
        axes[0].set_xticklabels(["Gq Saline", "Gq CNO"])
        plots.annotate_boxplot(axes[0], boxplots, 24, 1.2, [((0, 1), p_value)])
        [axes[0].plot([1, 2], x, color='gray') for x in np.array(paired_data).T]

def single_dist():
    data: pd.DataFrame = pd.read_csv(analysis_folder.joinpath("single_power.csv"), usecols=[1, 2, 3, 4],
                                     index_col=[2, 1, 3]).sort_index()  # type: ignore
    bins = np.linspace(data['mi'].min(), data['mi'].max(), 50)  # type: ignore
    group_strs = ("wt", "glt1", "dredd")

    def scale_hist(x):
        res = np.histogram(x, bins=bins)[0]
        return res / res.sum()
    with Figure(fig_folder.joinpath("single-dist.svg"), figsize=(9, 9)) as axes:
        lines = [data.loc[x, ].groupby("case_id").apply(scale_hist).mean()[1:] for x in group_strs]
        boxes = list()
        bin_size = bins[1] - bins[0]
        for idx, (color, line) in enumerate(zip(colors, lines)):
            box = axes[0].bar((np.arange(len(line)) + idx / len(lines)) * bin_size, line,
                              facecolor=color, edgecolor=color, alpha=0.7, align='edge', width=bin_size / len(lines))
            boxes.append(box)
        axes[0].legend(boxes, ["WT", "GLT1", "Gq"])
        axes[0].set_xlabel("Mutual Information (bit/sample)")
        axes[0].set_ylabel("Mean Density")

def single_dist_moments():
    data = pd.read_csv(analysis_folder.joinpath("single_power.csv"), usecols=[1, 2, 3, 4],
                       index_col=[2, 1, 3]).sort_index()
    groups = ("wt", "glt1", "dredd")
    group_strs = ("WT", "GLT1", "Gq")

    def temp(data, ax, name, jitter=0.1):
        res = ["median: "]
        res.extend(str(data.groupby("group").median()).split("\n")[1: -1])
        annotation = list()
        for (idx, x), (idy, y) in combinations(enumerate(groups), 2):
            p = mannwhitneyu(data.loc[x, ], data.loc[y, ]).pvalue
            if p < 0.05:
                res.append(f"mann u {x} v. {y}, p={p}")
                annotation.append(((idx, idy), p))
        print_stats(name, res)
        values = [data.loc[x, ].tolist() for x in groups]
        boxplots = plots.boxplot(ax, values, zorder=1, whis=(10., 90.), showfliers=False, colors=colors)
        plots.dots(ax, values, zorder=3, s=24, jitter=jitter)
        plots.annotate_boxplot(ax, boxplots, 24, 1.2, annotation)
        ax.set_xticklabels(group_strs)
        ax.set_ylabel(name)
    with Figure(fig_folder.joinpath("single-dist-moments.svg"), figsize=(5, 9), grid=(2, 1)) as axes:
        medians = data.groupby("group").apply(lambda x: x.groupby("case_id").median()["mi"])
        temp(medians, axes[0], "Median", 0.001)
        axes[0].set_ylim(-0.001, 0.036)
        skewness = data.groupby("group").apply(lambda x: x.groupby("case_id").apply(lambda y: skew(y)[0]))
        temp(skewness, axes[1], "Skewness", 0.1)

def example_curve():
    xy, predicts = get_result([x.name for x in mice[1: 2]], [res_align_xy, res_predict], "astrocyte")
    with Figure() as ax:
        ax = ax[0]
        ax.plot(xy[0][1].values, color='blue')
        ax.plot(predicts[0], color='orange')

def save_example():
    xy, predicts = get_result([x.name for x in mice[1: 2]], [res_align_xy, res_predict], "astrocyte")
    dataframe = list()
    for (idx, trace), predict in zip(enumerate(xy[0][1].values), predicts[0]):
        dataframe.append((idx / 5.0, trace, predict))
    df = pd.DataFrame(dataframe, columns=["time", "trajectory", "predicted"])
    df.to_csv(proj_folder.joinpath("data", "analysis", "decoder_example.csv"))
    with Figure() as ax:
        ax = ax[0]
        ax.plot(df["time"], df["trajectory"], color="blue")
        ax.plot(df["time"], df["predicted"], color="orange")

def clip(x, r):
    return x[(x > r[0]) * (x < r[1])]

def composite_example():
    mice_df = pd.DataFrame([[x.id, x.fov, x.session, x.name] for x in mice], columns=["id", "fov", "session", "name"])
    mice_df = mice_df.set_index(["id", "fov", "session"]).sort_index()
    choices = [mice_df.loc[x[0], :, x[1]].name.values[0] for x in (('19286', 1), ('14029', 8), ('27215', 13))]
    xy, predicts, log = get_result(choices, [res_align_xy, res_predict, res_trial_log], "astrocyte")
    example_select = {'wt': ((-6, 6), ((0, 1500), (510, 560))),
                      'glt1': ((-3, 6), ((0, 1500), (80, 130))),
                      'dredd': ((-2, 7), ((0, 1500), (1000, 1050)))}
    fig = plt.figure(figsize=(9, 12))
    gs = fig.add_gridspec(3, 3)
    color_indices = [3, 1, 2]
    for idx, (key, value) in enumerate(example_select.items()):
        ax = fig.add_subplot(gs[idx, 0])
        ax.plot(scale_features(xy[idx][1].values), color=colors[0], alpha=0.7, linewidth=1)
        ax.plot(scale_features(predicts[idx]), color=colors[color_indices[idx]], alpha=0.7, linewidth=1)
        ax.set_ylim(*value[0])
        ax.set_xlim(*value[1][0])
        ax.axes.get_yaxis().set_visible(False)
        ax2 = fig.add_subplot(gs[idx, 1: 3])
        ax2.plot(scale_features(xy[idx][1].values), color=colors[0], alpha=0.7, linewidth=2)
        ax2.plot(scale_features(predicts[idx]), color=colors[color_indices[idx]], alpha=0.7, linewidth=2)
        trial_start = clip(log[idx].timestamps / log[idx].sample_rate * 5, value[1][1])
        motion_onset = clip(log[idx].trial_anchors / log[idx].sample_rate * 5, value[1][1])
        ax2.plot(motion_onset, np.full(motion_onset.shape, 1), marker='v', color=colors[0], linestyle='', markersize=16)
        ax2.plot(trial_start, np.full(trial_start.shape, 1), marker='v', color=colors[0], linestyle='', markersize=16,
                 fillstyle='none')
        ax2.set_ylim(*value[0])
        ax2.set_xlim(*value[1][1])
        ax2.axes.get_yaxis().set_visible(False)
    add_scalebar(ax, (900, 4, 1400, 4.4), 1)
    add_scalebar(ax2, (1030, 4, 1040, 4.4), 1)
    plt.tight_layout(0)
    plt.savefig(fig_folder.joinpath("decoder-example.svg"))
    plt.show()

def composite_trial():
    mice_df = pd.DataFrame([[x.id, x.fov, x.session, x.name] for x in mice], columns=["id", "fov", "session", "name"])
    mice_df = mice_df.set_index(["id", "fov", "session"]).sort_index()
    choices = [mice_df.loc[x[0], :, x[1]].name.values[0] for x in (('19286', 1), ('14029', 8), ('27215', 13))]
    xy, predicts = get_result(choices, [res_trial_xy, res_predict_trial], "astrocyte")
    example_select = {'wt': ((-6, 6), ((0, 1500), (680, 730))),
                      'glt1': ((-3, 6), ((0, 1500), (120, 170))),
                      'dredd': ((-2, 7), ((0, 1500), (330, 380)))}
    fig = plt.figure(figsize=(9, 12))
    gs = fig.add_gridspec(3, 3)
    color_indices = [3, 1, 2]
    for idx, (key, value) in enumerate(example_select.items()):
        ax = fig.add_subplot(gs[idx, 0])
        ax.plot(scale_features(xy[idx][1].values), color=colors[0], alpha=0.7, linewidth=1)
        ax.plot(scale_features(predicts[idx]), color=colors[color_indices[idx]], alpha=0.7, linewidth=1)
        ax.set_ylim(*value[0])
        ax.set_xlim(*value[1][0])
        ax.axes.get_yaxis().set_visible(False)
        ax2 = fig.add_subplot(gs[idx, 1: 3])
        ax2.plot(scale_features(xy[idx][1].values), color=colors[0], alpha=0.7, linewidth=2)
        ax2.plot(scale_features(predicts[idx]), color=colors[color_indices[idx]], alpha=0.7, linewidth=2)
        ax2.set_ylim(*value[0])
        ax2.set_xlim(*value[1][1])
        ax2.axes.get_yaxis().set_visible(False)
    add_scalebar(ax, (500, 5, 1000, 5.4), 1)
    add_scalebar(ax2, (360, 5, 370, 5.4), 1)
    plt.tight_layout(0)
    plt.savefig(fig_folder.joinpath("decoder-example-trial.svg"))
    plt.show()
    
def example_validation(xy, predicts):
    xy, predicts = get_result([x.name for x in mice[1: 2]], [res_align_xy, res_predict], "astrocyte")
    with Figure(fig_folder.joinpath("decoder_validation.svg"), (9, 6)) as ax:
        ax = ax[0]
        ax.plot(np.arange(1800), xy[0][1].values[0: 1800], color='blue')
        ax.plot(np.arange(1800, 2100), xy[0][1].values[1800: 2100] - 5.0, color='blue')
        ax.plot(np.arange(2100, 3000), xy[0][1].values[2100: 3000], color='blue')
        ax.plot(np.arange(1800, 2100), predicts[0][1800: 2100] - 10.0, color='orange')
        for idx, neuron in enumerate(xy[0][0].values[:20, :]):
            scaled = neuron * 2 / neuron.max()
            ax.plot(np.arange(1800), scaled[:1800] + 5.0 + idx, color='red')
            ax.plot(np.arange(1800, 2100), scaled[1800: 2100] + 10.0 + idx, color='red')
            ax.plot(np.arange(2100, 3000), scaled[2100: 3000] + 5.0 + idx, color='red')

def single_scatter():
    powers = get_result(mice.name.to_list(), [res_neuron_info])[0]
    x0 = np.linspace(0, 40, 240, endpoint=False)
    labeled_powers = pd.DataFrame(powers, index=mice.index).join(grouping)
    colors = {"wt": "#619CFFFF", "glt1": "#00BA38FF", "dredd": "#F8766DFF"}

    models: Dict[str, List[Tuple[PieceLinear3, np.ndarray]]] = {"wt": list(), "dredd": list(), "glt1": list()}
    for power in labeled_powers:
        grp_str = power[-1]
        power = np.array(power[0:-2])
        transformed = np.log(np.flip(np.sort(power[np.greater(power, 0)])))
        x = np.arange(transformed.shape[0])
        model = PieceLinear2.fit(x, transformed, [[10, -8, -np.inf, -np.inf], [np.inf, np.inf, np.inf, 0]])
        # model = PieceLinear3.fit(x, transformed, [[0, -np.inf, 10, -np.inf, -np.inf, -np.inf],
        #                                           [5, np.inf, 40, 0, 0, 0]])
        models[grp_str].append((model, transformed))
    with Figure(proj_folder.joinpath("report", "fig", "decoder-single-power.svg"), (6, 8)) as axes:
        ax = axes[0]
        for grp_str, values in models.items():
            color = colors[grp_str]
            for model, power in values:
                x = np.arange(len(power))
                ax.scatter(x, power, alpha=0.5, s=4, color=color)
                ax.plot(x0, model.predict(x0), linewidth=0.75, alpha=0.5, color=color)
                ax.scatter(model.x0, model.y0, marker='x', color=color)
                # ax.scatter(model.x1, model.y0 + (model.k1 * (model.x1 - model.x0)), marker='x', color=color)
        ax.set_xlim(0, 40)
        ax.set_ylim(-8, -2)
        ax.xaxis.set_ticks([0, 9, 19, 29, 39])
        ax.xaxis.set_ticklabels(["1st", "10th", "20th", "30th", "40th"])
        ax.set_xlabel("order of neurons in same field of view")
        ax.set_ylabel("log(mutual information)")
        rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8})

    with open(proj_folder.joinpath("data", "analysis", "single_power_piecelinear_fit.pkl"), 'wb') as fpb:
        dump(models, fpb)

    k0s = {x: np.array([y[0].k0 for y in v]) for x, v in models.items()}
    with Figure(proj_folder.joinpath("report", "fig", "decoder-single-power-comp.svg"), (4, 8)) as axes:
        ax = axes[0]
        sns.boxplot(data=[k0s['dredd'], k0s['glt1'], k0s['wt']], width=0.75, notch=False, whis=1.5, fliersize=0, ax=ax)
        sns.stripplot(data=[k0s['dredd'], k0s['glt1'], k0s['wt']], color='black', ax=ax)
        ax.set_ylabel("Slope In Mutual Information\nvs. Neuron ordering")

def slope_tests():
    with open(proj_folder.joinpath("data", "analysis", "single_power_piecelinear_fit.pkl"), 'rb') as fpb:
        models = load(fpb)
    k0s = {x: np.array([y[0].k0 for y in v]) for x, v in models.items()}
    x0s = {x: np.array([y[0].x0 for y in v]) for x, v in models.items()}
    print("DREADD vs. WT k0: {}".format(mannwhitneyu(k0s['dredd'], k0s['wt'])))
    print("GLT-1 vs. WT k0: {}".format(mannwhitneyu(k0s['glt1'], k0s['wt'])))
    print("DREADD vs. WT x0: {}".format(mannwhitneyu(x0s['dredd'], x0s['wt'])))
    print("GLT-1 vs. WT x0: {}".format(mannwhitneyu(x0s['glt1'], x0s['wt'])))

def permutation_for_sample_size():
    from random import choices
    from scipy.stats import linregress
    with open(proj_folder.joinpath("data", "analysis", "single_power_piecelinear_fit.pkl"), 'rb') as fpb:
        models = load(fpb)
    # permutation
    perm_no = 1000
    result = list()
    for _ in range(perm_no):
        temp_res = list()
        for (model, mi), (_, mi_d) in zip(choices(models['wt'], k=perm_no), choices(models['dredd'], k=perm_no)):
            new_sample = np.flip(np.sort(np.random.choice(mi[0: int(model.x0) + 1], len(mi_d))))
            temp_res.append(linregress(np.arange(len(mi_d)), new_sample)[0])
        result.append(temp_res)
    result = np.array(result)
    np.savetxt(proj_folder.joinpath("data", "analysis", "single_power_slope_dredd_wt_permutation.csv"),
               result, delimiter=',')
    plt.hist(np.median(result, axis=1), 50)
    np.median(np.array([y[0].k0 for y in models['dredd']]))
    plt.hist(np.median(result, axis=1))
##
