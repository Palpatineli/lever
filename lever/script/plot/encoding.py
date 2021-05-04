##
from itertools import cycle, combinations
import numpy as np
import pandas as pd
from matplotlib import rc
from scipy.stats import mannwhitneyu
from algorithm.array.main import search_ar
from mplplot import Figure, plots
from sklearn.decomposition import PCA
from pypedream import get_result
from lever.script.steps.encoding import res_encoding, res_behavior, res_align_xy, res_predictor
from lever.script.steps.encoding import mice, proj_folder, predictor_names
from lever.script.utils import print_stats_factory

data_folder = proj_folder.joinpath("data", "analysis")
fig_folder = proj_folder.joinpath("report", "fig", "encoder")
colors = ["#00272B", "#099963", "#f94040", "#1481BA", "#E0FF4F"]
print_stats = print_stats_factory(proj_folder.joinpath("report", "stat", "encoding.txt"))
grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "index.csv")).set_index(["id", "session"])  # type: ignore

def r2imshow():
    r2 = get_result(mice.name.to_list(), [res_encoding], "r2")[0]
    r2 = [x[0] for x in r2]
    fig_path = proj_folder.joinpath("report", "fig", "encoding")
    for single_r2, (_, mouse) in zip(r2, mice.iterrows()):
        if mouse.name in ["14039-2-02"]:  # ["14032-1-01", "14039-2-02", "14039-1-01", "27215-3-18"]:
            print(mouse.name)
            with Figure(fig_path.joinpath(mouse.name + ".svg"), (10, 20)) as axes:  # type: ignore
                ax = axes[0]
                rc('font', family='Lato', weight='bold', size=12)
                order = PCA(n_components=2).fit_transform(single_r2)  # type: ignore
                data = plots.Array(np.maximum(single_r2[np.argsort(-order[:, 0]), :-1].T, 0),
                                   [predictor_names, np.arange(single_r2.shape[0])])
                data.axes = [data.axes[1], data.axes[0]]
                data.values = data.values.T
                im, colorbar = plots.heatmap(ax, data, vmin=0, vmax=0.10, colorbar={"orientation": "vertical"})
                plots.annotate_heatmap(im, data.values * 100, valfmt="{x:2.0f}")

##
def single_order_compare():
    """compare different ordering of the encoding r2 heatmap."""
    from sklearn.manifold import TSNE
    r2 = get_result(["14032-1-01"], [res_encoding], "r2")[0]
    single_r2 = r2[0][0]
    with Figure(grid=(1, 3), figsize=(20, 20)) as axes:
        axes[0].imshow(single_r2)
        order = PCA(n_components=2).fit_transform(single_r2)
        axes[1].imshow(single_r2[np.argsort(-order[:, 1]), :])
        order = TSNE(n_components=1).fit_transform(single_r2)
        axes[2].imshow(single_r2[np.argsort(-order.ravel()), :])

##
def spline_examples():
    case_id, trial_id = 0, 3
    behavior, align_xy, predictors = get_result([mice.iloc[case_id]["name"]], [res_behavior, res_align_xy, res_predictor])
    preds, y, preds_group = predictors[0]
    splines = preds.event[0:5, trial_id, :].T
    coefs = np.array([[1, 2, 3, 4, 3],
                      [4, 2, 3, 3, 2],
                      [5, 7, 1, 1, 1]])
    fig_path = proj_folder.joinpath("report", "fig", "encoding")
    with Figure(fig_path.joinpath("splines.svg"), grid=(coefs.shape[0], 1), figsize=(15, 15)) as axes:
        for idx, coef in enumerate(coefs):
            axes[idx].plot(splines * coef.reshape(1, -1))
            axes[idx].plot(splines.dot(coef.reshape(-1, 1)))
    with Figure(fig_path.joinpath("splines_coef.svg"), grid=(coefs.shape[0], 1), figsize=(5, 15)) as axes:
        for idx, coef in enumerate(coefs):
            axes[idx].barh(np.arange(coef.shape[0]), coef)

def predictor_examples():
    behavior, align_xy, predictors = get_result([x.name for x in mice], [res_behavior, res_align_xy, res_predictor])
    fig_folder = proj_folder.joinpath("report", "fig", "encoder")
    case_id, trial_id = 0, 3
    (amplitude, _, delay, hit, _), (spike, trace) = behavior[case_id], align_xy[case_id]
    onsets = np.rint(trace.timestamps * 5 / 256).astype(np.int)
    preds, y, preds_group = predictors[case_id]
    with Figure(fig_folder.joinpath("pred-start.svg"), (16, 2)) as axes:  # start: event
        axes[0].plot(preds.event[0:5, trial_id, :].T)
        axes[0].axvline(0, color="#88498F")
    with Figure(fig_folder.joinpath("pred-reward.svg"), (16, 2)) as axes:  # reward: event
        axes[0].plot(preds.event[5:10, trial_id, :].T)
        axes[0].axvline(np.flatnonzero(preds.event[5:10, trial_id, :][0])[0], color="#88498F")
    with Figure(fig_folder.joinpath("pred-delay.svg"), (16, 2)) as axes:  # delay: period
        delay_trace = np.zeros(25).astype(np.bool_)
        delay_trace[0: int(round(delay[trial_id] * 5))] = True
        axes[0].plot(delay_trace, color="#619CFF")
    with Figure(fig_folder.joinpath("pred-trajectory.svg"), (16, 4)) as axes:  # trajectory: trace
        axes[0].plot(trace.values[onsets[trial_id]: onsets[trial_id] + 25])
        axes[0].plot(trace.values[onsets[trial_id]: onsets[trial_id] + 25] ** 2)
        axes[0].plot(trace.values[onsets[trial_id]: onsets[trial_id] + 25] ** 3)
    with Figure(fig_folder.joinpath("pred-speed.svg"), (16, 2)) as axes:  # speed: trace
        axes[0].plot(np.diff(trace.values[onsets[trial_id] - 1: onsets[trial_id] + 25]))
    print(f"Trial Variables: [Hit: {hit[trial_id]}], [Amplitude: {amplitude[trial_id]:1.3f}], "
          "[Max Speed: {max_speed[trial_id]:1.3f}]," f" [Delay Time: {delay[trial_id]:1.3f}]")
    with Figure(fig_folder.joinpath("y-neurons.svg"), (12, 2), (1, 5)) as axes:  # neuron spikes
        for idx, neuron_id in enumerate(search_ar(np.array((660337, 470451, 290457, 280493, 780456)), spike.axes[0])):
            neuron = spike.values[neuron_id, onsets[trial_id]: onsets[trial_id] + 25]
            axes[idx].plot(neuron)

def complex_example():
    behavior, align_xy, predictors = get_result([x.name for x in mice], [res_behavior, res_align_xy, res_predictor])
    fig_folder = proj_folder.joinpath("report", "fig", "encoder")
    case_id, trial_id = 1, 8
    trace = behavior[case_id], align_xy[case_id][1][1]
    onsets = np.rint(trace.timestamps * 5 / 256).astype(np.int)  # type: ignore
    preds, y, preds_group = predictors[case_id]
    with Figure(fig_folder.joinpath("complex-example.svg")) as axes:
        axes[0].plot(trace.values[onsets[trial_id] - 5: onsets[trial_id] + 25])  # type: ignore
        axes[0].axvline(5)
        axes[0].axvline(np.flatnonzero(preds.event[5:10, trial_id, :][0])[0], color="#88498F")

def glm_icon():
    from statsmodels import api as sm
    from mplplot import Figure
    x = np.linspace(0, 100, 100)
    y = np.exp(x / 50) * 10 + np.random.randn(100) * 5
    model = sm.GLM(y, sm.add_constant(x), family=sm.families.Gaussian(sm.families.links.log())).fit()
    y_hat = model.predict(sm.add_constant(x))
    with Figure(proj_folder.joinpath("report", "fig", "glm_icon.svg"), (6, 6)) as axes:
        ax = axes[0]
        ax.scatter(x, y)
        ax.plot(x, y_hat, color='red')

def boxplot_main():
    data = pd.read_csv(data_folder.joinpath("encoding_minimal.csv"))
    columns = data.columns[1:9]
    data[columns] = data[columns].clip(0)
    data = data.set_index(['group', 'case_id', 'session_id', 'Unnamed: 0']).sort_index()
    data.index.names = data.index.names[:3] + ['id']
    data_wt = data.query("group =='wt'")
    data_glt = data.query("group =='glt1'")
    data_dredd = data.query("group =='dredd'")
    features = ['all', 'hit', 'delay', 'speed']
    columns = [x[y] for y in features for x in (data_wt, data_glt, data_dredd)]
    data_groups = data.groupby("group")
    data_medians = data_groups.median()
    for feature in features:
        res = [f"median: {data_medians[feature]}"]
        for a, b in combinations(data_groups[feature], 2):
            p = mannwhitneyu(a[1], b[1], True, 'two-sided').pvalue
            res.append(f"mann u, {a[0]} v. {b[0]} p = {p}")
        print_stats(f"{feature}: ", res)
    positions = np.arange(4 * len(features)).reshape(-1, 4)[:, 1:].ravel()
    xticks = np.arange(4 * len(features)).reshape(-1, 4)
    xticks[:, :2] += 1
    xticks_labels = ["WT", "GLT1", "\n", "DREADD"] * len(features)
    for idx, feature in enumerate(features):
        xticks_labels[2 + idx * 4] = "\n" + feature
    with Figure(fig_folder.joinpath("boxplot-all.svg"), figsize=(9, 6)) as axes:
        patches = plots.boxplot(axes[0], columns, positions=positions, notch=True, colors=cycle(colors[:3]),
                                whiskerprops={"linewidth": 2}, showfliers=False, whis=(10., 90.))
        axes[0].set_xticks(xticks.ravel())
        axes[0].set_xticklabels(xticks_labels, {"fontsize": 16})
        axes[0].set_ylabel("Proportion of Neural Activity\nExplained ($R^2$)", {"fontsize": 24})
        axes[0].set_ylim([-0.01, 0.3])
        plots.annotate_boxplot(axes[0], patches, 16, 1.0,
                               [((0, 1), 3.01E-15), ((0, 2), 3.55E-11), ((3, 4), 1.17E-17), ((6, 7), 1.38E-22),
                                ((6, 8), 2.01E-3), ((9, 10), 2.62E-17), ((9, 11), 7.01E-6)])
        for start, end in zip(patches["medians"][::3], patches["medians"][2::3]):
            left_line = start.get_data()[0]
            width = left_line[1] - left_line[0]
            right_line = end.get_data()[0]
            x = [left_line[0] - width / 2, right_line[1] + width / 2]
            axes[0].plot(x, [start.get_data()[1][0]] * 2, linestyle=":", color="black", zorder=4, linewidth=2)
            axes[0].plot(x, [start.get_data()[1][0]] * 2, linestyle=":", color="black", zorder=4, linewidth=2)

def boxplot_all():
    data: pd.DataFrame = pd.read_csv(data_folder.joinpath("encoding_minimal.csv")).set_index(["group", "id", "fov", "session"]).sort_index()["all"]
    data = data.drop([('gcamp6f', 51551, 1, 1), ('gcamp6f', 51551, 3, 4), ('gcamp6f', 51551, 4, 7), ('gcamp6f', 51551, 5, 8), ('gcamp6f', 51552, 3, 4)], axis='index')
    with Figure(fig_folder.joinpath("encoder-all-comp.svg"), figsize=(8, 9)) as axes:
        res = ["median: "] + str(data.groupby('group').median()).split('\n')[2:]
        group_names = ('wt', "gcamp6f", 'glt1', 'dredd')
        group_strs = ["WT", "gcamp6f", "GLT1", "Gq"]
        annotation = list()
        for (idx, x), (idy, y) in combinations(enumerate(group_names), 2):
            p = mannwhitneyu(data.loc[x, ], data.loc[y, ], True, 'two-sided').pvalue
            res.append(f"mann u, {x} v. {y} p={p}")
            if p < 0.05:
                annotation.append(((idx, idy), p))
        print_stats("encoder full r2", res)
        values = [data.loc[x, ].groupby(["id", "session"]).mean() for x in group_names]
        boxplots = plots.boxplot(axes[0], values, whis=(10., 90.), zorder=1, showfliers=False, colors=colors)
        plots.dots(axes[0], values, zorder=3, s=24, jitter=0.02)
        axes[0].set_xticklabels(group_strs)
        plots.annotate_boxplot(axes[0], boxplots, 24, 1.2, annotation)


def piechart():
    """the proportion of R^2 split among behavior, as measured by R^2 averaged across neurons."""
    data: pd.DataFrame = pd.read_csv(data_folder.joinpath("encoding_minimal.csv")).set_index(["group", "id", "fov", "session", "name"])  # type: ignore
    data[data < 0] = 0
    group_mean = data.groupby("group").mean().assign(
        rest=lambda x: ((x['start'] + x['reward'] + x['isMoving'] + x['trajectory'])
                        / 4))[["hit", "delay", "speed", "rest"]]
    labels = ["Hit/Miss", "Response Time", "Speed", "Rest"]
    for group_str in ('wt', 'glt1', 'dredd', "gcamp6f"):
        group = group_mean.loc[group_str]
        explode = np.zeros(len(group), dtype=float)
        explode[np.argmax(group)] = 0.2
        with Figure(fig_folder.joinpath(f"pie-{group_str}.svg"), figsize=(6, 6)) as axes:
            axes[0].pie(group.values / group.values.sum(), explode, labels,
                        ["#619CFF", "#00BA38", "#F8766D", "#84b1b0"], "%1.1f%%", shadow=True)
            axes[0].axis("equal")

def piechart_data():
    """export data to be used in d3"""
    data: pd.DataFrame = pd.read_csv(data_folder.joinpath("encoding_minimal.csv")).set_index(["group", "id", "fov", "session", "name"])  # type: ignore
    data[data < 0] = 0
    # group_mean = data.groupby("group").mean().assign(
    #    rest=lambda x: ((x['start'] + x['reward'] + x['isMoving'] + x['trajectory'])
    #                    / 4))[["all", "delay", "hit", "speed", "rest"]]
    group_mean = data.groupby("group").median().assign(
        rest=lambda x: ((x['all'] - x['delay'] - x['hit'] - x['speed'])
                        / 4))[["all", "delay", "hit", "speed", "rest"]]
    labels = ["All", "Response Time", "Hit/Miss", "Speed", "Rest"]
    group_mean.columns = labels
    group_mean.index = pd.Index(["Gq", "gCaMP6f", "GLT1", "WT"], name="group")
    group_mean.tn_csv(data_folder.joinpath("encoding_minimal_mean.csv"))

def piechart_winner():
    """the proportion of neurons where a behavior as the most R^2"""
    data: pd.DataFrame = pd.read_csv(data_folder.joinpath("encoding_minimal.csv")).set_index(["group", "id", "fov", "session", "name"])  # type: ignore
    data[data < 0] = 0
    shortened = data.assign(
        rest=lambda x: ((x['start'] + x['reward'] + x['isMoving'] + x['trajectory'])
                        / 4))[["hit", "delay", "speed", "rest"]]
    groups = shortened.idxmax(1).groupby(level=0).value_counts().unstack()[["hit", "delay", "speed", "rest"]]
    labels = ["Hit/Miss", "Response Time", "Speed", "Rest"]
    for group_str in ("wt", "glt1", "dredd", "gcamp6f"):
        group = groups.loc[group_str]
        explode = np.zeros(len(group), dtype=float)
        explode[np.argmax(group)] = 0.2
        with Figure(fig_folder.joinpath(f"pie-{group_str}-max-neuron.svg"), figsize=(6, 6)) as axes:
            axes[0].pie(group.values / group.values.sum(), explode, labels,
                        ["#619CFF", "#00BA38", "#F8766D", "#84b1b0"], "%1.1f%%", shadow=True)
            axes[0].axis("equal")

def boxplot_rest():
    def notch(x) -> float:
        low_perc, high_perc = np.quantile(x, [0.25, 0.75])
        iqr = high_perc - low_perc
        return iqr * 0.5 * 1.57 / np.sqrt(len(x))
    data = pd.read_csv(data_folder.joinpath("encoding_minimal.csv"))
    columns = data.columns[1:9]
    data[columns] = data[columns].clip(0)
    features = ['start', 'reward', 'isMoving', 'trajectory']
    groups = ["wt", "glt1", "dredd"]
    columns = [x[y] for y in features for x in (data.query(f"group == '{i}'") for i in groups)]
    positions = np.arange(4 * len(features)).reshape(-1, 4)[:, 1:].ravel()
    xticks = np.arange(4 * len(features)).reshape(-1, 4)
    xticks[:, :2] += 1
    xticks_labels = ["WT", "GLT1", "\n", "Gq"] * len(features)
    for idx, feature in enumerate(features):
        xticks_labels[2 + idx * 4] = "\n" + feature
    with Figure(fig_folder.joinpath("boxplot-rest.svg"), figsize=(12, 6)) as axes:
        patches = plots.boxplot(axes[0], columns, positions=positions, notch=True, colors=cycle(colors[0: 3]),
                                whiskerprops={"linewidth": 2}, whis=(10., 90.), showfliers=False)
        axes[0].set_xticks(xticks.ravel())
        axes[0].set_xticklabels(xticks_labels, {"fontsize": 16})
        axes[0].set_ylabel("Proportion of Neural Activity\nExplained ($R^2$)", {"fontsize": 16})
        axes[0].set_ylim([-0.01, 0.12])
        for group, color in zip(groups, cycle(colors[0: 3])):
            group_data = data.query(f"group == '{group}'")["all"]
            notch_size = notch(group_data)
            median = np.median(group_data)
            axes[0].axhline(median, color=color)
            axes[0].fill_between([0.5, len(features) * 4 - 0.5],
                                 median - notch_size, median + notch_size, color=color, alpha=0.5)
        plots.annotate_boxplot(axes[0], patches, 16, 1.0, [((3, 4), 4.30E-7), ((9, 11), 0.048)])
        for start, end in zip(patches["medians"][::3], patches["medians"][2::3]):
            left_line = start.get_data()[0]
            width = left_line[1] - left_line[0]
            right_line = end.get_data()[0]
            x = [left_line[0] - width / 2, right_line[1] + width / 2]
            axes[0].plot(x, [start.get_data()[1][0]] * 2, linestyle=":", color="black", zorder=4, linewidth=2)
            axes[0].plot(x, [start.get_data()[1][0]] * 2, linestyle=":", color="black", zorder=4, linewidth=2)

def main_distributions():
    data = pd.read_csv(data_folder.joinpath("encoding_minimal.csv"))
    for behavior in ("all", "hit", "delay", "speed"):
        values = data[behavior].values
        value_by_group = [values[data["group"] == group_str] for group_str in ("wt", "glt1", "dredd")]
        value_by_group = [value[value > 0.01] for value in value_by_group]
        with Figure(fig_folder.joinpath(f"dist-{behavior}.svg"), figsize=(9, 3), show=True) as axes:
            axes[0].hist(value_by_group, 50, color=colors[0: 3], rwidth=1, lw=0)

df = pd.read_csv(data_folder.joinpath("encoding_minimal.csv"))
df.groupby("group").mean()
df_mean = pd.read_csv(data_folder.joinpath("encoding_minimal_mean_bak.csv"))
