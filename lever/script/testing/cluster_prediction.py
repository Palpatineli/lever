"""Extract lever trajectory and correlate with neuronal activity.
Cluster trials by trajectory hierarchy (distance by fastdtw).
And give prediction score from neuron to trajectory clustering."""
##
from typing import Union, Dict, List
from os.path import join, expanduser
import pickle as pkl
import numpy as np
import toml
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from mplplot import Figure, tsplot
from noformat import File
from algorithm.array import DataFrame
from algorithm.utils import quantize, map_tree, map_tree_parallel, zip_tree, unflatten, flatten
from algorithm.time_series import fold_by
from lever.cluster.main import get_cluster_labels, get_linkage, kNNdist, min_points, pca_bisect
from lever.decoding.cluster import precision, get_neurons
from lever.plot import get_threshold
from lever.filter import devibrate_trials
from lever.utils import get_trials, MotionParams

project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
img_folder = join(project_folder, 'report', 'img')
res_folder = join(project_folder, 'report', 'measure')
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 1.4}

with open(join(project_folder, 'data', 'recording.toml')) as fp:
    mice = {group_str: [{'group': group_str, **x} for x in group] for group_str, group in toml.load(fp).items()}
files = map_tree(lambda x: (File(join(project_folder, "data", x["path"]))), mice)
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff", "#50D0B8FF"]
# Interactively set threshold and save in file attrs
def get_thresholds(data_file: File, overwrite: bool = False):
    linkage, mask = get_linkage(data_file, motion_params)
    if overwrite or ('hierarchy_threshold' not in data_file.attrs):
        threshold = get_threshold(linkage)
        data_file.attrs['hierarchy_threshold'] = threshold
        data_file.attrs.flush()
    threshold = data_file.attrs['hierarchy_threshold']
    clusters = get_cluster_labels(linkage, threshold)
    return clusters, mask
# Show Templates from two clustering methods: kmeans vs. manual hierarchical
def draw_cluster_3d(data_file: File, params: MotionParams, labels: np.ndarray, info: Dict[str, Union[str, int]]):
    trials = get_trials(data_file, params).values
    points = PCA(3).fit_transform(trials)
    session_name = f"{info['group']}-{info['id']}-{info['fov']:02d}{info['session']:02d}"
    with Figure(join(img_folder, 'pca', session_name + '.svg'), grid=(1, 2), projection='3d') as axes:
        for (clusters, title), ax in zip(((k_means(trials, 2)[1], "k-means"), (quantize(labels), "hierarchy")), axes):
            ax.set_title(title)
            for unique, color in zip(np.unique(clusters), COLORS):
                cluster = clusters == unique
                ax.scatter(points[cluster, 0], points[cluster, 1], points[cluster, 2], c=color)
# Compare templates between kmeans and hierarchical main cluster
def draw_template(data_file: File, params: MotionParams, labels: np.ndarray, info: Dict[str, Union[str, int]]):
    trials = get_trials(data_file, params).values
    labels = (k_means(trials, 2)[1] == 1, quantize(labels).astype(np.bool))
    session_name = f"template_cmp-{info['group']}-{info['id']}-{info['fov']:02d}{info['session']:02d}.svg"
    with Figure(join(img_folder, 'template', session_name), (6, 4), grid=(1, 2)) as axes:
        for ax, label, colors, name in zip(axes, labels, (COLORS[0: 2], COLORS[2:]), ('kmeans', 'hierarchy')):
            ax.set_title(name)
            tsplot(ax, trials[label, :], color=colors[0])
            tsplot(ax, trials[~label, :], color=colors[1])

def visualize_kmeans(data_file: File, params: MotionParams):
    lever = get_trials(data_file)
    lever.scale()
    dist = kNNdist(lever.values, 4)[:, -1]
    plt.plot(sorted(dist))
    print("min points: ", min_points(lever.values, 16.7))
# comapre corr and raw
def draw_twoway_comp(scores: List[List[np.ndarray]], comp_names: List[str], group_names: List[str]):
    group_size, comp_size = len(scores), len(scores[0])
    assert(group_size == len(group_names))
    assert(comp_size == len(comp_names))
    means = [[np.mean(y) for y in x] for x in scores]
    sems = [[np.std(y) / np.sqrt(len(y)) for y in x] for x in scores]
    ylim = min(0.45, np.floor(min([min(x) for x in means]) * 20) / 20)
    with Figure(join(img_folder, "prediction_performance_bar.png"), (6, 4)) as (ax,):
        ax.set_ylim(ylim, 1.0)
        ax.axhline(y=0.5, color='k')
        for idx, (mean, sem) in enumerate(zip(means, sems)):
            bars = ax.bar([idx * 3, idx * 3 + 1], mean, yerr=[x * 2 for x in sem], color=COLORS[: 2])
        ax.set_xticks([x + 0.5 for x in range(0, 3 * group_size, 3)])
        ax.set_xticklabels(group_names)
        ax.legend(bars, comp_names)

def draw_perm_comp(scores: List[Dict[str, List[np.ndarray]]], names: List[str]):
    def _perm_shuffle(x, iter_no):
        return np.array([np.mean(np.random.choice(x, x.shape[0], True)) for _ in range(iter_no)])
    scores = [{group_str: np.vstack(group) for group_str, group in score.items()} for score in scores]
    new_scores = list()
    for score in scores:
        new_score = dict()
        for name, group in score.items():
            new_score[name] = np.vstack(group)
        new_scores.append(new_score)
    return new_scores

    for group in score_raw.keys():
        with Figure(join(img_folder, "prediction", f"pred-perf-bis-{group}.png"), (6, 4)) as (ax,):
            for score, name, color in zip(scores, names, COLORS):
                ax.hist(_perm_shuffle(score, 1000), 50, alpha=0.5, color=color, label=name)
            ax.legend()

def visualize_prediction(data_file: File, params: MotionParams):
    """Show 3 PCs of neuron activity and the plane in linear SVC classifier."""
    lever = get_trials(data_file, params)
    trials = fold_by(DataFrame.load(data_file['spike']), lever, data_file.attrs['frame_rate'], True)
    k_cluster = k_means(lever.values, 2)[1].astype(np.bool)
    main_cluster, null_cluster = np.flatnonzero(k_cluster)[:20], np.flatnonzero(~k_cluster)[:10]
    all_neurons = np.hstack([trials.values.take(main_cluster, 1), trials.values.take(null_cluster, 1)])
    all_results = np.array([0] * 20 + [1] * 10)
    training_X = all_neurons.swapaxes(0, 1).reshape(all_neurons.shape[1], -1)
    pca_weigths = PCA(20).fit_transform(training_X)
    classifier = SVC(kernel='linear')
    classifier.fit(pca_weigths, all_results)
    classifier.score(pca_weigths, all_results)
    coef = classifier.coef_[0]
    intercept = classifier.intercept_
    xx, yy = np.meshgrid(np.linspace(pca_weigths[:, 0].min(), pca_weigths[:, 0].max(), 20),
                         np.linspace(pca_weigths[:, 1].min(), pca_weigths[:, 1].max(), 20))
    z = (-intercept - xx * coef[0] - yy * coef[1]) / coef[2]
    with Figure(projection='3d') as ax:
        ax[0].scatter(*pca_weigths[:, 6: 9].T, color=np.asarray(COLORS)[all_results], s=50)
        ax[0].plot_surface(xx, yy, z, alpha=0.4, color=COLORS[-1])
        ax[0].set_zlim(pca_weigths[:, 2].min(), pca_weigths[:, 2].max())

def get_score_mask(x, **kwargs):
    data_file, (clusters, mask) = x
    return precision(get_neurons(data_file, motion_params), data_file.file_name, clusters, mask=mask, **kwargs)

def get_score(x, **kwargs):
    data_file, clusters = x
    return precision(get_neurons(data_file, motion_params), data_file.file_name, clusters, **kwargs)

# Actually running
## Label Clusters
def run_label_clusters():
    results = map_tree(lambda x: get_thresholds(x, True), files)
    with open(join(res_folder, "clustering.pkl"), 'wb') as fpb:
        pkl.dump(results, fpb)
## Draw Template
def run_draw_template():
    cluster_labels = unflatten(np.load(join(res_folder, "clustering.npz")))
    map_tree(lambda x: draw_cluster_3d(*x), zip_tree(files, cluster_labels, mice))
    map_tree(lambda x: draw_template(*x), zip_tree(files, cluster_labels, mice))
## Get precision
# clustered with k_means
def run_k_means_precision():
    def _get_k_means(data_file: File):
        return k_means(get_trials(data_file, motion_params).values, 2)[1]
    cluster_labels = map_tree(_get_k_means, files)
    score_raw = map_tree(get_score, zip_tree(files, cluster_labels))
    score_corr = map_tree(lambda x: get_score(x, corr=True), zip_tree(files, cluster_labels))
    np.savez_compressed(join(res_folder, "k_prediction_raw.npz"), **flatten(score_raw))
    np.savez_compressed(join(res_folder, "k_prediction_corr.npz"), **flatten(score_corr))
    print('done')
## Clustered with hierarchy
def run_hierarchy_precision():
    with open(join(res_folder, "clustering.pkl"), 'rb') as fpb:
        cluster_labels = pkl.load(fpb)
    score_raw = map_tree_parallel(get_score_mask, zip_tree(files, cluster_labels))
    score_corr = map_tree_parallel(lambda x: get_score_mask(x, corr=True), zip_tree(files, cluster_labels))
    np.savez_compressed(join(res_folder, "h_prediction_raw.npz"), **flatten(score_raw))
    np.savez_compressed(join(res_folder, "h_prediction_corr.npz"), **flatten(score_corr))
    print('done')
## clustered with bisection in PC1
def run_pc_bisec_precision():
    def _get_bisect(data):
        mask, filtered = devibrate_trials(get_trials(data[0], motion_params)[0], motion_params["pre_time"])
        return pca_bisect(filtered), mask
    bisect_labels = map_tree(_get_bisect, files)
    score_raw = map_tree(get_score_mask, zip_tree(files, bisect_labels))
    score_corr = map_tree(lambda x: get_score_mask(x, corr=True), zip_tree(files, bisect_labels))
    np.savez_compressed(open(join(res_folder, "bis_prediction_raw.npz"), 'wb'), **flatten(score_raw))
    np.savez_compressed(open(join(res_folder, "bis_prediction_corr.npz"), 'wb'), **flatten(score_corr))
    print('done')
## and draw bisect prediction
def run_draw_prediction():
    score_raw = unflatten(np.load(join(res_folder, "bis_prediction_raw.npz")))
    score_corr = unflatten(np.load(join(res_folder, "bis_prediction_corr.npz")))
    print('mean (raw vs. corr): {} vs. {}', map_tree(np.mean, score_raw), map_tree(np.mean, score_corr))

    def _draw_pred(raw, corr, info):
        name = f"pred-perf-bis-{info['group']}-{info['idx']}.png"
        with Figure(join(img_folder, "prediction", name), (6, 4)) as (ax,):
            ax.hist(raw, 50, alpha=0.5, color=COLORS[2])
            ax.hist(corr, 50, alpha=0.5, color=COLORS[1])
    map_tree(lambda x: _draw_pred(x[0], x[1], x[2]), zip_tree(score_raw, score_corr, mice))
    draw_perm_comp([score_raw, score_corr], ['raw', 'corr'])
    ## show comaprison of scoring between raw and corr
    score_raw = unflatten(np.load(join(res_folder, 'k_prediction_raw.npz')))
    score_corr = unflatten(np.load(join(res_folder, 'k_prediction_corr.npz')))
    score_ids = [("wt", [1]), ("glt1", [1]), ("dredd", [0])]
    score = list()
    for group, indices in score_ids:
        raw_group = np.hstack([score_raw[group][idx] for idx in indices])
        corr_group = np.hstack([score_corr[group][idx] for idx in indices])
        score.append([raw_group, corr_group])
    draw_twoway_comp(score, ['raw', 'corr'], ['wt', 'glt1', 'dredd'])
## permutation test on these scores
##
if __name__ == '__main__':
    run_hierarchy_precision()
##
