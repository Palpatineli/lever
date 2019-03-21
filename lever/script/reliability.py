##
from typing import Tuple, List, Dict
from itertools import combinations
from os.path import join, expanduser
import toml
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from noformat import File
from algorithm.array import DataFrame
from algorithm.stats import perm_test
from algorithm.utils import map_tree, zip_tree
from algorithm.time_series import fold_by
from lever.filter import devibrate_trials
from lever.utils import MotionParams, get_trials, filter_empty_trial_sets
from mplplot.importer import Figure

motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 0.9}
project_folder = expanduser("~/Sync/project/2017-leverpush")
img_folder = join(project_folder, "report", "img")
res_folder = join(project_folder, "report", "measure")
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
with open(join(project_folder, "data/recording.toml")) as fp:
    mice = toml.load(fp)
files = {x['session']: File(join(project_folder, 'data', x['path'])) for x in mice['wt']['0304']['3']}
##
def neuron_lever_unsampled(data_file: File, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        neural_activity, lever
    """
    lever = get_trials(data_file)
    neurons = fold_by(DataFrame.load(data_file["spike"]), lever, data_file.attrs['frame_rate'], True)
    neurons, lever = filter_empty_trial_sets(neurons.values, lever.values[0])
    mask, filtered = devibrate_trials(lever, params["pre_time"])
    return neurons[:, mask, :], lever[mask, :]

def neural_vs_motor_reliability(record_file: File, params: MotionParams, neural_no: int = 10, trial_no: int = 10,
                                repeat: int = 1000) -> np.ndarray:
    """Bootstrap the distribution to """
    neural, lever = neuron_lever_unsampled(record_file, params)
    trial_corr_s = list()
    neuron_corr_s = list()
    for _ in range(repeat):
        trial_indices = np.random.choice(np.arange(neural.shape[1]), trial_no, False)
        neuron_indices = np.random.choice(np.arange(neural.shape[0]), neural_no, False)
        trial_corr_s.append(np.corrcoef(lever[trial_indices, :])[np.triu_indices(len(trial_indices), 1)].mean())
        neuron_corr_s.append(np.mean(
            [np.corrcoef(neural[neuron_index, trial_indices, :])[np.triu_indices(len(trial_indices), 1)].mean()
             for neuron_index in neuron_indices]))
    print("done: ", record_file.file_name)
    return trial_corr_s, neuron_corr_s

def pairwise_corr(record_file: File, params: MotionParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        lever_corr, neuron_corr
    """
    neural, lever = neuron_lever_unsampled(record_file, params)
    indices = np.triu_indices(lever.shape[0], 1)
    return np.corrcoef(lever)[indices], np.stack([np.corrcoef(neuron)[indices] for neuron in neural])

def lever_corr(record_file, record_info, params: MotionParams):
    lever = get_trials(record_file, params)
    mask, filtered = devibrate_trials(lever.values, params['pre_time'])
    lever_value = lever.values[0, mask, :]
    indices = np.triu_indices(lever_value.shape[0], 1)
    return np.corrcoef(lever_value)[indices]

def slope_dist(points: np.ndarray, repeats: int) -> np.ndarray:
    result: List[float] = list()
    size = points.shape[0]
    sample_range = np.arange(size)
    for _ in range(repeats):
        samples = points[np.random.choice(sample_range, size, True), :]
        result.append(lstsq(np.vstack([samples[:, 0], np.ones(size)]).T, samples[:, 1])[0][0])
    return result
##
def find_slope_dist(params: MotionParams):
    def _get_best_corr(x):
        lever_corr, neural_corr = pairwise_corr(x[0], params)
        return np.vstack([lever_corr, np.quantile(neural_corr, 0.8, axis=0)]).T
    corr_cmp = map_tree(_get_best_corr, files)
    result, raw_result = dict(), dict()
    for group_str, group in corr_cmp.items():
        points = np.vstack(group)
        result[group_str] = slope_dist(points, 200)
        raw_result[group_str] = points
    with Figure(join(img_folder, "corr_dist_08.svg"), (3, 8)) as (ax,):
        ax.scatter(raw_result['wt'][:, 0], raw_result['wt'][:, 1], color="#268bd2", s=1)
        ax.scatter(raw_result['glt1'][:, 0], raw_result['glt1'][:, 1], color="#d33682", s=1)
        ax.scatter(raw_result['dredd'][:, 0], raw_result['dredd'][:, 1], color="#859900", s=1)
    with Figure(join(img_folder, "slope_dist_06.svg"), (6, 4)) as (ax,):
        ax.hist(result['wt'], 20, color="#268bd2", alpha=0.75)
        ax.hist(result['glt1'], 20, color="#d33682", alpha=0.75)
        ax.hist(result['dredd'], 20, color="#859900", alpha=0.75)
##
def find_lever_corr():
    corrs = map_tree(lambda x: lever_corr(x[0], x[1], motion_params), zip_tree(files, mice))
    print("wt vs. dredd: ", perm_test(np.hstack(corrs['wt']), np.hstack(corrs['dredd'])))
    print([np.median(case) for case in corrs['dredd']])
    print([perm_test(x0, x1) for x0, x1 in combinations(corrs["wt"], 2)])
##
def find_corr():
    corr_pairs = map_tree(lambda x: pairwise_corr(x, motion_params), files)
    corr_pairs['glt1'] = corr_pairs['glt1'][0: 2] + corr_pairs['glt1'][4: 6] + corr_pairs['glt1'][7: 8]
    results = dict()
    for group in ('wt', 'glt1', 'dredd'):
        result = list()
        for idx, (lever, neuron) in enumerate(corr_pairs[group]):
            # mask = (0.75 > lever) & (lever > 0.5)
            mask = lever > 0.75
            result.append(neuron[:, mask].ravel())
        results[group] = result
    fig, axes = plt.subplots(nrows=3, sharex=True)
    for ax, group in zip(axes, ('wt', 'glt1', 'dredd')):
        ax.hist(results[group], 50, density=True)

if __name__ == '__main__':
    find_corr()
##
