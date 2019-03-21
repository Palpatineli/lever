##
from typing import Tuple, List
from os.path import join, expanduser
import numpy as np
from numpy import newaxis
import toml
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from noformat import File
from tqdm import tqdm
from algorithm.stats import perm_test
from algorithm.utils import map_tree, map_tree_parallel
from lever.reader import load_mat
from entropy_estimators import mutual_info
from decoder import kalman, linear, particle, svr
from decoder.validate import cross_predict, decoder_power
from decoder.utils import Bounds
from cmp_utils import jitter

project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
img_folder = join(project_folder, 'report', 'img')
res_folder = join(project_folder, 'report', 'measure')
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 1.4}

with open(join(project_folder, 'data', 'recording.toml')) as fp:
    mice = {group_str: [{'group': group_str, **x} for x in group] for group_str, group in toml.load(fp).items()}
files = map_tree(lambda x: (File(join(project_folder, "data", x["path"]))), mice)
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
#
def null_dist(path_hat, y):
    res = [mutual_info(np.random.permutation(path_hat), y) for _ in tqdm(range(1000))]
    with open(join(project_folder, "report/measure/svr_power_wt2_perm.npz"), 'wb') as fp:
        np.savez_compressed(fp, result=np.array(res))
    plt.hist(res, 50)

def null_fit(data_file):
    lever = load_mat(data_file['response'])
    y = InterpolatedUnivariateSpline(lever.axes[0], lever.values[0])(data_file['spike']['y'])[1:]
    X = data_file['spike']['data'][:, 1:]
    res = list()
    for _ in range(200):
        y_perm = np.random.permutation(y.copy())
        y_hat_perm, _ = cross_predict(X, y_perm, svr.predictor_factory(y_perm))
        res.append(mutual_info(y_perm, y_hat_perm))
    with open(join(project_folder, "report", "measure", ""), 'wb') as fp:
        np.savez_compressed(fp, res=res)

def compare_decoder():
    import matplotlib.pyplot as plt
    data_file = files['wt'][2]
    lever = load_mat(data_file['response'])
    y = InterpolatedUnivariateSpline(lever.axes[0], lever.values[0])(data_file['spike']['y'])[1:]
    X = data_file['spike']['data'][:, 1:]
    bounds: Bounds = (tuple(np.quantile(y, [0.001, 0.999])), (-2, 1), (-5, 5))  # type: ignore
    print("mutual info:")
    plt.plot(y, color=COLORS[0], alpha=0.5, label="trajectory")
    path_hats = dict()
    powers = dict()
    for color, (name, decoder) in zip(COLORS[1:], (("particle", particle.decoder_factory(bounds)),
                                                   ("kalman", kalman.decoder_factory(bounds)),
                                                   ("linear", linear.decoder_factory(bounds)),
                                                   ("svr", svr.decoder_factory(SVR('rbf', 3, 1E-7, cache_size=1000))))):
        path_hat, power = cross_predict(X, y, decoder)
        info = mutual_info(y, path_hat)
        path_hats[name] = path_hat
        powers[name] = power
        plt.plot(path_hat, color=color, alpha=0.5, label="{}: {}".format(name, info))
    plt.legend()
##
def main():
    result = map_tree(lambda x: decoder_power(x, particle.decoder_factory), files)
    flatten = {group_str: np.array(group) for group_str, group in result.items()}
    with open(join(project_folder, "report", "measure", "decoding.npz"), 'wb') as fp:
        np.savez_compressed(fp, **flatten)

def run_svr_power():
    def svr_power(data_file: File, neuron_no: int = 20) -> Tuple[float, List[float]]:
        lever = load_mat(data_file['response'])
        resampled = InterpolatedUnivariateSpline(lever.axes[0], lever.values[0])(data_file['spike']['y'])[1:]
        neurons = data_file['spike']['data'][:, 1:].copy()
        decoder = svr.decoder_factory(SVR(kernel="rbf", gamma=1E-8, cache_size=1000))
        single_power = [cross_predict(neuron[:, newaxis], resampled, decoder)[1].mean()
                        for neuron in neurons]
        mask = np.greater_equal(single_power, sorted(single_power)[-neuron_no])
        path_hat, _ = cross_predict(neurons[mask, :], resampled, decoder)
        return mutual_info(resampled, path_hat), single_power

    result = map_tree_parallel(svr_power, files, verbose=2)
    from pickle import dump
    with open(join(project_folder, "report", "measure", "svr_power.pkl"), 'wb') as fp:
        dump(result, fp)
    flatten = {group_str: np.array([np.median(x[1]) for x in group]) for group_str, group in result.items()}
    with open(join(project_folder, "report", "measure", "svr_power.npz"), 'wb') as fp:
        np.savez_compressed(fp, **flatten)

def show_power():
    result = {y: x.copy() for y, x in np.load(join(project_folder, "report", "measure", "svr_power.npz")).items()}
    for x in ('wt', 'glt1', 'dredd'):
        result[x][result[x] < 0] = 0
    fig, (ax, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
    j_wt = jitter(0, result['wt'], ax, color=COLORS[0])
    j_glt = jitter(1, result['glt1'][2:], ax, color=COLORS[1])
    j_dredd = jitter(2, result['dredd'], ax, color=COLORS[2])
    ax.set_ylim(-1, 3)
    ax.set_xlim(-0.01, 0.25)
    perm_test(result['wt'], result['glt1'][2:])
    perm_test(result['wt'], result['dredd'])
    dist_wt = [np.mean(np.random.choice(result['wt'], len(result['wt']), True)) for _ in range(1000)]
    dist_glt1 = [np.mean(np.random.choice(result['glt1'][2:], len(result['glt1'][2:]), True)) for _ in range(1000)]
    dist_dredd = [np.mean(np.random.choice(result['dredd'], len(result['dredd']), True)) for _ in range(1000)]
    bins = np.linspace(-0.01, 0.25, 100)
    ax1.hist(dist_wt, bins, color=COLORS[0], alpha=0.5, label='wildtype')
    ax1.hist(dist_glt1, bins, color=COLORS[1], alpha=0.5, label='GLT-1')
    ax1.hist(dist_dredd, bins, color=COLORS[2], alpha=0.5, label='dredd')
    ax2.barh(np.arange(3), [np.mean(dist_wt), np.mean(dist_glt1), np.mean(dist_dredd)],
             xerr=[np.std(dist_wt), np.std(dist_glt1), np.std(dist_dredd)],
             color=COLORS[0: 3])
    ax2.legend([j_wt, j_glt, j_dredd], ['wildtype', 'GLT-1', 'Dredd'])
    ax2.set_xlabel("information (bit)")

def show_traces():
    data_file = files['wt'][2]
    lever = load_mat(data_file['response'])
    y = InterpolatedUnivariateSpline(lever.axes[0], lever.values[0])(data_file['spike']['y'])[1:]
    X = data_file['spike']['data'][:, 1:]
    lever_hat, powers = cross_predict(X, y, svr.predictor_factory(y))
    plt.plot(y, color='blue')
    plt.plot(lever_hat, color='red')
    print("mutual info: ", mutual_info(y, lever_hat))

if __name__ == '__main__':
    null_fit(files['wt'][2])
##
