##
from typing import Tuple, List, Dict
from os.path import join, expanduser
import pickle as pkl
import numpy as np
from numpy import newaxis
import toml
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import ttest_ind
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from noformat import File
from tqdm import tqdm
from mplplot import Figure, labeled_heatmap
from entropy_estimators import mutual_info
from algorithm.array import DataFrame
from algorithm.stats import perm_test, combine_test
from algorithm.utils import map_tree, map_tree_parallel, map_table
from lever.reader import load_mat
from lever.filter import devibrate
from lever.decoding import kalman, linear, particle, svr
from lever.decoding.validate import cross_predict, decoder_power
from lever.decoding.utils import Bounds
from lever.plot import plot_scatter

project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
img_folder = join(project_folder, 'report', 'img')
res_folder = join(project_folder, 'report', 'measure')
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 1.4}

with open(join(project_folder, 'data', 'recording.toml')) as fp:
    mice = {group_str: [{'group': group_str, **x} for x in group] for group_str, group in toml.load(fp).items()}
dredd_mice = toml.load(join(project_folder, 'data', 'cno.toml'))
files = map_tree(lambda x: (File(join(project_folder, "data", x["path"]))), mice)
dredd_files = map_tree(lambda x: (File(join(project_folder, "data", x["path"]))), dredd_mice)
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
#
def null_dist(path_hat, y):
    res = [mutual_info(np.random.permutation(path_hat), y) for _ in tqdm(range(1000))]
    np.savez_compressed(join(res_folder, "svr_power_wt2_perm.npz"), result=np.array(res))
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
    np.savez_compressed(join(project_folder, "report", "measure", ""), res=res)

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
    np.savez_compressed(join(res_folder, "decoding.npz"), **flatten)

def run_svr_power():
    def svr_power(data_file: File, neuron_no: int = 20) -> Tuple[float, List[float]]:
        lever = load_mat(data_file['response'])
        values = devibrate(lever.values[0], sample_rate=lever.sample_rate)
        y = InterpolatedUnivariateSpline(lever.axes[0], values)(data_file['spike']['y'])[1:]
        X = data_file['spike']['data'][:, 1:].copy()
        decoder = svr.predictor_factory(y, gamma=3E-9, C=12, epsilon=1E-3)
        single_power = [cross_predict(x[newaxis, :], y, decoder, section_mi=True)[1].mean() for x in X]
        mask = np.greater_equal(single_power, sorted(single_power)[-neuron_no])
        path_hat, _ = cross_predict(X[mask, :], y, decoder)
        return mutual_info(y, path_hat), single_power

    result = map_tree_parallel(svr_power, dredd_files, verbose=2)
    from pickle import dump
    with open(join(res_folder, "dredd_svr_power.pkl"), 'wb') as fp:
        dump(result, fp)
    print('done')

def show_power():
    with open(join(res_folder, 'dredd_svr_power.pkl'), 'br') as fp:
        result = pkl.load(fp)
    scores = {y: [a[0] for a in x] for y, x in result.items()}
    fig, (ax, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
    plot_scatter(scores, COLORS, ax)
    print(combine_test(scores, [perm_test, ttest_ind]))
    bins = np.linspace(-0.01, 0.25, 100)
    means = list()
    stds = list()
    for (group_str, group), color in zip(scores.items(), COLORS):
        dist = [np.mean(np.random.choice(group, len(group), True)) for _ in range(1000)]
        ax1.hist(dist, bins, color=color, alpha=0.5, label=group_str)
        means.append(np.mean(dist))
        stds.append(np.std(dist))
    ax2.barh(np.arange(3), means, xerr=stds, color=COLORS[0: 3])
    ax2.set_xlabel("information (bit)")
    plt.show()

def show_traces():
    data_file = files['wt'][0]
    lever = load_mat(data_file['response'])
    values = devibrate(lever.values[0], sample_rate=lever.sample_rate)
    y = InterpolatedUnivariateSpline(lever.axes[0], values)(data_file['spike']['y'])[1:]
    X = data_file['spike']['data'][:, 1:]
    lever_hat, powers = cross_predict(X, y, svr.predictor_factory(y))
    lever_hat = svr.predictor_factory(y)(X, y, X)
    plt.plot(y, color='blue')
    plt.plot(lever_hat, color='red')
    print("mutual info: ", mutual_info(y, lever_hat))

## Test: SVR parameters
def svr_parameters(data_file: File, info: Dict[str, str]):
    lever = load_mat(data_file['response'])
    values = devibrate(lever.values[0], sample_rate=lever.sample_rate)
    y = InterpolatedUnivariateSpline(lever.axes[0], values)(data_file['spike']['y'])[1:]
    X = data_file['spike']['data'][:, 1:]
    gammas = np.linspace(-8, -5, 12, endpoint=False)
    Cs = np.linspace(3, 15, 12, endpoint=False)

    def pred(gamma, C):
        hat = cross_predict(X, y, svr.predictor_factory(y, gamma=10 ** gamma, C=C, epsilon=1E-3), section_mi=False)
        return mutual_info(y, hat)
    res = map_table(pred, gammas, Cs)
    save_path = join(res_folder, f"svr_params_test_{info['id']}_{info['session']}.npz")
    np.savez_compressed(save_path, values=np.asarray(res), axes=[gammas, Cs])
    res_df = DataFrame(np.asarray(res), [gammas, Cs])
    with Figure() as (ax,):
        labeled_heatmap(ax, res_df.values, res_df.axes[1], res_df.axes[0])
    print('done')
##
def single_power():
    """Requires data file from run_svr_power."""
    with open(join(res_folder, "svr_power.pkl"), 'rb') as fp:
        result = pkl.load(fp)
    ind_scores = {x: [a[1] for a in y] for x, y in result.items()}

    def fn(x):
        return np.cumsum(x / sum(x))

    with Figure(grid=(1, 3)) as (ax1, ax2, ax3):
        ind = [sorted(x, reverse=True) for x in ind_scores['wt']]
        [ax1.plot(fn(x), color='blue', alpha=0.5) for x in ind]
        ind = [sorted(x, reverse=True) for x in ind_scores['glt1']]
        [ax2.plot(fn(x), color='red', alpha=0.5) for x in ind]
        ind = [sorted(x, reverse=True) for x in ind_scores['dredd']]
        [ax3.plot(fn(x), color='green', alpha=0.5) for x in ind]

    wt = [np.divide(sorted(x[0: 50], reverse=True), sum(x[0: 50])) for x in ind_scores['wt']]
    wt_no = np.hstack([np.arange(len(a)) for a in wt])
    glt1 = [np.divide(sorted(x[0: 50], reverse=True), sum(x[0: 50])) for x in ind_scores['glt1']]
    glt1_no = np.hstack([np.arange(len(a)) for a in glt1])
    dredd = [np.divide(sorted(x[0: 50], reverse=True), sum(x[0: 50])) for x in ind_scores['dredd']]
    dredd_no = np.hstack([np.arange(len(a)) for a in dredd])
    plt.plot(wt_no)
    plt.show()
    from scipy.stats import linregress
    slope, _, _, _, std = linregress(wt_no, np.log(np.hstack(wt)))
    wt_res = (slope - std * 2.58, slope + std * 2.58)
    slope, _, _, _, std = linregress(glt1_no, np.log(np.hstack(glt1)))
    glt1_rest = (slope - std * 2.58, slope + std * 2.58)
    slope, _, _, _, std = linregress(dredd_no, np.log(np.hstack(dredd)))
    dredd_rest = (slope - std * 2.58, slope + std * 2.58)

def examine_saline(data_file):
    data_file = dredd_files['cno'][5]
    lever = load_mat(data_file['response'])
    values = devibrate(lever.values[0], sample_rate=lever.sample_rate)
    y = InterpolatedUnivariateSpline(lever.axes[0], values)(data_file['spike']['y'])[1:]
    X = data_file['spike']['data'][:, 1:]
    decoder = svr.predictor_factory(y, gamma=3E-7, C=11, epsilon=1E-3)
    single_power = [mutual_info(y, cross_predict(x[newaxis, :], y, decoder, section_mi=False)) for x in X]
    hat_0 = cross_predict(X, y, decoder, section_mi=False)
    mask = np.greater_equal(single_power, sorted(single_power)[-20])
    hat_1 = cross_predict(X[mask, :], y, decoder, section_mi=False)
    plt.plot(y, color='blue')
    hat = decoder(X, y, X)
    plt.plot(hat, color='green')
    plt.plot(hat_0, color='red')
    plt.plot(hat_1, color='orange')
    print("hat_1: ", mutual_info(hat_1, y), " hat_0: ", mutual_info(hat_0, y))

##
## SVR.fit is reentrant: does not carry over previous fit result
if __name__ == '__main__':
    # run_svr_power()
    svr_parameters(dredd_files['saline'][2], dredd_mice['saline'][2])
