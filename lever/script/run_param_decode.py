## setup
from typing import Tuple
from os.path import join, expanduser
import numpy as np
import toml
from sklearn.svm import SVR
from entropy_estimators import mutual_info
from noformat import File
from algorithm.utils import map_tree_parallel, map_tree
from algorithm.array import DataFrame
from algorithm.time_series import take_segment
from lever.filter import devibrate_trials
from lever.utils import get_trials
from decoder.validate import cross_predict
from utils_cmp import plot_scatter, compare

project_folder = expanduser("~/Sync/project/2017-leverpush")
res_folder = join(project_folder, "report", "measure")
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
with open(join(project_folder, "data/recording.toml")) as fp:
    mice = toml.load(fp)
files = map_tree(lambda x: File(join(project_folder, 'data', x['path'])), mice)
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 0.9}
## actual running
def run_amp_power(data_file: File) -> Tuple[float, float, float, float]:
    """Try to decode the max lever trajectory amplitude of each trial.
    Returns:
        pre_amp_power: mutual info between predicted (from neuron activity before motor onset)
            and real amplitude of trials in one session
        post_amp_power: mutual info between predicted (from neuron activity before motor onset)
            and real amplitude of trials in one session
    """
    lever = get_trials(data_file, motion_params)
    neuron = DataFrame.load(data_file['spike'])
    resampled_onsets = np.rint(lever.trial_anchors * (5 / 256)).astype(np.int_) - 3
    folded = np.stack([take_segment(trace, resampled_onsets, 6) for trace in neuron.values])
    mask, filtered = devibrate_trials(lever.values, motion_params['pre_time'])
    mask &= np.any(folded > 0, axis=(0, 2))
    amp = filtered[mask, 25: 64].max(axis=1) - filtered[mask, 0: 15].mean(axis=1)
    speed = np.diff(filtered[mask, 5:50], axis=1).max(axis=1)
    svr_rbf = SVR('rbf', 3, 1E-7, cache_size=1000)
    X = folded[:, mask, 0: 3].swapaxes(0, 1).reshape(mask.sum(), -1)
    pre_amp_hat = cross_predict(X.T, amp, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    pre_v_hat = cross_predict(X.T, speed, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    X = folded[:, mask, 3:].swapaxes(0, 1).reshape(mask.sum(), -1)
    post_amp_hat = cross_predict(X.T, amp, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    post_v_hat = cross_predict(X.T, speed, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    return (mutual_info(pre_amp_hat, amp, 'sturges'), mutual_info(post_amp_hat, amp, 'sturges'),
            mutual_info(pre_v_hat, amp, 'sturges'), mutual_info(post_v_hat, speed, 'sturges'))

def run_amp_mi(data_file: File) -> Tuple[float, float, float, float]:
    """Try to decode the max lever trajectory amplitude of each trial.
    Returns:
        pre_amp_power: mutual info between predicted (from neuron activity before motor onset)
            and real amplitude of trials in one session
        post_amp_power: mutual info between predicted (from neuron activity before motor onset)
            and real amplitude of trials in one session
    """
    lever = get_trials(data_file, motion_params)
    neuron = DataFrame.load(data_file['spike'])
    resampled_onsets = np.rint(lever.trial_anchors * (5 / 256)).astype(np.int_) - 3
    folded = np.stack([take_segment(trace, resampled_onsets, 6) for trace in neuron.values])
    mask, filtered = devibrate_trials(lever.values, motion_params['pre_time'])
    mask &= np.any(folded > 0, axis=(0, 2))
    amp = filtered[mask, 25: 64].max(axis=1) - filtered[mask, 0: 15].mean(axis=1)
    speed = np.diff(filtered[mask, 5:50], axis=1).max(axis=1)
    svr_rbf = SVR('rbf', 3, 1E-7, cache_size=1E3)
    X = folded[:, mask, 0: 3].swapaxes(0, 1).reshape(mask.sum(), -1)
    pre_amp_hat = cross_predict(X.T, amp, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    pre_v_hat = cross_predict(X.T, speed, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    X = folded[:, mask, 3:].swapaxes(0, 1).reshape(mask.sum(), -1)
    post_amp_hat = cross_predict(X.T, amp, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    post_v_hat = cross_predict(X.T, speed, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), 10, False)
    return (mutual_info(pre_amp_hat, amp), mutual_info(post_amp_hat, amp),
            mutual_info(pre_v_hat, amp), mutual_info(post_v_hat, speed))

## testing
## main
if __name__ == '__main__':
    res = map_tree_parallel(run_amp_power, files)
    with open(join(res_folder, "param_decode_amp_disc_mi.npz"), 'bw') as fpb:
        np.savez_compressed(fpb, wt=res['wt'], glt1=res['glt1'], dredd=res['dredd'])
##
data_file = files['wt'][2]
print('done')
data = np.load(join(res_folder, "param_decode_amp_disc_mi.npz"))
plot_scatter({a: b[:, 1] for a, b in data.items()})
compare({a: b[:, 3] for a, b in data.items()})
## testing mi
