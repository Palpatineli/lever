## setup
from typing import Tuple
from os.path import join, expanduser
import numpy as np
import toml
from scipy.stats import ttest_ind, ks_2samp
from sklearn.svm import SVR
from entropy_estimators import mutual_info
from noformat import File
from algorithm.utils import map_tree_parallel, map_tree
from algorithm.array import DataFrame
from algorithm.stats import combine_test, perm_test
from algorithm.time_series import take_segment
from lever.filter import devibrate_trials
from lever.utils import get_trials
from lever.decoding.validate import cross_predict
from lever.plot import plot_scatter

project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
res_folder = join(project_folder, "report", "measure")
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
mice = toml.load(join(project_folder, "data/recording.toml"))
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
    VALIDATE_FOLD = 10
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
    pre_amp_hat = cross_predict(X.T, amp, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), VALIDATE_FOLD, False)
    pre_v_hat = cross_predict(X.T, speed, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), VALIDATE_FOLD, False)
    X = folded[:, mask, 3:].swapaxes(0, 1).reshape(mask.sum(), -1)
    post_amp_hat = cross_predict(X.T, amp, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), VALIDATE_FOLD, False)
    post_v_hat = cross_predict(X.T, speed, lambda x, y, y_t: svr_rbf.fit(x.T, y).predict(y_t.T), VALIDATE_FOLD, False)
    return (mutual_info(pre_amp_hat, amp), mutual_info(post_amp_hat, amp),
            mutual_info(pre_v_hat, speed), mutual_info(post_v_hat, speed))
## testing
## main
if __name__ == '__main__':
    res = map_tree_parallel(run_amp_power, files)
    np.savez_compressed(join(res_folder, "param_decode_amp_mi.npz"), **res)
##
data = dict(np.load(join(res_folder, "param_decode_amp_mi.npz")))
values = {a: b[:, 2] for a, b in data.items()}
values['glt1'] = [values['glt1'][2:], values['glt1'][:2]]
plot_scatter(values, COLORS)
values['glt1'] = values['glt1'][0]
print(combine_test({a: b for a, b in values.items()}, [ks_2samp, ttest_ind, perm_test]))
## testing mi
