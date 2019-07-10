## compare sectioned lever trajectory
from os.path import expanduser, join
import toml
import numpy as np
from scipy.stats import ttest_ind, ks_2samp
from noformat import File
from algorithm.utils import map_tree
from algorithm.stats import combine_test, perm_test
from lever.utils import MotionParams, get_trials
from lever.plot import plot_scatter
from lever.filter import devibrate_rec

motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.3, "post_time": 0.7}
proj_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
res_folder = join(proj_folder, "report", "measure")
mice = toml.load(join(proj_folder, 'data', 'recording.toml'))
files = map_tree(lambda x: File(join(proj_folder, 'data', x['path'])), mice)
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
#
def reliability(data: np.ndarray) -> float:
    t = data.shape[0]
    coef = 2 / (t ** 2 - t)
    return np.corrcoef(data, rowvar=True)[np.triu_indices(t, 1)].sum() * coef

def get_initial(data_file: File, params: MotionParams):
    lever = devibrate_rec(get_trials(data_file, params))
    pre_value = lever.values[:, :lever._pre // 2].mean(axis=1, keepdims=True)
    lever_off = int(np.median(np.argmax(lever.values[:, lever._pre:] <= pre_value, axis=1))) + lever._pre
    return reliability(lever.values[:, lever._pre // 2: lever_off])

def get_rise(data_file: File, params: MotionParams):
    lever = devibrate_rec(get_trials(data_file, params))
    lever_top = int(np.median(np.argmax(lever.values[:, lever._pre:], axis=1))) + lever._pre
    return reliability(lever.values[:, lever._pre // 2: lever_top])

def get_fall(data_file: File, params: MotionParams):
    lever = devibrate_rec(get_trials(data_file, params))
    pre_value = lever.values[:, :lever._pre // 2].mean(axis=1, keepdims=True)
    lever_off = np.argmax(lever.values[:, lever._pre:] <= pre_value, axis=1) + lever._pre
    lever_top = int(np.median([np.argmax(x[lever._pre: y]) for x, y in zip(lever.values, lever_off) if y > lever._pre])) + lever._pre
    lever_off = int(np.median(lever_off))
    return reliability(lever.values[:, lever_top: lever_off])

def fall_spread(data_file: File, params: MotionParams):
    lever = devibrate_rec(get_trials(data_file, params))
    pre_value = lever.values[:, :lever._pre // 2].mean(axis=1, keepdims=True)
    lever_off = np.argmax(lever.values[:, lever._pre:] <= pre_value, axis=1)
    return np.std(lever_off)

def get_later(data_file: File, params: MotionParams):
    lever = get_trials(data_file, params)
    pre_value = lever.values[:, :lever._pre // 2].mean(axis=1, keepdims=True)
    lever_off = int(np.median(np.argmax(lever.values[:, lever._pre:] <= pre_value, axis=1))) + lever._pre
    return reliability(lever.values[:, lever_off:])

##
initial_rel = map_tree(lambda x: get_initial(x, motion_params), files)
initial_rel['glt1'] = [initial_rel['glt1'][:2], initial_rel['glt1'][2:]]
plot_scatter(initial_rel, COLORS)
initial_rel['glt1'] = initial_rel['glt1'][1]
print(combine_test(initial_rel, [perm_test, ttest_ind, ks_2samp]))
##
rise_phase = map_tree(lambda x: get_rise(x, motion_params), files)
rise_phase['glt1'] = [rise_phase['glt1'][:2], rise_phase['glt1'][2:]]
plot_scatter(rise_phase, COLORS)
rise_phase['glt1'] = rise_phase['glt1'][1]
print(combine_test(rise_phase, [perm_test, ttest_ind, ks_2samp]))
##
fall_phase = map_tree(lambda x: get_fall(x, motion_params), files)
fall_phase['glt1'] = [fall_phase['glt1'][:2], fall_phase['glt1'][2:]]
plot_scatter(fall_phase, COLORS)
fall_phase['glt1'] = fall_phase['glt1'][1]
fall_phase = {x: [i for i in y if not np.isnan(i)] for x, y in fall_phase.items()}
print(combine_test(fall_phase, [perm_test, ttest_ind, ks_2samp]))
##
fall = map_tree(lambda x: fall_spread(x, motion_params), files)
fall['glt1'] = [fall['glt1'][:2], fall['glt1'][2:]]
plot_scatter(fall, COLORS)
fall['glt1'] = fall['glt1'][1]
print(combine_test(fall, [perm_test, ttest_ind, ks_2samp]))
##
after = map_tree(lambda x: get_later(x, motion_params), files)
after['glt1'] = [after['glt1'][:2], after['glt1'][2:]]
plot_scatter(after, COLORS)
after['glt1'] = after['glt1'][1]
print(combine_test(after, [perm_test, ttest_ind, ks_2samp]))
## compare cno and saline in same animals
from scipy.stats import ttest_rel
cno_mice = toml.load(join(proj_folder, 'data', 'cno.toml'))
cno_files = map_tree(lambda x: File(join(proj_folder, 'data', x['path'])), cno_mice)
cno_res = map_tree(lambda x: fall_spread(x, motion_params), cno_files)
ttest_rel(cno_res['cno'], cno_res['saline'])
##
