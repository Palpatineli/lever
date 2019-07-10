## get behaviorla differences between animal groups
from os.path import join, expanduser
from noformat import File
import numpy as np
import toml
from scipy.stats import linregress, ttest_ind, ks_2samp
from algorithm.utils import map_tree
from algorithm.stats import combine_test, perm_test
from algorithm.time_series.event import find_response_onset
from lever.utils import get_trials
from lever.filter import devibrate_trials
from lever.reader import load_mat
from lever.plot import plot_scatter
from mplplot import Figure

project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
# project_folder = expanduser("~/Sync/project/2017-leverpush")
img_folder = join(project_folder, 'report', 'img')
res_folder = join(project_folder, 'report', 'measure')
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 1.4}

mice = toml.load(join(project_folder, 'data', 'recording.toml'))
files = map_tree(lambda x: (File(join(project_folder, "data", x["path"]))), mice)
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
## Amplitude
def get_amp(data_file: File) -> float:
    mask, filtered = devibrate_trials(get_trials(data_file, motion_params).values, motion_params['pre_time'])
    return np.quantile(filtered[mask, 25: 64].max(axis=1) - filtered[mask, 0: 15].mean(axis=1), 0.75)

def get_speed(data_file: File) -> float:
    mask, filtered = devibrate_trials(get_trials(data_file, motion_params).values, motion_params['pre_time'])
    speed = np.diff(filtered[mask, 5:50], axis=1).max(axis=1)
    return np.mean(speed)

def get_delay(data_file: File) -> float:
    lever = load_mat(data_file['response'])
    params = {x: y for x, y in motion_params.items() if x in ('quiet_var', 'window_size', 'event_thres')}
    motion_onsets, stim_onsets, _, correct_trials, _ = find_response_onset(lever, **params)
    return np.mean((motion_onsets - stim_onsets[correct_trials]) / lever.sample_rate)

def get_hitrate(data_file: File) -> float:
    lever = load_mat(data_file['response'])
    params = {x: y for x, y in motion_params.items() if x in ('quiet_var', 'window_size', 'event_thres')}
    _, _, _, correct_trials, _ = find_response_onset(lever, **params)
    return correct_trials.mean()

def get_reliability(data_file: File) -> float:
    mask, filtered = devibrate_trials(get_trials(data_file, motion_params).values, motion_params['pre_time'])
    return np.corrcoef(filtered[mask, 15:64])[np.triu_indices(mask.sum(), 1)].mean()

##
amps = map_tree(get_amp, files)
np.savez_compressed(join(res_folder, 'behavior', 'amps.npz'), **amps)
plot_scatter(amps, COLORS)
print(combine_test(amps, [ttest_ind, ks_2samp, perm_test]))
## Speed
speeds = map_tree(get_speed, files)
np.savez_compressed(join(res_folder, 'behavior', 'speeds.npz'), **speeds)
plot_scatter(speeds, COLORS)
print(combine_test(speeds, [ttest_ind, ks_2samp, perm_test]))
## Delay
delays = map_tree(get_delay, files)
np.savez_compressed(join(res_folder, 'behavior', 'delay.npz'), **delays)
plot_scatter(delays, COLORS)
print(combine_test(delays, [ttest_ind, ks_2samp, perm_test]))
## Hit rate
hitrate = map_tree(get_hitrate, files)
plot_scatter(hitrate, COLORS)
print(combine_test(hitrate, [ttest_ind, ks_2samp, perm_test]))
np.less(hitrate['wt'], 0.5)
## Reliability (mean cross-correlation)
reliability = map_tree(get_reliability, files)
reliability['glt1'] = [reliability['glt1'][:2], reliability['glt1'][2:]]
plot_scatter(reliability, COLORS)
reliability['glt1'] = reliability['glt1'][1]
print(combine_test(reliability, [ttest_ind, ks_2samp, perm_test]))
## Tryout: All the above vs. prediction performance
perf = np.load("/home/palpatine/Sync/project/2018-leverpush-chloe/report/measure/svr_power.npz")
with Figure() as (ax,):
    ax.scatter(np.r_[perf['wt'], perf['glt1'], perf['dredd']],
               np.r_[speeds['wt'], speeds['glt1'], speeds['dredd']])
    plot_scatter(perf, ax=ax)
    ax.scatter(perf['wt'], hitrate['wt'], color=COLORS[0])
    ax.scatter(perf['glt1'][2:], hitrate['glt1'][2:], color=COLORS[1])
    ax.scatter(perf['dredd'], hitrate['dredd'], color=COLORS[2])
    ax.scatter(perf['glt1'][:2], hitrate['glt1'][:2], color=COLORS[3])
print("speed vs. perf (wt): ", linregress(speeds['wt'], perf['wt']))
print("hitrate vs. perf (glt1): ", linregress(hitrate['glt1'][2:], perf['glt1'][2:]))
## CNO vs. saline temporary
cno_mice = toml.load(join(project_folder, 'data', 'cno.toml'))
cno_files = map_tree(lambda x: File(join(project_folder, 'data', x['path'])), cno_mice)
from scipy.stats import ttest_rel
with Figure(grid=(5, 1)) as axes:
    for fn, ax in zip((get_amp, get_speed, get_delay, get_hitrate, get_reliability), axes):
        cno_feature = {x: y[1:] for x, y in map_tree(fn, cno_files).items()}
        print(f"{fn.__name__}:")
        print(combine_test(cno_feature, [ttest_rel, perm_test]))
        # plot_scatter(cno_feature, COLORS, ax)
        # ax.set_ylim((-1, 2))
##
plot_scatter(cno_features)
