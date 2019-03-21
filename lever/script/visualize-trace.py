## Show traces, multiple traces, or lever traces with neuron traces
from typing import Set
import numpy as np
from os import makedirs
from os.path import join, expanduser
from matplotlib.axes import Axes
import toml
from noformat import File
from algorithm.array import DataFrame
from algorithm.utils import map_tree
from algorithm.time_series import resample
from mplplot import Figure
from lever.reader import load_mat

project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
img_folder = join(project_folder, 'report', 'img')
res_folder = join(project_folder, 'report', 'measure')
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 1.4}

with open(join(project_folder, 'data', 'recording.toml')) as fp:
    mice = {group_str: [{'group': group_str, **x} for x in group] for group_str, group in toml.load(fp).items()}
files = map_tree(lambda x: (File(join(project_folder, "data", x["path"]))), mice)
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
#
def _scale(x):
    return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)

def example_traces(ax: Axes, record_file: File, start: float, end: float, cells: Set[int]):
    """Visualize calcium trace of cells and the lever trajectory"""
    lever_trajectory = load_mat(record_file["response"])
    calcium_trace = DataFrame.load(record_file["measurement"])
    neuron_rate = record_file.attrs['frame_rate']
    l_start, l_end = np.rint(np.multiply([start, end], lever_trajectory.sample_rate)).astype(np.int_)
    c_start, c_end = np.rint(np.multiply([start, end], neuron_rate)).astype(np.int_)
    ax.plot(np.linspace(0, l_end - l_start, l_end - l_start),  # lever trajectory
            _scale(lever_trajectory.values[0][l_start: l_end]), COLORS[1])
    time = np.linspace(0, calcium_trace.shape[1] / neuron_rate, lever_trajectory.shape[1])
    spacing = iter(range(0, 500, 2))
    for idx, row in enumerate(calcium_trace.values):
        if idx in cells:
            ax.plot(time[c_start: c_end] - l_start, _scale(row[c_start: c_end]) + next(spacing))
    stim_onsets = lever_trajectory.timestamps[
        (lever_trajectory.timestamps > l_start) & (lever_trajectory.timestamps < l_end)]\
        / lever_trajectory.sample_rate - l_start
    for x in stim_onsets:
        ax.axvline(x=x, color=COLORS[2])

def all_traces(record_file: File, ax: Axes):
    """plot full traces of all neurons and trial onsets"""
    lever_trajectory = load_mat(record_file["response"])
    calcium_trace = _scale(DataFrame.load(record_file["measurement"]).values)
    time = np.linspace(0, lever_trajectory.shape[1] / lever_trajectory.sample_rate, lever_trajectory.shape[1])
    ax.plot(time, _scale(lever_trajectory.values[0]) - 5, COLORS[1])
    for idx, row in enumerate(calcium_trace):
        ax.plot(time, row + idx * 5)
    for point in lever_trajectory.timestamps / lever_trajectory.sample_rate:  # trial onsets
        ax.axvline(x=point, color=COLORS[2])

def full_trace(record_file: File, ax: Axes):
    activity = DataFrame.load(record_file["spike"])
    lever = load_mat(record_file['response'])
    new_lever = resample(lever.values[0], lever.sample_rate, record_file.attrs['frame_rate'])
    for x in range(activity.shape[0]):
        ax.plot(_scale(activity.values[x, :] / 5))
    ax.plot(_scale(new_lever) / 2 - 3, color=COLORS[0])
## WT example
all_plots = [
    ("wt", 0, (121.5, 123.5), {4, 12, 16, 20, 25, 30}),
    ("glt1", 1, (368.75, 370.75), {0, 18, 28, 35, 40, 41}),
    ("dredd", 0, (33.2, 35.2), {3, 7, 36, 37, 44, 49}),
    ("wt", 0, (53, 63), {29}),
    ("wt", 0, (96.5, 106.5), {0}),
    ("wt", 0, (121, 131), {33}),
    ("wt", 0, (275, 285), {29}),
    ("wt", 0, (72.5, 82.5), {22}),
]
makedirs(join(img_folder, "example-trace"), exist_ok=True)
for idx, (group_str, session_idx, (start, end), neuron_ids) in enumerate(all_plots):
    with Figure(join(img_folder, "example-trace", f"sample_{group_str}_{idx}.svg"), (6, 2)) as ax:
        example_traces(ax[0], files[group_str][session_idx][0], start, end, neuron_ids)
##
group_str, session_id = "wt", 0
with Figure(join(img_folder, f"sample_all_{group_str}_{session_id}.svg"), (6, 2)) as ax:
    all_traces(files[group_str][session_id], ax[0])
##
group_str, session_id = "dredd", 3
with Figure(join(img_folder, f"all_{group_str}_{session_id}.svg"), (6, 2)) as ax:
    all_traces(files[group_str][session_id], ax[0])
