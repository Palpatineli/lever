## Series of small plots of lever trajectory between one WT case and one KO case, plus some neuron trace
from os.path import expanduser, join
import numpy as np
import toml
from noformat import File
from algorithm.array import DataFrame
from algorithm.time_series import fold_by
from mplplot import Figure, tsplot, labeled_heatmap
from lever.utils import MotionParams, load_mat
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.5, "post_time": 1.0}
project_folder = expanduser("~/Sync/project/2017-leverpush")
img_folder = join(project_folder, "report", "img")
res_folder = join(project_folder, "report", "measure")
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
mice = toml.load(join(project_folder, "data/recording.toml"))
files = {x['session']: File(join(project_folder, 'data', x['path'])) for x in mice['wt']['0304']['3']}
## cell: lever trajectory compare to neuron activity
def draw_neuron(day: int, data_file: File, neuron_id: int, params: MotionParams):
    """Draw one neuron in trial for one session, with bootstrapped spread as shadow."""
    lever = load_mat(data_file['response'])
    lever.center_on("motion", **params)
    neuron = DataFrame.load(data_file['spike'])
    traces = fold_by(neuron, lever, data_file.attrs['frame_rate'])
    traces = traces[np.flatnonzero(traces.axes[0] == neuron_id)[0], :, :]
    mask = np.all(traces.values > 0, axis=1)
    pre_value = traces.values[mask, 0: int(round(params['pre_time'] * lever.sample_rate))].mean(axis=1, keepdims=True)
    trace_values = traces.values[mask, :] - pre_value
    with Figure(join(img_folder, "neuron-trace", f"day-{day:02d}.svg"), (1, 4)) as (ax,):
        tsplot(ax, trace_values, time=traces.axes[2], color=COLORS[4])
        ax.set_title(f"day_{day:02d}")

# cell: neuron trajectory between animals
def compare_neurons(day: int, data_file_0: File, data_file_1: File, params: MotionParams):
    values = list()
    for data_file in (data_file_0, data_file_1):
        lever = load_mat(data_file['response'])
        lever.center_on('motion', **params)
        lever.fold_trials()
        pre_value = lever.values[0, :, 0: int(round(params['pre_time'] * lever.sample_rate))]\
            .mean(axis=1, keepdims=True)
        values.append(lever.values[0, :, :] - pre_value)
    with Figure(join(img_folder, 'neuron-trace', f"comp-day-{day:02d}.svg"), (1, 4)) as (ax,):
        tsplot(ax, values[0], time=lever.axes[2], color=COLORS[0])
        tsplot(ax, values[1], time=lever.axes[2], color=COLORS[1])
        ax.set_title(f"day-{day:02d}")

# cell: rasterplot between neurons of same animal
def draw_rasterplot(day: int, data_file: File, neuron_id: int, params: MotionParams):
    lever = load_mat(data_file['response'])
    lever.center_on('motion', **params)
    neurons = DataFrame.load(data_file['spike'])
    neuron_rate = data_file.attrs['frame_rate']
    traces = fold_by(neurons, lever, neuron_rate, True)[np.flatnonzero(neurons.axes[0] == neuron_id)[0], :, :]
    mask = np.all(traces.values > 0, axis=1)
    onset = int(round(params['pre_time'] * neuron_rate))
    with Figure(join(img_folder, 'neuron-trace', f"raster-day-{day}.svg"), (2, 4)) as (ax,):
        labeled_heatmap(ax, traces[mask, :] - traces[mask, 0: onset].mean(axis=1, keepdims=True), cmap="coolwarm")
        ax.set_title(f"day-{day}")
## Actual Running
for day in (5, 7, 9, 10, 12):
    draw_neuron(day, files[day], 9, motion_params)

wt_lever = {x['session']: File(join(project_folder, 'data', x['path'])) for x in mice['wt']['0303']['1']}
het_lever = {x['session']: File(join(project_folder, 'data', x['path'])) for x in mice['wt']['0301']['1']}
for day in (set(wt_lever.keys()) & set(het_lever.keys())):
    compare_neurons(day, wt_lever[day], het_lever[day], motion_params)

for day in (5, 7, 9, 10, 12):
    draw_rasterplot(day, files[day], 9, motion_params)
##
