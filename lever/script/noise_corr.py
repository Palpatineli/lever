## plot day to day network connections for a common set of neurons
from typing import Optional, Dict, List
from os.path import join, expanduser
import numpy as np
from scipy.cluster.hierarchy import linkage
import toml
from noformat import File
from algorithm.array import DataFrame, common_axis, search_ar
from algorithm.time_series import fold_by
from mplplot import network as corr_graph
from lever.analysis import motion_corr, classify_cells, noise_autocorrelation, noise_correlation, reliability
from lever.utils import load_mat, MotionParams
from lever.plot import fancy_dendrogram
from mplplot import Figure, stacked_bar

motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 0.9}
project_folder = expanduser("~/Sync/project/2017-leverpush")
img_folder = join(project_folder, "report", "img")
res_folder = join(project_folder, "report", "measure")
COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
with open(join(project_folder, "data/recording.toml")) as fp:
    mice = toml.load(fp)
files = {x['session']: File(join(project_folder, 'data', x['path'])) for x in mice['wt']['0304']['3']}

def _sem(x):
    return np.std(x) / np.sqrt(len(x))

# Cell: classify cells from the last day
def draw_classify_neurons(data_file: File, neuron_ids: Optional[np.ndarray] = None):
    lever = load_mat(data_file['response'])
    neuron = DataFrame.load(data_file['spike'])
    if neuron_ids is not None:
        neuron = neuron[search_ar(neuron_ids, neuron.axes[0]), :]
    neuron_rate = data_file.attrs['frame_rate']
    corr = motion_corr(lever, neuron, neuron_rate, 16000, motion_params)
    good, bad, anti = [corr[x, 0] for x in classify_cells(corr, 0.001)]
    with Figure(join(img_folder, "good_unrelated_cmp.svg"), (4, 6)) as ax:
        ax[0].bar((0, 1), [good.mean(), np.r_[bad, anti].mean()], yerr=[_sem(good), _sem(np.r_[bad, anti])])

# cell: network graph
def draw_network_graph(data_files: Dict[int, File], params: MotionParams, threshold: int = 16000):
    """Draw neuron functional connection for each session, with neurons colored by the last session.
    Args:
        data_files: {day_id: int, data_file: File}
        params: classify_cells need ["quiet_var", "window_size", "event_thres", "pre_time"]
        threshold: threshold for motion_corr, single linked cluster distance
    """
    last_day = data_files[max(data_files.keys())]
    neurons = common_axis([DataFrame.load(x['spike']) for x in data_files.values()])
    neuron_rate = last_day.attrs['frame_rate']
    final_corr_mat = noise_autocorrelation(load_mat(last_day['response']), neurons[-1], neuron_rate)
    categories = classify_cells(motion_corr(last_day, neurons[-1], neuron_rate, threshold, params), 0.001)
    layout = corr_graph.get_layout(final_corr_mat, neurons[-1].axes[0])
    for (day_id, data_file), neuron in zip(data_files.items(), neurons):
        corr_mat = noise_autocorrelation(load_mat(data_file['response']), neuron, neuron_rate)
        with Figure(join(img_folder, f"network-day-{day_id:02d}.svg")) as ax:
            corr_graph.corr_plot(ax[0], corr_mat, categories, neuron.axes[0], layout=layout)
    print('done')

# Cell: cluster proportions
def draw_hierarchy(data_files: Dict[int, File]):
    neurons = common_axis([DataFrame.load(x['spike']) for x in files.values()])
    for (day_id, data_file), neuron in zip(files.items(), neurons):
        lever = load_mat(data_file['response'])
        corr_mat = noise_autocorrelation(lever, neuron, data_file.attrs['frame_rate'])
        with Figure() as (ax,):
            ax.set_title(f"day-{day_id:02d}")
            fancy_dendrogram(linkage(corr_mat, 'average'), ax=ax)

# Cell: stacked bar from manually labeled files
ClusterFile = Dict[str, Dict[str, List[int]]]

def draw_stacked_bar(cluster_file: ClusterFile):
    days = ('5', '10', '13', '14')
    res = [[len(cluster_file[day].get(str(cluster_id), []))
            for cluster_id in range(1, 15)] for day in days]
    with Figure() as (ax,):
        stacked_bar(ax, res, COLORS)
        ax.set_xticks(range(len(days)), days)

# Cell: noise correlation
def _take_triu(x):
    return x[np.triu_indices(x.shape[0], 1)]

def draw_noise(data_files: Dict[int, File], neuron_id: int, params: MotionParams):
    last_day = max(data_files.keys())
    lever = load_mat(data_files[last_day]['response'])
    neuron_rate = data_files[last_day].attrs['frame_rate']
    neurons = common_axis([DataFrame.load(x['spike']) for x in data_files.values()])
    good, bad, anti = classify_cells(motion_corr(
        lever, neurons[-1], neuron_rate, 16000, params), 0.001)
    amp = list()
    corrs: Dict[str, List[List[float]]] = {'good': [], 'unrelated': [], 'between': []}
    for (day_id, data_file), neuron in zip(data_files.items(), neurons):
        if day_id == last_day:
            continue
        lever = load_mat(data_file['response'])
        corrs['good'].append(_take_triu(noise_autocorrelation(lever, neuron[good], neuron_rate)))
        corrs['unrelated'].append(_take_triu(noise_autocorrelation(lever, neuron[bad | anti], neuron_rate)))
        corrs['between'].append(_take_triu(noise_correlation(lever, neuron[good], neuron[bad | anti], neuron_rate)))
        lever.center_on("motion", **params)
        neuron_trials = fold_by(neuron, lever, neuron_rate, True)
        amp.append(neuron_trials.values[np.argwhere(neuron.axes[0] == neuron_id)[0, 0], :, :].max(axis=1))
    with Figure(join(project_folder, 'report', 'img', f'noise_corr_{neuron_id}.svg')) as (ax,):
        day_ids = [x for x in data_files.keys() if x != last_day]
        for idx, (group_str, group) in enumerate(corrs.items()):
            ax.errorbar(day_ids, [np.mean(x) for x in group],
                        yerr=[_sem(x) for x in group], color=COLORS[idx], label=group_str)
        ax2 = ax.twinx()
        ax2.errorbar(day_ids, [np.mean(x) for x in amp], [_sem(x) for x in amp], color=COLORS[-1])
        ax.set_title(str(neuron_id))
        ax.legend()

# Cell: Mesuare the inter-cell correlation between trials of typical pushes for single neurons on different days
def draw_neuron_corr(data_files: Dict[int, File], params: MotionParams, fov_id: str = None):
    neurons = common_axis([DataFrame.load(x['spike']) for x in data_files.values()])
    last_day = max(data_files.keys())
    lever = load_mat(data_files[last_day]['response'])
    neuron_rate = data_files[last_day].attrs['frame_rate']
    good, bad, anti = classify_cells(motion_corr(
        lever, neurons[-1], neuron_rate, 16000, params), 0.001)
    result_list = list()
    for (day, data_file), neuron in zip(data_files.items(), neurons):
        lever.center_on('motion')  # type: ignore
        motion_neurons = fold_by(neuron, lever, neuron_rate, True)
        result_list.append([reliability(motion_neuron) for motion_neuron in motion_neurons.values])
    result = np.array(result_list)

    with Figure(join(img_folder, ("neuron_corr.svg" if fov_id is None else f"{fov_id}.svg"))) as ax:
        ax[0].plot(list(data_files.keys()), result[:, good])
## actual running
common_id = common_axis([DataFrame.load(x['spike']) for x in files.values()])[-1].axes[0]
draw_classify_neurons(files[14], common_id)
draw_hierarchy(files)
draw_stacked_bar(toml.load(join(res_folder, 'cluster.toml')))  # type: ignore
neuron_ids = toml.load(join(res_folder, "0304-neurons.toml"))['neuron_id']
draw_noise(files, 27, motion_params)
draw_neuron_corr(files, motion_params)
##
