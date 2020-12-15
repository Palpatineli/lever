##
from typing import Tuple, Dict
from pathlib import Path
import numpy as np

from pypedream import Task, get_result
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec, fold_by
from lever.filter import devibrate_rec
from lever.script.steps.log import res_filter_log
from lever.script.steps.align import res_spike
from lever.script.steps.utils import read_index
from mplplot import Figure

proj_folder = Path("~/Sync/project/2018-leverpush-chloe").expanduser()
Task.save_folder = proj_folder.joinpath("data", "interim")
fig_folder = proj_folder.joinpath("report", "fig")
mice = read_index(proj_folder)

motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 1.9}

def make_sample_neurons(spike_framerate: Tuple[Dict[str, np.ndarray], float], log: SparseRec,
                        params: Dict[str, float]) -> SparseRec:
    lever = log.center_on("motion", **params).fold_trials()
    lever.values = np.squeeze(lever.values, 0)
    lever.axes = lever.axes[1:]
    filtered = devibrate_rec(lever, params)
    spikes, frame_rate = spike_framerate
    return fold_by(DataFrame.load(spikes), filtered, frame_rate, True)

task_trial_neuron = Task(make_sample_neurons, "2020-09-01T23:04", "trial-neuron-2s", extra_args=(motion_params,))
res_trial_neuron = task_trial_neuron([res_spike, res_filter_log])

##
def main():
    trial_neurons = get_result([x.name for x in mice][0:1], [res_trial_neuron], 'trial-neuron-2s-run')
    values = trial_neurons[0][0].values
    with Figure(fig_folder.joinpath("classifier", "example-neurons.svg"), show=True) as axes:
        for id_neuron, neuron in enumerate(values[:20, 0: 4, :]):
            for id_trial, trial in enumerate(neuron):
                axes[0].plot(range(id_trial * 11, id_trial * 11 + 10), trial / trial.max() * 5 + id_neuron * 6,
                             color='red')

if __name__ == '__main__':
    main()
