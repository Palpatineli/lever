##
from typing import Tuple, Dict, List
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec, fold_by
from pypedream import Task, get_result
from lever.script.steps.log import res_trial_log, res_trial_log_quiet
from lever.script.steps.align import res_spike

__all__ = ['res_trial_neuron']
proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe")
Task.save_folder = proj_folder.joinpath("data", "interim")
mice: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "index.csv")).set_index(["id", "session"])  # type: ignore

def make_trial_neuron(trial_log: SparseRec, spike_framerate: Tuple[Dict[str, np.ndarray], float]) -> DataFrame:
    spikes, frame_rate = spike_framerate
    # trial_neurons should be [neuron, trial, time_points]
    return fold_by(DataFrame.load(spikes), trial_log, frame_rate, True)

task_trial_neuron = Task(make_trial_neuron, "2019-05-02T16:27", "trial-neuron")
res_trial_neuron = task_trial_neuron([res_trial_log, res_spike])

def scale(x: np.ndarray) -> np.ndarray:
    """x should not be larger than 3D. scale on last axis"""
    std = x.std(axis=-1, keepdims=True)
    mask = std == 0
    std[mask] = 1
    z = (x - x.mean(axis=-1, keepdims=True)) / std
    tile_size = (z.shape[0],) if z.ndim == 1 else ((1, z.shape[1]) if z.ndim == 2 else (1, 1, z.shape[2]))
    z[np.tile(mask, tile_size)] = 0
    return z

def make_related_neurons(trial_log: SparseRec, spike_framerate: Tuple[Dict[str, np.ndarray], float]) -> np.ndarray:
    """Calcualte the pearson r between neuron activity and trajectory. Returns an array of p values for each neuron."""
    spike, frame_rate = spike_framerate
    trial_spikes = fold_by(DataFrame.load(spike), trial_log, frame_rate, False)
    trajectory = scale(trial_log.values).ravel()
    p_values = [pearsonr(neuron, trajectory)[1] for neuron
                in scale(trial_spikes.values).reshape(trial_spikes.shape[0], -1)]
    return np.array(p_values)

task_related_neurons = Task(make_related_neurons, "2019-07-10T15:46", "related-neurons")
res_related_neurons = task_related_neurons([res_trial_log_quiet, res_spike])

##
def merge(result: List[np.ndarray]):
    cno_grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "cno-schedule.csv")).set_index(["id", "session"])  # type: ignore
    grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "grouping.csv")).set_index(["id", "session"])  # type: ignore
    grouping = pd.concat([grouping, cno_grouping])
    merged = pd.DataFrame([x for y in result for x in y], index=np.repeat(mice.index, [len(x) for x in result]), columns=["p"])\
        .join(grouping)
    merged.to_csv(proj_folder.joinpath("data", "analysis", "related.csv"))

def main():
    result = get_result(mice.name.to_list(), [res_related_neurons])[0]
    merge(result)

if __name__ == '__main__':
    main()
##
