##
from typing import Dict, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import toml
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec, fold_by
from lever.reader import load_mat
from lever.filter import devibrate_rec, devibrate
from pypedream import Task, getLogger, Input, InputObj

from align import res_spike
__all__ = ['res_trial_neuron', 'res_filter_log']

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe")
Task.save_folder = proj_folder.joinpath("data", "interim")
Input.save_folder = proj_folder.joinpath("data")
mice = [dict(x) for x in toml.load(proj_folder.joinpath("data", "index", "index.toml"))["recordings"]]
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 0.9}

class LogInput(InputObj):
    def __init__(self, data_folder: Path, name: str):
        self.file_path = data_folder
        case_id, fov_id, session_id = name.split('-')
        self.name = f"{case_id}-{session_id}"

    def time(self) -> float:
        log_path = self.file_path.joinpath("original", "log", self.name + ".mat")
        return log_path.stat().st_mtime

    def load(self) -> SparseRec:
        log_path = self.file_path.joinpath("original", "log", self.name + ".mat")
        return load_mat(str(log_path))

input_log = Input(LogInput, "2019-04-30T18:01")

def filter_log(log: SparseRec) -> SparseRec:
    log.values[0] = devibrate(log.values[0], log.sample_rate)
    return log

task_filtered_log = Task(filter_log, "2019-04-30T18:28", "filtered-log")
res_filter_log = task_filtered_log(input_log)

def make_trial_log(log: SparseRec, params: Dict[str, float]) -> SparseRec:
    lever = log.center_on("motion", **params).fold_trials()
    lever.values = np.squeeze(lever.values, 0)
    lever.axes = lever.axes[1:]
    return devibrate_rec(lever, params)

task_trial_log = Task(make_trial_log, "2019-04-30T18:38", "trial-log", extra_args=(motion_params, ))
res_trial_log = task_trial_log(res_filter_log)

def make_trial_neuron(trial_log: SparseRec, spike_framerate: Tuple[Dict[str, np.ndarray], float]) -> DataFrame:
    spikes, frame_rate = spike_framerate
    # trial_neurons should be [neuron, trial, time_points]
    return fold_by(DataFrame.load(spikes), trial_log, frame_rate, True)

task_trial_neuron = Task(make_trial_neuron, "2019-05-02T16:27", "trial-neuron")
res_trial_neuron = task_trial_neuron([res_trial_log, res_spike])

##
def main():
    logger = getLogger("astrocyte", "exp-log.log")
    pool = Pool(max(1, cpu_count() - 2))
    params_dict = [(info['name'], logger) for info in mice]
    result = pool.starmap(res_trial_neuron.run, params_dict)
    return result

##
if __name__ == '__main__':
    main()
