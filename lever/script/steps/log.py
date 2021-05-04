##
from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
from algorithm.time_series import SparseRec
from lever.reader import load_mat
from lever.filter import devibrate_rec, devibrate
from pypedream import Task, Input, InputObj

__all__ = ['res_filter_log']

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe")
Task.save_folder = proj_folder.joinpath("data", "interim")
Input.save_folder = proj_folder.joinpath("data")
mice: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "index.csv"))  # type: ignore
motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 0.9}

class LogInput(InputObj):
    def __init__(self, data_folder: Path, name: str):
        self.file_path = data_folder
        self.name = name

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

def make_trial_log(log: SparseRec, params: Dict[str, float], center: str = "motion") -> SparseRec:
    lever = log.center_on(center, **params).fold_trials()
    lever.values = np.squeeze(lever.values, 0)
    lever.axes = lever.axes[1:]
    return devibrate_rec(lever, params)

task_trial_log = Task(make_trial_log, "2019-04-30T18:38", "trial-log", extra_args=(motion_params, ))
res_trial_log = task_trial_log(res_filter_log)

quiet_motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 1.0, "post_time": 1.0}
task_trial_log_quiet = Task(make_trial_log, "2019-04-30T18:38", "trial-log", extra_args=(quiet_motion_params, ))
res_trial_log_quiet = task_trial_log_quiet(res_filter_log)

full_trial_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.0, "post_time": 1.0}
task_stim_log = Task(make_trial_log, "2020-12-16T20:39", "stim-log", extra_args=(full_trial_params, "stim"))
res_stim_log = task_stim_log(res_filter_log)

##
def main():
    from pypedream import get_result
    result = get_result(mice.name.to_list(), [res_filter_log], "filtered-log")[0]
    return result

if __name__ == '__main__':
    main()
