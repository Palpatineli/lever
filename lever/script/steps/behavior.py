from typing import Tuple
from pathlib import Path
import numpy as np
from algorithm.time_series import SparseRec
from pypedream import Task, Input, getLogger
from log import res_filter_log, make_trial_log

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe")
Task.save_folder = proj_folder.joinpath("data", "interim")
Input.save_folder = proj_folder.joinpath("data")

def get_lever_shape(trial_log: SparseRec, peak_window: Tuple[float, float]) -> Tuple[float, float, float]:
    cutoff = int(round((trial_log.sample_rate * peak_window[0])))
    max_cutoff = int(round((trial_log.sample_rate * peak_window[1])))
    max_idx = trial_log.values[:, cutoff: max_cutoff].argmax(axis=1) + cutoff
    amplitude = np.quantile(trial_log.values[range(trial_log.shape[0]), max_idx]
                            - trial_log.values[:, :cutoff].mean(axis=1), 0.75)
    speed = np.array([np.diff(x[5: max(idx, cutoff * 2)]).max() for x, idx in zip(trial_log.values, max_idx)]).mean()
    reliability = np.corrcoef(trial_log.values[:, cutoff: max_cutoff])[np.triu_indices(trial_log.shape[0], 1)].mean()
    return amplitude, speed, reliability

def get_delay_hitrate(log: SparseRec) -> Tuple[float, float]:
    pass

def build_task_shape() -> Task:
    motionParams = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 0.9}
    task_trial_log = Task(make_trial_log, "2019-04-30T18:38", "trial-log", extra_args=(motionParams, ))
    task_behavior = Task(get_lever_shape, "2019-05-19T13:47", extra_args=((motionParams['pre_time'], 0.3), ))
    return task_behavior(task_trial_log(res_filter_log))

def main():
    import toml
    from multiprocessing import Pool, cpu_count
    mice = [dict(x) for x in toml.load(proj_folder.joinpath("data", "index", "index.toml"))["recordings"]]
    logger = getLogger("astrocyte", "exp-log.log")
    pool = Pool(max(1, cpu_count() - 2))
    params_dict = [(info['name'], logger) for info in mice]
    res_behavior = build_task_shape()
    result = pool.starmap(res_behavior.run, params_dict)
    return result
