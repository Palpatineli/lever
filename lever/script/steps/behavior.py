##
from typing import Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
from algorithm.time_series import SparseRec
from pypedream import Task, Input, get_result
from lever.script.steps.log import res_filter_log, motion_params

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe")
Task.save_folder = proj_folder.joinpath("data", "interim")
Input.save_folder = proj_folder.joinpath("data")
mice: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "index.csv"))  # type: ignore

def _reliability(array: np.ndarray) -> float:
    """axis -2 is trial"""
    trial_no = array.shape[-2]
    coef = 2 / (trial_no * (trial_no - 1))
    return (np.corrcoef(array)[np.triu_indices(array.shape[0], 1)]).sum() * coef

def get_behavior(filtered_log: SparseRec, peak_window: Tuple[float, float])\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Get delay and hitrate, need both original log file and log with motion onset found.
    Also peak amplitude, top speed, and reliability (intertrial-correlation).
    Args:
        trial_log: folded trajectory log
        peak_window: when to find the initial push peak (ignore the later peak some aimals tend to do post trial)
    Returns:
        delay: in seconds
        hitrate: [0, 1]
    """
    filtered_log.center_on("motion", **motion_params)
    trial_log = filtered_log.fold_trials()
    trial_log.values = np.squeeze(trial_log.values, 0)
    trial_log.axes = trial_log.axes[1:]
    hit_rate = filtered_log.mask
    trial_onsets = np.asarray(filtered_log.stimulus['timestamps'], dtype=int)[filtered_log.mask]
    delay = (filtered_log.trial_anchors - trial_onsets) / filtered_log.sample_rate
    cutoff = int(round((trial_log.sample_rate * peak_window[0])))
    max_cutoff = int(round((trial_log.sample_rate * peak_window[1])))
    max_idx = trial_log.values[:, cutoff: max_cutoff].argmax(axis=1) + cutoff
    amplitude = trial_log.values[range(trial_log.shape[0]), max_idx] - trial_log.values[:, :cutoff].mean(axis=1)
    speed = np.array([np.diff(x[5: max(idx, cutoff * 2)]).max() for x, idx in zip(trial_log.values, max_idx)])
    reliability = _reliability(trial_log.values[:, cutoff: max_cutoff])
    return amplitude, speed, delay, hit_rate, reliability
peak_window = (motion_params['pre_time'], 0.3)
task_behavior = Task(get_behavior, "2019-06-21T13:02", "delay-hitrate", extra_args=(peak_window,))
res_behavior = task_behavior(res_filter_log)

##
def main():
    result = get_result(mice.name.to_list(), [res_behavior], "astrocyte_exp_log")[0]
    return result 

def merge_behavior(result: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]):
    grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "grouping.csv"))  # type: ignore
    mean_result = np.array([(np.mean(a), np.mean(b), np.mean(c), np.mean(d), e) for a, b, c, d, e in result])
    merged = pd.DataFrame(mean_result, index=mice.set_index(["id", "session"]).index, columns=("amplitude", "speed", "delay", "hit_rate", "reliability"))\
        .join(grouping.set_index(["id", "session"]), how='inner').sort_index()
    merged.to_csv(proj_folder.joinpath("data", "analysis", "behavior.csv"))

def check_length():
    behavior = get_result(mice.name.to_list(), [res_behavior], "check_length")[0]
    for (_, speed, delay, hit_rate, _), mouse in zip(behavior, mice.name):
        print(f"case: {mouse}, \tdelay: {len(delay)}, \thitlen: {len(hit_rate)}, \thit: {hit_rate.sum()},"
              f"\tspeed: {len(speed)}")

if __name__ == '__main__':
    behavior = main()
    merge_behavior(behavior)
##
