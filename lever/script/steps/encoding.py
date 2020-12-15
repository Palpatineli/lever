##
from typing import Tuple, List
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
from algorithm.time_series import SparseRec
from algorithm.time_series.utils import take_segment
from algorithm.array import DataFrame
from pypedream import Task, get_result
from encoding_model import build_predictor, run_encoding, Predictors, Grouping, bspline_set
from lever.script.steps.decoder import res_align_xy
from lever.script.steps.behavior import res_behavior
from lever.script.steps.utils import read_index, read_group, group, group_nested, Case

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
Task.save_folder = proj_folder.joinpath("data", "interim")
mice = read_index(proj_folder)

def get_predictor(behavior: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
                  aligned_spike: Tuple[DataFrame, SparseRec], sample_rate: int,
                  spline: np.ndarray) -> Tuple[Predictors, np.ndarray, Grouping]:
    """Build Predictors from lever trajectory, trial parameters and spikes.
    Args:
        behavior: from behavior::get_behavior, amplitude, speed, delay, hit and reliability
            reliability is one float, hit is a bool array with k trials. while amplitude, speed and delay are all floats
            same length as number of hit trials.
        aligned_spike: (spike, lever_trajectory) from decoder::align_xy, spike has n * k * t with n neurons, k trials
            and t timepoints in a trial. lever trajectory has a value of k * t
    Returns:
        predictor_mat: a Predictor, with each of (event, period, trial, temporal) being 3D event and trial predictors,
            n * k * t with n features, k trials and t timepoints
        y: 3D neuronal activity, n * k * t with n neurons, k trials and t timepoints
        grouping: a Predictor of 1d grouping arrays (event, period, trial, temporal)
    """
    (amplitude, max_speed, delay, hit, _), (spike, trace) = behavior, aligned_spike
    trial_samples = int(round(sample_rate * 5.0))
    delay_sample = np.minimum(trial_samples - 1, np.rint(delay * sample_rate).astype(np.int))
    delay_period = np.vstack(list(np.broadcast(0, delay_sample))).T
    onset = np.rint(trace.timestamps * 5 / 256).astype(np.int)
    trajectory = take_segment(trace.values, onset, trial_samples)
    speed = take_segment(np.diff(trace.values), onset - 1, trial_samples)
    y = np.array([take_segment(neuron, onset, trial_samples) for neuron in spike.values])
    preds = Predictors((0, delay_sample), (delay_period,), (hit, delay), (trajectory, speed))
    grouping = Grouping(np.array([1, 2]), np.array([3]), np.array([4, 5]), np.array([6, 7]))
    preds, grouping = build_predictor(preds, grouping, hit, spline)
    return preds, y, grouping
task_predictor = Task(get_predictor, "2020-02-20T13:04", "encoding-predictor-minimal",
                      extra_args=(5, bspline_set(np.arange(7), 2)))
res_predictor = task_predictor([res_behavior, res_align_xy])

task_encoding = Task(run_encoding, "2020-02-24T09:29", "encoding-r2-minimal")
res_encoding = task_encoding(res_predictor)

from encoding_model.main import build_model
task_model = Task(build_model, "2019-06-27T21:18", "encoding-model")
res_model = task_model(res_predictor)
predictor_names = ["start", "reward", "isMoving", "hit", "delay", "trajectory", "speed", "all"]

##
def merge(result: List[Tuple[np.ndarray, np.ndarray]]):
    r2mean = np.array([np.mean(np.maximum(0, x[0]), axis=0) for x in result])
    groups = read_group(proj_folder, 'grouping')
    mean = pd.DataFrame(group(r2mean, mice, groups), columns=predictor_names + ["case_id", "session_id", "group"])
    mean.to_csv(proj_folder.joinpath("data", "analysis", "encoding_mean_minimal.csv"))
    r2s = [x[0] for x in result]
    merged = pd.DataFrame(group_nested(r2s, mice, groups), columns=predictor_names + ["case_id", "session_id", "group"])
    merged.to_csv(proj_folder.joinpath("data", "analysis", "encoding_minimal.csv"))

def check_predictor_size():
    predictors = get_result([x.name for x in read_index(proj_folder)], [res_predictor])[0]
    groups = pd.DataFrame([(x.id, x.session, group_str) for group_str, value
                           in read_group(proj_folder, "grouping").items() for x in value])
    groups.columns = ["id", "session", "group"]
    mice = pd.DataFrame([[x.id, x.session, x.name] for x in read_index(proj_folder)])
    mice.columns = ["id", "session", "name"]
    lookup = groups.join(mice.set_index(["id", "session"]), on=["id", "session"]).set_index("name").sort_index()
    res = {"wt": [], "dredd": [], "glt1": []}
    for mouse, predictor in zip(read_index(proj_folder), predictors):
        if mouse.name in lookup.index:
            res[lookup.loc[mouse.name, "group"]].append(predictor[1].shape[0: 2])
    res = {key: np.sum(value, axis=0) for key, value in res.items()}

if __name__ == '__main__':
    merge(get_result([x.name for x in mice], [res_encoding], "log-encoding")[0])
    # get_result([x.name for x in mice], [res_model], "log-encoding-model")[0]
