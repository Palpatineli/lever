from typing import Tuple, TypeVar, Dict, Union, Generator, List
import numpy as np

from noformat import File
from algorithm.array import DataFrame
from algorithm.time_series.sparse_rec import SparseRec
from algorithm.time_series import resample, utils, event
from algorithm.utils import is_list
from .filter import devibrate_trials
Data = TypeVar('Data', bound=DataFrame)
MotionParams = Dict[str, Union[int, float]]

from lever.reader import load_mat

def filter_empty_trials(data: Data) -> Data:
    sums = np.abs(np.nansum(data.values, (0, 2)))
    mask = ~((sums > 1e100) | (sums < 1e-100))
    return data[:, mask, :]

def filter_empty_trial_sets(data: np.ndarray, results: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """filter together neurons and response"""
    sums = np.abs(np.nansum(data, (0, 2)))
    eps = np.spacing(1)
    mask = ((sums > eps) | (sums < -eps)) & ~np.any(np.isnan(data), axis=(0, 2))
    return data[:, mask, :], np.compress(mask, results, 0)

def get_trials(data_file: File, motion_params: MotionParams) -> SparseRec:
    lever = load_mat(data_file['response'])
    lever.center_on("motion", **motion_params)
    lever.fold_trials()
    lever.values = np.squeeze(lever.values, 0)
    lever.axes = lever.axes[1:]
    return lever

def neuron_lever(lever: SparseRec, neuron: DataFrame, neuron_rate: float,
                 motion_params: MotionParams) -> Tuple[DataFrame, np.ndarray]:
    """Read lever and corresponding neuron data by trial. I reimplement [fold_by] here to
    acheive same length of the two segments.
    Args:
        data_file: file including spike and response(lever)
        motion_params: MotionParams, a dict but all params are optional
            may have [pre_time] and [post_time] in seconds, [quiet_var] and [event_thres] in float,
            [window_size] in sample_no
    Retursn:
        neurons: a DataFrame with folded neurons
        lever: a DataFrame with folded lever, resampled to neuron sample rate
    """
    params = {x: motion_params[x] for x in ("quiet_var", "window_size", "event_thres") if x in motion_params}
    event_onsets = event.find_response_onset(lever, **params)[0]
    lever_resample = resample(lever.values[0], lever.sample_rate, neuron_rate)
    anchor = np.rint((event_onsets / lever.sample_rate - motion_params['pre_time']) * neuron_rate).astype(np.int)
    duration = int(round((motion_params.get('pre_time', 0.1) + motion_params.get('post_time', 0.9)) * neuron_rate))
    lever_folded = utils.take_segment(lever_resample, anchor, duration)
    neuron_trials = np.stack([utils.take_segment(trace, anchor, duration) for trace in neuron.values])
    mask, filtered = devibrate_trials(lever_folded, motion_params['pre_time'])
    return neuron_trials[:, mask, :], lever_folded[mask, :]

def _fixed_sum_round(score, sums):
    temp_sum = score.astype(int).sum(axis=1)
    if np.any(sums > temp_sum):
        minor = score - score.astype(int)
        mask = np.argsort(np.argsort(minor, axis=1), axis=1)\
            > (score.shape[1] - 1 - (sums - temp_sum)).reshape(-1, 1)
        return score.astype(int) + mask
    return score.astype(int)

def iter_groups(x: Dict[str, List[Union[List[float], float]]]) -> Generator[Tuple[int, str, List[float]], None, None]:
    for idx, (group_str, group) in enumerate(x.items()):
        if is_list(group[0]):
            for sess in group:
                yield idx, group_str, sess
        else:
            yield idx, group_str, group
