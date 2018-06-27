"""algorithms for lever push data analysis"""
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
from algorithm.time_series.event import find_response_onset
from .utils import fetch_lever


def hit_rate_daily(case_id: str, fov_id: int) -> pd.Series:
    """hit rate a one animal across days"""
    day_ids = list()
    day_hit_rate = list()
    for day_id, lever_data in fetch_lever(case_id, fov_id):
        hit = lever_data.hit_trials
        trial_count = len(hit)
        if hit[-1] is False:
            trial_count -= 1
        day_hit_rate.append(np.count_nonzero(hit) / trial_count)
        day_ids.append(day_id)
    return pd.Series(day_hit_rate, index=day_ids, name='hit rate')


def amplitude_one_case(case_id: str, fov_id: int) -> pd.DataFrame:
    """push trajectory peak-valley after motion onset. one registry per trial
    with day_id tag, for one animal/fov"""
    day_ids = list()
    day_amplitude = list()
    for day_id, lever_data in fetch_lever(case_id, fov_id):
        extrema_neighbor = int(0.125 * lever_data.sample_rate)
        lever_data.set_trials(find_response_onset(lever_data)[0], 1.0, lever_data.stim_time)
        trials = lever_data.fold_trials()
        for push_trace in trials.values:
            arg_maximum = next(iter(argrelextrema(push_trace, np.greater_equal,
                                                  order=extrema_neighbor)))
            # arg_minimum = next(iter(argrelextrema(push_trace, np.less, order=extrema_neighbor)))
            maximum = push_trace[arg_maximum[0]] if len(arg_maximum) > 0 else push_trace.max()
            # minimum = push_trace[arg_minimum[0]] if len(arg_minimum) > 0 else push_trace.min()
            day_amplitude.append(maximum.mean())
            day_ids.append(day_id)
    return pd.DataFrame({'day_id': day_ids, 'values': day_amplitude})


def amplitude_normalized(case_id: str, fov_id: int) -> pd.DataFrame:
    data = amplitude_one_case(case_id, fov_id)
    data['values'] /= data['values'][data['day_id'] == 1].min()
    return data


def latency_one_case(case_id: str, fov_id: int) -> pd.DataFrame:
    day_ids, data = zip(*fetch_lever(case_id, fov_id))
    onsets = (find_response_onset(datum) for datum in data)
    day_latency = [(onset[0] - datum.timestamps[onset[1]]) / datum.sample_rate for onset, datum in zip(onsets, data)]
    day_ids = np.repeat(day_ids, [len(x) for x in day_latency])
    return pd.DataFrame({'day_id': day_ids, 'values': np.hstack(day_latency)})


def t1_one_case(case_id: str, fov_id: int) -> pd.DataFrame:
    """push rising time for each trial tagged by day_id for one fov"""
    day_ids = list()
    t1s = list()
    for day_id, lever_data in fetch_lever(case_id, fov_id):
        for push_trace in lever_data.trials(lever_data.motion_stamp, (0.0, 2.0)):
            t1s.append(push_trace.argmax() / lever_data.sample_rate)
            day_ids.append(day_id)
    return pd.DataFrame({'day_id': day_ids, 'values': t1s})


def reliability(trace_list: np.ndarray) -> float:
    length = len(trace_list)
    coef = 2 / (length * (length - 1))
    return (np.corrcoef(trace_list)[np.triu_indices(length)].sum() - length) * coef
