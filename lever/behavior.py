import numpy as np
from noformat import File
from algorithm.time_series.event import find_response_onset
from .filter import devibrate_trials
from .utils import load_mat, get_trials, MotionParams

def get_amp(data_file: File, params: MotionParams) -> float:
    """Get the max amplitude of lever push subtracted by the quiet baseline pre-trial."""
    mask, filtered = devibrate_trials(get_trials(data_file, params).values, params['pre_time'])
    return np.quantile(filtered[mask, 25: 64].max(axis=1) - filtered[mask, 0: 15].mean(axis=1), 0.75)

def get_speed(data_file: File, params: MotionParams) -> float:
    """Get the max positive speed of lever trajectory in the first 1/4 second of onset."""
    mask, filtered = devibrate_trials(get_trials(data_file, params).values, params['pre_time'])
    speed = np.diff(filtered[mask, 5:50], axis=1).max(axis=1)
    return np.mean(speed)

def get_delay(data_file: File, params: MotionParams) -> float:
    """Get delay in seconds. Delay as the time between EarlyCueTime and actual motion
    onset (passing threshold)."""
    lever = load_mat(data_file['response'])
    params = {x: y for x, y in params.items() if x in ('quiet_var', 'window_size', 'event_thres')}
    motion_onsets, stim_onsets, _, correct_trials, _ = find_response_onset(lever, **params)
    return np.mean((motion_onsets - stim_onsets[correct_trials]) / lever.sample_rate)

def get_hitrate(data_file: File, params: MotionParams) -> float:
    """Get hitrate (hit / trial_no) for one session. Hit is based reanalysis of trajectory and not
    whether reward was given."""
    lever = load_mat(data_file['response'])
    params = {x: y for x, y in params.items() if x in ('quiet_var', 'window_size', 'event_thres')}
    _, _, _, correct_trials, _ = find_response_onset(lever, **params)
    return correct_trials.mean()

def get_reliability(data_file: File, params: MotionParams) -> float:
    """Get inter-trial correlation in one session for successful trials."""
    mask, filtered = devibrate_trials(get_trials(data_file, params).values, params['pre_time'])
    return np.corrcoef(filtered[mask, 15:64])[np.triu_indices(mask.sum(), 1)].mean()

def get_t1(data_file: File, params: MotionParams) -> float:
    """Get push rising time for one trial."""
    lever = get_trials(data_file, params)
    mask, filtered = devibrate_trials(lever.values, params['pre_time'])
    return filtered[mask, 25: 64].argmax(axis=1) / lever.sample_rate - params['pre_time']
