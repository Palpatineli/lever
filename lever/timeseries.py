from typing import Tuple
import numpy as np
from plpnt.algorithm.array import DataFrame
from plptn.algorithm.time_series.main import take_segment
from .reader import LeverData, Choice

def psth(data: LeverData, cutoff: Tuple[float, float]=(-1.0, 2.0)) -> np.ndarray:
    motion_onsets = np.subtract(data.motion_stamp[data.hit_trials], 1)
    cutoff_idx = int(cutoff[0] * data.sample_rate), int(cutoff[1] * data.sample_rate)
    return take_segment(data.full_trace, motion_onsets, cutoff_idx)


def peri_motion(data: LeverData, cutoff: Tuple[float, float]=(-1.0, 2.0)) -> np.ndarray:
    correct_trials = data.trials(Choice.RIGHT_PUSH)
    motion_onsets = np.array(data.motion_stamp)[correct_trials] - 1
    return np.add(*np.meshgrid(motion_onsets, np.multiply(cutoff, data.sample_rate))).T


def peri_stim(data: LeverData, cutoff: Tuple[float, float]=(-1.0, 2.0), hit=True) -> np.ndarray:
    """convert trace in data to a list of peristimulus traces
    Args:
        data: the LeverData, with lever trace, stimulus start points and sample rate
        cutoff: start and end of desired traces relative to stim onset
        hit: whether pick hit trials or miss trials
    Return:
        2D array with each row been a peristimulus trace
    """
    correct_trials = data.trials(Choice.RIGHT_PUSH)
    if not correct_trials[-1]:
        correct_trials = correct_trials[:-1]
    trials = correct_trials if hit else np.logical_not(correct_trials)
    stim_onsets = data.stimulus_stamp[trials] - 1
    return np.add(*np.meshgrid(stim_onsets, np.multiply(cutoff, data.sample_rate))).T


def resample_points(data: LeverData, segments: np.ndarray, target: float) -> np.ndarray:
    return (np.round(segments * (target / data.sample_rate))).astype(int)
