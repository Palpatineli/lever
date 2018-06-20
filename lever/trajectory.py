"""wrap around leverpush data"""
from enum import IntEnum
from typing import Tuple

import numpy as np
from scipy.signal import fftconvolve, gaussian
from algorithm.time_series.utils import take_segment

_SMOOTH_SIZE = 0.125

def apply_gaussian(x: np.ndarray, std=5) -> np.ndarray:
    kernel_size = std * 10 + 1
    kernel = gaussian(kernel_size, std)
    kernel /= kernel.sum()
    return fftconvolve(x, kernel, 'same')


class Choice(IntEnum):
    RIGHT_PUSH = 2
    TIMEOUT = 5


class LeverData(object):
    """wrap around lever data"""
    full_trace = None  # type: np.ndarray
    sample_rate = None  # type: float
    stimulus_stamp = None  # type: np.ndarray
    motion_stamp = None  # type: np.ndarray
    choice = None  # np.ndarray
    _hit_trials = None  # type: np.ndarray

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.full_trace = apply_gaussian(self.full_trace, int(round(_SMOOTH_SIZE * self.sample_rate)))

    def outcome(self, choice: Choice) -> np.ndarray:
        return np.equal(self.choice, int(choice))

    @property
    def hit_trials(self):
        if not self._hit_trials:
            hit_trials = self.outcome(Choice.RIGHT_PUSH)
            self._hit_trials = hit_trials if hit_trials[-1] else hit_trials[0: -1]
        return self._hit_trials

    @property
    def miss_trials(self):
        return np.logical_not(self.hit_trials)

    def trials(self, align_by: np.ndarray, cutoff: Tuple[float, float] = (-1.0, 2.0)) -> np.ndarray:
        """Extract trajectory surrounding time points
        Args:
            align_by: an array of time points surround which to extract
            cutoff: from and to for the surround, in seconds
        Returns:
            list of trajectory traces in rows
        """
        motion_onsets = np.subtract(align_by, 1)
        return take_segment(self.full_trace, motion_onsets, np.multiply(cutoff, self.sample_rate).astype(int))
