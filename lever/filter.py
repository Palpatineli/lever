from typing import Tuple, TypeVar, Dict
from os.path import join, split
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import firwin, fftconvolve
from scipy.io import loadmat, savemat
from algorithm.time_series import Recording

__all__ = ["devibrate_trials"]

def devibrate(trace: np.ndarray, sample_rate: int, target_freq: int = 30, cut_off_buffer: int = 5) -> np.ndarray:
    """Filter continuous lever trace"""
    filter_size = sample_rate // 5 - (sample_rate // 5) % 2 + 1
    fir = firwin(filter_size, target_freq - cut_off_buffer, fs=sample_rate)
    return fftconvolve(trace, fir, mode='same')

def devibrate_trials(trials: np.ndarray, motion_onset: float,
                     target_freq: int = 30, sample_rate: int = 256,
                     cut_off_buffer: int = 5, power_range: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Try to filter out some strange vibration from lever push trials.
    And exclude the trials that have too much vibration that filtering doesn't work.
    Args:
        trials: 2-D matrix of float, each row is one trial, each column is one sample
        motion_onset: when aligning trials, how many seconds included before motion onset
        target_freq: frequency of the vibration noise
        sample_rate: sampling rate of the data
        cut_off_buffer: target_freq - cut_off_buffer is the cutoff frequency of the FIR filter
        power_range: for the exclusion criterion, the max power in the range of
            (target_freq - power_range, target_freq + power_range) is used.
    Returns:
        mask: 1-D boolean mask for good trials
        filtered: filtered trials
            filtered[mask, :] is what you (ideally) should use
    """
    filter_size = sample_rate // 5 - (sample_rate // 5) % 2 + 1
    fir = firwin(filter_size, target_freq - cut_off_buffer, fs=sample_rate)
    motion_end = int((motion_onset + 3.0 / target_freq) * sample_rate) + 1
    filtered = np.array([fftconvolve(a, fir, mode='same') for a in trials])
    x_axis_with_motion = fftfreq(trials.shape[1], 1 / sample_rate)
    freq_start = np.searchsorted(x_axis_with_motion[0: len(x_axis_with_motion) // 2], target_freq - power_range)
    filtered_freq_with_motion = np.abs(fft(filtered)[:, 1: freq_start]).mean(axis=1)
    x_axis_no_motion = fftfreq(trials.shape[1] - motion_end, 1 / sample_rate)
    freq_start, freq_end = np.searchsorted(
        x_axis_no_motion[0: len(x_axis_no_motion) // 2], [target_freq - power_range, target_freq + power_range + 1])
    absolute_power = np.max(np.abs(fft(filtered[:, motion_end:])[:, freq_start: freq_end]), axis=1)
    relative_power = absolute_power / filtered_freq_with_motion
    mask = (relative_power * absolute_power) < 0.4
    return mask, filtered

T = TypeVar("T", bound=Recording)
MotionParams = Dict[str, float]

def devibrate_rec(trials: T, params: MotionParams = None) -> T:
    if trials.trial_anchors is None:
        trials.center_on("motion", **params)
    if not trials.converted:
        trials.fold_trials()
    if trials.values.shape[0] == 1:
        values = trials.values[0]
        trials.axes = trials.axes[1:]
    else:
        values = trials.values
    mask, filtered = devibrate_trials(values, trials.pre_time, sample_rate=trials.sample_rate)
    trials.trial_anchors = trials.trial_anchors[mask]  # type: ignore
    trials.axes[0] = trials.axes[0][mask]
    trials.values = filtered[mask, :]
    return trials

def convert(file_path: str, motion_onset: float) -> None:
    trials = loadmat(file_path)
    mask, filtered = devibrate_trials(trials, motion_onset)
    savemat(join(split(file_path)[0], 'filtered.mat'), {'mask': mask, 'filtered': filtered})

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        convert(str(sys.argv[1]), float(sys.argv[2]))
    else:
        print("call with mat file path as 1st argument and motion_onset as 2nd argument.\n\t"
              "motion onset: when aligning trials, how many seconds included before motion onset")
