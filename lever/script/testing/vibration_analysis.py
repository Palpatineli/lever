"""While building devibration filter for lever data, first testing out and
visualize said traces and filters.
"""
## setup
import numpy as np
from os.path import join, expanduser
from numpy.fft import fft, fftfreq
from scipy.signal import firwin, fftconvolve
import matplotlib.pyplot as plt
import toml

from noformat import File
from lever.utils import get_trials

motion_params = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 1.0}
project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
mice = toml.load(join(project_folder, "data/recording.toml"))
files = {x: [File(join(project_folder, "data", a["path"])) for a in y] for x, y in mice.items()}
##
FREQ = 256
fir = firwin(51, 25, fs=256)
motion_end = int((motion_params['pre_time'] + 1 / 10) * FREQ) + 1
trace = get_trials(files['dredd'][3], motion_params).values
filtered = np.array([fftconvolve(a, fir, mode='same') for a in trace])
filtered_freq = fft(filtered[:, motion_end:])
x_axis = fftfreq(trace.shape[1] - motion_end, 1 / FREQ)
start, end = np.searchsorted(x_axis[0: len(x_axis) // 2], [28, 33])
# mask = (np.max(np.abs(filtered_freq[:, start: end]), axis=1) / np.abs(fft(filtered)[:, 1: start]).mean(axis=1) < 0.5)\
#   & (np.max(np.abs(filtered_freq[:, start: end]), axis=1) < 0.75)
mask = ((np.max(np.abs(filtered_freq[:, start: end]), axis=1) / np.abs(fft(filtered)[:, 1: start]).mean(axis=1))
        * (np.max(np.abs(filtered_freq[:, start: end]), axis=1))) < 0.4
plt.plot(filtered[mask, :].T)
## bad trials
plt.plot(filtered[~mask, :].T)
plt.plot(filtered[~mask, :][filtered[~mask, 100] > 0.2, :].T)
##
plt.hist(np.max(np.abs(filtered_freq[:, start: end]), axis=1) / np.abs(filtered_freq[:, 0]).mean(), 20)
plt.hist(np.max(np.abs(filtered_freq[mask, start: end]), axis=1), 20)
##
[plt.plot(fftfreq(trace.shape[1] - motion_end, 1 / 256), a, 'r') for a in filtered_freq]
[plt.plot(fftfreq(trace.shape[1], 1 / 256), a, 'g') for a in fft(filtered)]
## examine single trial
trial_no = 84
plt.plot(filtered[trial_no, :])
np.abs(fft(filtered[trial_no, motion_end:])[start: end]).max() / np.abs(fft(filtered[trial_no, :]))[1: start].mean()
np.abs(fft(filtered[trial_no, motion_end:])[start: end]).max()\
    / np.abs(fft(filtered[trial_no, motion_end:]))[1: start].mean()
np.abs(fft(filtered[trial_no, motion_end:])[start: end]).mean()
##
plt.plot(filtered[mask, :][np.max(np.abs(filtered_freq[mask, start: end]), axis=1) > 1].T)
np.nonzero(filtered[mask, :][:, 60] < -0.2)
np.nonzero(mask)[0][19]
filtered[~mask, :][filtered[~mask, 100] > 0.2, :]
wrong_mask = np.flatnonzero(~mask)[filtered[~mask, 100] > 0.2]
##
trials = wrong_mask.copy()
plt.plot(filtered[trials, :].T)
power = np.abs(fft(filtered[trials, motion_end:])[:, start: end]).max(axis=1)
print('includes start: ', power / np.abs(fft(filtered[trials, :]))[:, 1: start].mean(axis=1))
print('excludes start: ', power / np.abs(fft(filtered[trials, motion_end:]))[:, 1: start].mean(axis=1))
print('absolute power: ', np.abs(fft(filtered[trials, motion_end:])[:, start: end]).mean(axis=1))
##
