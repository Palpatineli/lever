from typing import Union
from pathlib import Path
import numpy as np
from algorithm.time_series import SparseRec
from algorithm.array import DataFrame
from .motorlever_utils import unpack, flatten_events, events2trials

__all__ = ["load_plain"]

def load_plain(file_path: Union[str, Path]) -> SparseRec:
    raw_log = unpack(file_path)
    mode_switches = flatten_events(raw_log['event'], ('passive', 'wait_push', 'push', 'pull'))
    trial_switches = flatten_events(raw_log['event'], ('trial', 'reward', 'intertrial'))
    trials = events2trials(trial_switches, mode_switches)
    sample_rate = 1000
    config = {**raw_log["design"], **raw_log["hardware"], "blank_time": raw_log["design"]["program"]["violation"] / 1E6,
              "stim_time": raw_log["design"]["program"]["trial"] / 1E6}
    stimulus = {'config': config, "timestamps": trials["trial"], "rewardstamps": trials["reward"],
                "sequence": {"trials": trials}}
    trace = DataFrame(32768 - raw_log['lever']['value'][np.newaxis, :],
                      [np.array(["right-lever"]), raw_log['lever']['timestamp']])
    return SparseRec(trace, stimulus, sample_rate)
