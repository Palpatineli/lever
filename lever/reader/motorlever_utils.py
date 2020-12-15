##
from typing import Union, Dict, List, Iterable, Tuple
from pathlib import Path
from struct import Struct
from dataclasses import dataclass
import numpy as np
import toml
from mplplot import Figure

formats = {'f64': 'd', 'f32': 'f', 'i64': 'q', 'u64': 'Q', 'i32': 'i', 'u32': 'I', 'i16': 'h', 'u16': 'H',
           'u8': 'B', 'i8': 'b'}

event_types = {1: "intertrial", 2: "violation", 3: "trial", 4: "reward", 5: "end", 6: "water_on",
               7: "wait_push", 8: "push", 9: "passive", 10: "pull"}

__all__ = ["unpack", "load_plain"]

def unpack(file_path: Union[Path, str]):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    result = {'spout': list(), 'lever': list(), 'event': dict()}
    for attr_type in ("design", "hardware"):
        result[attr_type] = toml.load(file_path.joinpath(attr_type + ".toml"))
    packet = Struct("<BIi")
    with file_path.joinpath("temp.raw").open('rb') as fp:
        for pack in packet.iter_unpack(fp.read()):
            if 1 == pack[0]:
                result['spout'].append(pack[1:])
            elif 2 == pack[0]:
                result['lever'].append(pack[1:])
            elif 3 == pack[0]:
                event_type = event_types[pack[2]]
                if event_type in result['event']:
                    result['event'][event_type].append(pack[1])
                else:
                    result['event'][event_type] = [pack[1]]
            else:
                raise ValueError(f"wrong packet! {pack}")
        result['lever'] = np.array(result['lever'], dtype=[('timestamp', '<I'), ('value', '<i')])
        result['spout'] = np.array(result['spout'], dtype=[('timestamp', '<I'), ('value', '<i')])
        result['event'] = {key: np.array(value, dtype='<I') for key, value in result['event'].items()}
    return result

def flatten_events(events: List[Tuple[int, str]], event_types: Iterable[str]) -> np.ndarray:
    flattened = [(timestamp / 1E6, x) for x in event_types for timestamp in events.get(x, [])]
    if len(flattened) == 0:
        return np.array([], dtype=[("timestamp", "f4"), ("event_type", "U10")])
    else:
        return np.array(sorted(flattened), dtype=[("timestamp", "f4"), ("event_type", "U10")])

def events2trials(trial_switches: np.ndarray, mode_switches: np.ndarray) -> np.ndarray:
    """Convert trial events.
    Args:
        trial_switches: structured ndarray, (trial_switches, (timestamp, event_type)) and timestamps in seconds
        mode_switches: structured ndarray, (mode_switches, (timestamp, mode_type)) and timestamps in seconds
    Returns:
        structured ndarray, (trial_number, (intertrial, trial, reward, mode)) and first 3 cols are timestamps in seconds
    """
    trials = trial_switches['event_type'] == "trial"
    res = np.zeros(np.sum(trials), dtype=[("intertrial", ">f4"), ("trial", ">f4"), ("reward", ">f4"), ("mode", "U10")])
    res["mode"] = "passive"
    res["trial"] = trial_switches[trials]["timestamp"]
    intertrial = trial_switches[trial_switches['event_type'] == "intertrial"]["timestamp"]
    res["intertrial"][1:] = intertrial[:trials.sum() - 1]
    rewards = trial_switches[trial_switches['event_type'] == "reward"]["timestamp"]
    rewards_idx = np.searchsorted(res["trial"], rewards)
    manual_idx = rewards_idx != np.searchsorted(res["intertrial"][1:], rewards) + 1
    res["reward"][rewards_idx[~manual_idx] - 1] = rewards[~manual_idx]
    mode_idx = np.searchsorted(mode_switches["timestamp"], res["trial"])
    res["mode"][mode_idx > 0] = mode_switches[mode_idx[mode_idx > 0] - 1]["event_type"]
    return res

def plot_timecourse(file_path: Union[Path, str]):
    log = unpack(file_path)
    mode_switches = flatten_events(log['event'], ('passive', 'wait_push', 'push', 'pull'))
    trial_switches = flatten_events(log['event'], ('trial', 'reward', 'intertrial'))
    trials = events2trials(trial_switches, mode_switches)
    move_mode = "passive"
    mode_color = {'passive': 'gray', 'wait_push': 'blue', 'push': 'red', 'pull': 'green'}
    with Figure() as axes:
        for intertrial, trial, reward, mode in trials:
            axes[0].plot((intertrial, trial), (0, 0), color=mode_color[move_mode])
            move_mode = mode
            axes[0].plot((trial, trial), (0, 1), color=mode_color[move_mode])
            if reward > 0:
                axes[0].plot((trial, reward), (1, 1), color=mode_color[move_mode])
                axes[0].plot((reward, reward), (1, 2), color=mode_color[move_mode])
                axes[0].plot((reward, reward + 5), (2, 2), color=mode_color[move_mode])
            else:
                axes[0].plot((trial, trial + 5), (1, 1), color=mode_color[move_mode])

@dataclass
class Performance(object):
    all: Tuple[int, int]
    passive: Tuple[int, int] = (0, 0)
    wait_push: Tuple[int, int] = (0, 0)
    push: Tuple[int, int] = (0, 0)
    pull: Tuple[int, int] = (0, 0)

    def table(self) -> str:
        res = list()
        for mode in ("all", "passive", "wait_push", "push", "pull"):
            hit, total = getattr(self, mode)
            res.append(fr"{mode}: {hit} / {total} ({hit / total * 100:2.0f})%")
        return "\n".join(res)

    def line(self) -> str:
        most = max((getattr(self, x)[0], x) for x in ("passive", "wait_push", "push", "pull"))[1]
        return f"hit: {self.all[0]}, {most}"

def log_info(file_path: Union[Path, str]) -> Performance:
    log = unpack(file_path)
    mode_switches = flatten_events(log['event'], ('passive', 'wait_push', 'push', 'pull'))
    trial_switches = flatten_events(log['event'], ('trial', 'reward', 'intertrial'))
    trials = events2trials(trial_switches, mode_switches)
    hit = np.count_nonzero(trials["reward"] != 0)
    perf = Performance((hit, len(trials)))
    for mode in ("passive", "wait_push", "push", "pull"):
        temp = trials['mode'] == mode
        hit = np.count_nonzero(trials[temp]["reward"] != 0)
        setattr(perf, mode, (hit, sum(temp)))
    return perf
##
def main():
    import re
    proj_folder = Path("~/Sync/project/2018-motorlever").expanduser()
    orig_folder = proj_folder.joinpath("data", "original")
    pattern = re.compile(r"^log-(?P<case_id>\d+)-(?P<year>\d{4})(?P<month>\d\d)"
                         r"(?P<day>\d\d)-session-(?P<session_id>\d+)-"
                         r".*2016-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.(?P<second>\d{4})\d+\+00:00$")
    cases = ("431", "423", "424")
    for case_id in cases:
        print(f"{case_id}")
        res = list()
        for log_file in orig_folder.joinpath("423", "log").iterdir():
            match = pattern.match(log_file.name)
            if not match:
                continue
            idx = match.groupdict()
            res.append(((idx["month"], idx["day"], idx["session_id"], idx["second"]), log_info(log_file)))
        res = sorted(res, key=lambda x: x[0])
        for item in res:
            print(f"{item[0][0]}/{item[0][1]}-{item[0][2]}-{item[0][3]}: {item[1].line()}")

if __name__ == '__main__':
    main()

##
