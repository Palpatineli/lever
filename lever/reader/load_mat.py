"""wrap around leverpush data"""
from typing import Union, Dict
import numpy as np
from scipy.io import loadmat
from algorithm.time_series.sparse_rec import SparseRec
from algorithm.array import DataFrame
from .object_array2dict import convert as cell2dict

def _calculate_full_trace(data_dict: dict) -> np.ndarray:
    raw_movement = data_dict['response']['mvmtdata'].ravel()
    calibration_factor = data_dict['params']['lev_cal']
    return raw_movement / calibration_factor

LEVER_CHOICE = {2: 'right', 5: 'timeout'}

def _convert_psychsr_lever(data_dict: dict) -> dict:
    """convert a psychsr rig recording to a data_dict"""
    lever_response = [LEVER_CHOICE[x] for x in data_dict['response']['choice']]
    config = data_dict['params']
    config['blank_time'] = config.pop('noMvmtTime')
    config['stim_time'] = config.pop('responseTime')
    result = {'config': config,
              'timestamps': data_dict['response']['samples_start'],
              'rewardstamps': data_dict['response']['samples_stop'],
              'sequence': {'lever_response': lever_response}}
    return result

def load_mat(file_name: Union[str, Dict[str, dict]]) -> SparseRec:
    if isinstance(file_name, str):
        data_dict = cell2dict(loadmat(file_name)['data'])
    else:
        data_dict = file_name['data']
    samples_rate: float = data_dict['card']['ai_fs']  # type: ignore
    stimulus = _convert_psychsr_lever(data_dict)  # type: ignore
    trace = _calculate_full_trace(data_dict).reshape(1, -1)  # type: ignore
    axes = [np.array(["right-lever"]), np.arange(trace.shape[1]) / samples_rate]
    return SparseRec(DataFrame(trace, axes), stimulus, samples_rate)
