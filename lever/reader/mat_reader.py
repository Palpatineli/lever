"""wrap around leverpush data"""
import numpy as np
from scipy.io import loadmat
from algorithm.time_series import Recording, DataFrame
from reader.object_array2dict import convert as cell2dict

def _calculate_full_trace(data_dict: dict) -> np.ndarray:
    raw_movement = data_dict['mvmtdata'].ravel()
    calibration_factor = data_dict['params']['lev_cal']
    return raw_movement / calibration_factor

def _convert_psychsr_lever(data_dict: dict) -> dict:
    """convert a psychsr rig recording to a data_dict"""
    lever_response = [choice[x] for x in ]
    result = {'config': data_dict['params'],
              'timestamps': data_dict['response']['samples_start'],
              'sequence': {'lever_response': lever_response}}
    return result

def load_mat(file_name: str) -> Recording:
    data_dict = cell2dict(loadmat(file_name)['data'])
    samples_rate = data_dict['card']['ai_fs']
    stimulus = _convert_psychsr_lever(data_dict)
    trace = _calculate_full_trace(data_dict).reshape(1, -1)
    axes = [np.array(['lever trajectory']), np.arange(trace.shape[1]) / samples_rate]
    return Recording(DataFrame(trace, axes), stimulus, samples_rate)
