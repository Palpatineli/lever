"""when a cell array is saved in matlab, scipy.io.loadmat loads it into numpy structured array.
This function converts such array into a dictionary tree, which is easier to read
"""
from typing import Union
import numpy as np

def convert(array) -> Union[np.ndarray, dict, None, list]:
    """The main conversion function"""
    if isinstance(array, np.ndarray):
        if not array.dtype.fields:
            if len(array) == 0:
                return None
            if len(array) == 1:
                return convert(array[0])
            elif array.dtype == np.dtype('O'):
                return [convert(item) for item in array]
            else:
                return array
        else:
            if len(array) == 0:
                return None
            result = dict()
            for field in array.dtype.fields.keys():
                result[field] = convert(array[field][0])
            return result
    else:
        return array
