from typing import Union
import json
from algorithm.time_series import SparseRec

def load_json(data_dict: Union[str, dict]) -> SparseRec:
    if isinstance(data_dict, str):
        with open(data_dict, 'r') as fp:
            data_dict = json.load(fp)
    return data_dict
