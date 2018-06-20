from typing import Iterator, Tuple, List

import pandas as pd
from plptn.reader.image_j_measure import read as tp_read
from plptn.files import TwoPhotonFOV

from lever.reader import load_mat
from lever.trajectory import LeverData


def fetch_data(animal_id: str, fov_id: int, behavior_type: str="lever-psychsr",
               neuron_type: str='spike') -> Iterator[Tuple[int, LeverData, pd.DataFrame]]:
    cases = TwoPhotonFOV(animal_id, fov_id, [behavior_type, neuron_type])
    for day_id, case in cases:
        yield day_id, load_mat(case[behavior_type]), tp_read(case[neuron_type], 20)


def fetch_lever(animal_id: str, fov_id: int, behavior_type: str="lever-psychsr") -> Iterator[Tuple[int, LeverData]]:
    cases = TwoPhotonFOV(animal_id, fov_id, [behavior_type])
    for day_id, case in cases:
        yield day_id, load_mat(case[behavior_type])


def common_cells(animal_id: str, fov_id: int) -> List[int]:
    """find the common cells"""
    return list(set.intersection(*[set(tp_read(case['spike'], 20).columns)
                                   for _, case in TwoPhotonFOV(animal_id, fov_id)]))
