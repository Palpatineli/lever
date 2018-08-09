from typing import Iterator, Tuple
from os import scandir
from os.path import join, expanduser, splitext

from reader.image_j_measure import read as tp_read
from algorithm.time_series.sparse_rec import SparseRec
from algorithm.array import DataFrame

from lever.reader.mat import load_mat

DATA_FOLDER = expanduser('~/Dropbox/data/2016-leverpush')
DATA_PATH = join(DATA_FOLDER, 'mouse-{case_id}', 'fov-{fov_id}', '{data_type}')

def get_files(case_id: str, fov_id: int, data_type: str = 'lever-psychsr') -> Iterator[Tuple[int, str]]:
    """List days and files of mouse.
    Returns:
        [(day, full_path)]
    """
    for entry in scandir(DATA_PATH.format(case_id=case_id, fov_id=fov_id, data_type=data_type)):
        if entry.is_file() and entry.name.startswith('day-'):
            basename = splitext(entry.name)[0]
            day = int(basename[basename.find('-') + 1:])
            yield day, entry.path

def fetch_data(case_id: str, fov_id: int, sample_rate: float) -> Iterator[Tuple[int, SparseRec, DataFrame]]:
    lever_days, lever_cases = list(zip(*get_files(case_id, fov_id, 'lever-psychsr')))
    for day_id, case in get_files(case_id, fov_id, 'spike'):
        try:
            yield day_id, load_mat(lever_cases[lever_days.index(day_id)]), tp_read(case, sample_rate)
        except ValueError:
            pass

def fetch_lever(case_id: str, fov_id: int) -> Iterator[Tuple[int, SparseRec]]:
    for day_id, case in get_files(case_id, fov_id, 'lever-psychsr'):
        yield day_id, load_mat(case)
