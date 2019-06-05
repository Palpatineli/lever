##
from typing import List, Dict, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import toml
from pypedream import FileObj, Task, getLogger, InputObj, Input
from aligner.align import Alignment
from aligner.roi_reader import Roi, read_roi_zip
from deconvolve.main import deconvolve

__all__ = ['res_spike', 'res_measure']

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe")
Task.save_folder = proj_folder.joinpath("data", "interim")
Input.save_folder = proj_folder.joinpath("data")
mice = [dict(x) for x in toml.load(proj_folder.joinpath("data", "index", "index.toml"))["recordings"]]

def _info_to_name(info: Dict[str, str]) -> str:
    return f"{info['id']}-{info['fov']}-{info['session']:02d}"

class AlignmentInput(InputObj):
    def time(self) -> float:
        tif_files = list(self.file_path.joinpath("original").glob(self.name + "*.tif"))
        if len(tif_files) == 0:
            return 0
        return sorted(tif_file.stat().st_mtime for tif_file in tif_files)[-1]

    def load(self):
        tif_files = list(self.file_path.joinpath("original").glob(self.name + "*.tif"))
        if len(tif_files) == 0:
            return 0
        tif_file = sorted((x.stat().st_mtime, x) for x in tif_files)[-1][1]
        return Alignment(tif_file)

def make_align(session: Alignment) -> Alignment:
    session.align()
    return session

input_tif = Input(AlignmentInput, "2015-01-01T00:00", "tif")

class AlignedmentCache(FileObj):
    def save(self, obj):
        obj.save(self.file_path)

    def load(self) -> Alignment:
        return Alignment.load(self.file_path)

    def time(self) -> float:
        displacement = self.file_path.joinpath("displacement.npy")
        return displacement.stat().st_mtime if displacement.exists() else 0

task_align = Task(make_align, "2018-04-30T15:12", "align", file_cacher=AlignedmentCache)
res_align = task_align(input_tif)

class RoiInput(InputObj):
    def time(self) -> float:
        zips = list(self.file_path.joinpath("interim", "align", self.name).glob("*.zip"))
        if len(zips) == 0:
            return 0
        return sorted(zip_file.stat().st_mtime for zip_file in zips)[-1]

    def load(self) -> float:
        zips = list(self.file_path.joinpath("interim", "align", self.name).glob("*.zip"))
        if len(zips) == 0:
            return 0
        return read_roi_zip(sorted((zip_file.stat().st_mtime, zip_file) for zip_file in zips)[-1][1])

input_roi = Input(RoiInput, "2015-01-01T00:00")

def get_measurement(alignment: Alignment, rois: List[Roi]) -> Tuple[Dict[str, np.ndarray], float]:
    return alignment.save_roi(rois), alignment.frame_rate

task_measure = Task(get_measurement, "2019-05-15T16:00", "measure")
res_measure = task_measure([res_align, input_roi])

DataFrame = Dict[str, np.ndarray]

def get_spike(x: Tuple[DataFrame, float]) -> Tuple[DataFrame, float]:
    measurement, frame_rate = x
    spikes = deconvolve(measurement['data'], frame_rate)[0]
    return ({'x': measurement['x'], 'y': measurement['y'], 'data': spikes}, frame_rate)

task_spike = Task(get_spike, "2019-04-30T14:33", "spike")
res_spike = task_spike(res_measure)

##
def main():
    logger = getLogger("astrocyte", "alignment.log")
    pool = Pool(max(1, cpu_count() - 2))
    params_dict = [(info['name'], logger) for info in mice]
    result = pool.starmap(res_spike.run, params_dict)
    return result

if __name__ == '__main__':
    main()
