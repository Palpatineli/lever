##
from typing import Tuple, TypeVar, Dict, List
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path
from warnings import simplefilter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from fastdtw import fastdtw
import toml
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec
from algorithm.utils import quantize
from lever.plot import get_threshold  # , plot_scatter
from lever.script.steps.grouping import group
from lever.script.steps import align, log
from lever.decoding.cluster import precision
from pypedream import Task, getLogger, to_nx, draw_nx  # noqa
simplefilter(action='ignore', category=FutureWarning)

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
Task.save_folder = proj_folder.joinpath("data", "interim")
mice = [dict(x) for x in toml.load(proj_folder.joinpath("data", "index", "index.toml"))["recordings"]]
motionParams = {"quiet_var": 0.001, "window_size": 1000, "event_thres": 0.3, "pre_time": 0.1, "post_time": 0.9}

def make_dtw_hierarchy(trial_log: SparseRec) -> np.ndarray:
    return linkage([fastdtw(x, y)[0] for x, y in combinations(trial_log.values, 2)])

task_dtw_hierarchy = Task(make_dtw_hierarchy, "2019-05-02T17:35", "dtw-hiearchy")

def make_threshold(linkage_mat: np.ndarray) -> float:
    return get_threshold(linkage_mat)

task_threshold = Task(make_threshold, "2019-05-02T17:46", "get-threshold")

def make_cluster(linkage_mat: np.ndarray, threshold: float) -> np.ndarray:
    """
    Args:
        linkage_mat: N x 4 linkage mat from scipy.cluster.hierarchy.linkage
        threshold: where to cut threshold
    Returns:
        1d array of cluster ids
    """
    return fcluster(linkage_mat, threshold, 'distance')

task_cluster = Task(make_cluster, "2019-05-02T17:59", "cluster")

def _corrcoef(x):
    return np.corrcoef(x)[np.triu(x.shape[0], 1)]

def classifier_power(trial_neuron: DataFrame, cluster: np.ndarray) -> Tuple[float, float, float]:
    """
    Args:
        trial_neuron: neuron activity segmented into trials
    Returns:
        score_raw: prediction precision score from raw neuron acitivty
        score_corr: the same from interneuron correlation
    """
    y = quantize(cluster, groups=1)
    score_all = precision(trial_neuron, y, transform="none", repeats=100, kernel='rbf')
    score_corr = precision(trial_neuron, y, transform="corr", repeats=100, kernel='rbf')
    score_mean = precision(trial_neuron, y, transform="mean", repeats=100, kernel='rbf')
    return score_all, score_corr, score_mean

task_classifier = Task(classifier_power, "2019-05-17T14:33", "classifier")

##
def build_task():
    task_trial_log = Task(log.make_trial_log, "2019-05-17T13:10", "trial-log", extra_args=(motionParams, ))
    res_trial_log = task_trial_log(log.task_filtered_log(log.input_log))
    res_trial_neuron = log.task_trial_neuron([res_trial_log, align.res_spike])
    res_linkage = task_dtw_hierarchy(res_trial_log)
    res_threshold = task_threshold(res_linkage)
    res_cluster = task_cluster([res_linkage, res_threshold])
    res_classifier_power = task_classifier([res_trial_neuron, res_cluster])
    return res_classifier_power

def _temp(x):
    counts = np.unique(x, return_counts=True)[1]
    return counts.max() / counts.sum()

T = TypeVar("T", bound=list)

def wide_to_long(x: Dict[str, List[T]], label: str) -> List[Tuple[T, int, str, str]]:
    result = list()
    idx = 0
    for key, case in x.items():
        for values in case:
            for value in values:
                result.append((value, idx, key, label))
            idx += 1
    return result

def main():
    res_classifier_power = build_task()
    logger = getLogger("astrocyte", "log-classifier.log")
    pool = Pool(max(1, cpu_count() - 2))
    param_dict = [(item['name'], logger) for item in mice]
    result = pool.starmap(res_classifier_power.run, param_dict)
    return result

def merge(result):
    none_result = [x[0] for x in result]
    mean_result = [x[1] for x in result]
    corr_result = [x[2] for x in result]
    grouping_file = proj_folder.joinpath("data", "index", "grouping.toml")
    grouping = {k: list(x) for k, x in toml.loads(grouping_file.read_text()).items()}
    none_list = pd.DataFrame(wide_to_long(group(none_result, mice, grouping), "none"))
    mean_list = pd.DataFrame(wide_to_long(group(mean_result, mice, grouping), "mean"))
    corr_list = pd.DataFrame(wide_to_long(group(corr_result, mice, grouping), "corr"))
    merged = pd.concat([none_list, mean_list, corr_list], ignore_index=True)
    merged.columns = ["precision", "id", "group", "type"]
    merged.to_pickle(proj_folder.joinpath("data", "analysis", "classifier_power.bz2"), compression="bz2")
    merged.to_csv(proj_folder.joinpath("data", "analysis", "classifier_power.csv"))
    print('done')

def plot():
    import seaborn as sns
    df = pd.read_csv(proj_folder.joinpath("data", "analysis", "classifier_power.csv"))
    mean_pred = df[df['type'] == 'corr'].groupby("id").mean()
    sns.scatterplot("group", "precision", mean_pred)

def test():
    res_classifier_power = build_task()
    logger = getLogger("astrocyte", "log-classifier.log")
    pool = Pool(max(1, cpu_count() - 2))
    param_once = [(item['name'], logger, {"exp-log": item, "align": item}) for item in mice[40: 41]]
    result = pool.starmap(res_classifier_power.run, param_once)
    return result
##
if __name__ == '__main__':
    # main()
    merge(main())
    # print_tree()
    # draw_tree()
##
