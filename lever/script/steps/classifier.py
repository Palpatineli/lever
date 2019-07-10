##
from typing import Tuple, List
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path
from warnings import simplefilter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from fastdtw import fastdtw
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec
from algorithm.utils import quantize
from lever.plot import get_threshold  # , plot_scatter
from lever.script.steps.utils import group, read_group, read_index
from lever.script.steps import log, trial_neuron
from lever.decoding.cluster import precision
from pypedream import Task, getLogger
simplefilter(action='ignore', category=FutureWarning)

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
Task.save_folder = proj_folder.joinpath("data", "interim")
mice = read_index(proj_folder)

def make_dtw_hierarchy(trial_log: SparseRec) -> np.ndarray:
    return linkage([fastdtw(x, y)[0] for x, y in combinations(trial_log.values, 2)])
task_dtw_hierarchy = Task(make_dtw_hierarchy, "2019-05-02T17:35", "dtw-hiearchy")
res_linkage = task_dtw_hierarchy(log.res_trial_log)

def make_threshold(linkage_mat: np.ndarray) -> float:
    return get_threshold(linkage_mat)
task_threshold = Task(make_threshold, "2019-05-02T17:46", "get-threshold")
res_threshold = task_threshold(res_linkage)

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
res_cluster = task_cluster([res_linkage, res_threshold])

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
res_classifier_power = task_classifier([trial_neuron.res_trial_neuron, res_cluster])

## Aggregate Analysis
def main():
    logger = getLogger("astrocyte", "log-classifier.log")
    pool = Pool(max(1, cpu_count() - 2))
    param_dict = [(item.name, logger) for item in mice]
    result = pool.starmap(res_classifier_power.run, param_dict)
    return result

def merge(result: List[np.ndarray]):
    grouping = read_group(proj_folder, 'grouping')
    result_types = ('none', 'mean', 'corr')
    type_res = {name: [x[idx] for x in result] for idx, name in enumerate(result_types)}
    all_types = [pd.DataFrame(group(type_res[x], mice, grouping)) for x in result_types]
    for df, name in zip(all_types, result_types):
        df['type'] = name
    merged = pd.concat(all_types, ignore_index=True)
    merged.columns = ["precision", "id", "group", "type"]
    merged.to_csv(proj_folder.joinpath("data", "analysis", "classifier_power.csv"))

def traces():
    from lever.script.steps import log
    from algorithm.utils import quantize
    logger = getLogger("astrocyte", "log-classifier.log")
    pool = Pool(max(1, cpu_count() - 2))
    param_dict = [(item.name, logger) for item in mice]
    index = 1
    cluster = pool.starmap(res_cluster.run, param_dict[index: index + 1])
    trials = pool.starmap(log.res_trial_log.run, param_dict[index: index + 1])
    result = list()
    for trial, y in zip(trials[0].values, quantize(cluster[0])):
        for idx, item in enumerate(trial):
            result.append((item, y, idx))
    df = pd.DataFrame(result, columns=('value', 'cluster', 'time'))
    df.to_csv(proj_folder.joinpath("data", "analysis", "cluster_traces.csv"))

def plot():
    import seaborn as sns
    df = pd.read_csv(proj_folder.joinpath("data", "analysis", "classifier_power.csv"))
    mean_pred = df[df['type'] == 'corr'].groupby("id").mean()
    sns.scatterplot("group", "precision", mean_pred)

def test():
    logger = getLogger("astrocyte", "log-classifier.log")
    pool = Pool(max(1, cpu_count() - 2))
    param_once = [(item.name, logger) for item in mice[40: 41]]
    result = pool.starmap(res_classifier_power.run, param_once)
    return result
##
if __name__ == '__main__':
    # main()
    merge(main())
    # print_tree()
    # draw_tree()
##
