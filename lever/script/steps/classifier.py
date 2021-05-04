##
from typing import Tuple, List
from itertools import combinations
from pathlib import Path
from warnings import simplefilter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from fastdtw import fastdtw
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec
from lever.plot import get_threshold
from lever.script.steps import log, trial_neuron
from lever.decoding.cluster import accuracy
from pypedream import Task, get_result
simplefilter(action='ignore', category=FutureWarning)

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
Task.save_folder = proj_folder.joinpath("data", "interim")
mice: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "index.csv")).set_index(["id", "session"])  # type: ignore
grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "grouping.csv")).set_index(["id", "session"])  # type: ignore

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

def classifier_power(trial_neuron: DataFrame, cluster: np.ndarray) -> Tuple[float, float, float]:
    """
    Args:
        trial_neuron: neuron activity segmented into trials
    Returns:
        score_raw: prediction accuracy score from raw neuron acitivty
        score_corr: the same from interneuron correlation
    """
    return [accuracy(trial_neuron, cluster, transform=trans, repeats=100, kernel='rbf',  # type: ignore
                     validate=True, C=4, gamma=1E-2) for trans in ("none", "corr", "mean")]
task_classifier = Task(classifier_power, "2019-07-22T17:25", "classifier-validated")
res_classifier_power = task_classifier([trial_neuron.res_trial_neuron, res_cluster])

## Aggregate Analysis
def merge(result: List[np.ndarray]):
    merged = pd.DataFrame(np.vstack(np.transpose(x) for x in result), index=np.repeat(mice.index, [len(x[0]) for x in result]),
                          columns=["none", "mean", "corr"]).join(grouping, how="inner").reset_index()
    merged = pd.melt(merged, id_vars=["group", "id", "session"], var_name="type", value_name="precision")
    merged.to_csv(proj_folder.joinpath("data", "analysis", "classifier_power_validated.csv"), index=False)

def traces():
    from lever.script.steps import log
    from algorithm.utils import quantize
    index = 1
    cluster = get_result(mice.name.to_list()[index: index + 1], [res_cluster])
    trials = get_result(mice.name.to_list()[index: index + 1], [log.res_trial_log])
    result = list()
    for trial, y in zip(trials[0].values, quantize(cluster[0])):
        for idx, item in enumerate(trial):
            result.append((item, y, idx))
    df = pd.DataFrame(result, columns=('value', 'cluster', 'time'))
    df.to_csv(proj_folder.joinpath("data", "analysis", "cluster_traces.csv"))

def plot():
    import seaborn as sns
    df: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "analysis", "classifier_power.csv"))  # type: ignore
    mean_pred = df[df['type'] == 'corr'].groupby(["id", "group"]).mean()
    sns.scatterplot(x="group", y="precision", data=mean_pred)

def check_data_size():
    neurons = get_result(mice.name.to_list(), [trial_neuron.res_trial_neuron])[0]
    res = pd.DataFrame([x.shape[0: 2] for x in neurons], index=mice.index, columns=["x", "y"]).join(grouping, how="inner")
    print(res.groupby("group").aggregate(sum))

##
def main():
    result = get_result(mice.name.to_list(), [res_classifier_power])[0]
    merge(result)

if __name__ == '__main__':
    main()
##
