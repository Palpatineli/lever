##
from typing import Tuple, Dict, List
from pathlib import Path
from warnings import simplefilter
from multiprocessing import cpu_count, Pool
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import linregress
import pandas as pd
from thundersvm import SVR
from entropy_estimators import mutual_info
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec
from algorithm.stats import split_time_series, scale_features  # noqa
from pypedream import Task, getLogger
from mplplot import Figure
from lever.script.steps.log import res_filter_log
from lever.script.steps.trial_neuron import res_spike
from lever.script.steps.utils import group, read_group, read_index

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
mice = read_index(proj_folder)
simplefilter(action='ignore', category=FutureWarning)
SVR_PARAMS = {"gamma": 1E-4, "C": 12, "epsilon": 1E-3, "cache_size": 1E3}

def align_XY(spike_sample_rate: Tuple[DataFrame, float], filterd_log: SparseRec) -> Tuple[DataFrame, SparseRec]:
    spike, sample_rate = spike_sample_rate
    resampled_trace = InterpolatedUnivariateSpline(filterd_log.axes[0], filterd_log.values[0])(spike['y'])
    y = filterd_log.create_like(scale_features(resampled_trace), [spike['y']])
    y.sample_rate = sample_rate
    spike_df = DataFrame(scale_features(spike['data'], axes=1), [spike['x'], spike['y']])
    return spike_df, y
task_align_xy = Task(align_XY, "2019-05-23T18:37", "align-xy")
res_align_xy = task_align_xy([res_spike, res_filter_log])

def neuron_info(spike_trajectory: Tuple[DataFrame, SparseRec], svr_params: Dict[str, float]) -> np.ndarray:
    """Give the prediction power of individual neurosn on push trajectory predicted in a rbf SVR."""
    spike, trajectory = spike_trajectory
    y = trajectory.values
    X = spike.values
    svr = SVR('rbf', **svr_params)
    y_hat_list = [svr.fit(n.reshape(-1, 1), y).predict(n.reshape(-1, 1)) for n in X]
    # y_real, y_hat_array = list(), list()
    # for X_tr, y_tr, X_te, y_te in split_time_series(X, y, 10):
    #     y_real.append(y_te)
    #     y_hat_array.append([svr.fit(n_tr.reshape(-1, 1), y_tr).predict(n_te.reshape(-1, 1))
    #                         for n_tr, n_te in zip(X_tr, X_te)])
    # y_te = np.hstack(y_real)
    # y_hat_list = [np.hstack(x) for x in zip(*y_hat_array)]
    single_powers = np.array([mutual_info(y_hat, y) for y_hat in y_hat_list])
    return single_powers
task_neuron_info = Task(neuron_info, "2019-06-16T16:35", "neuron-info", extra_args=(SVR_PARAMS,))
res_neuron_info = task_neuron_info(res_align_xy)

def order_slope(single_powers: np.ndarray) -> float:
    ordered_powers = np.sort(single_powers[single_powers > 0])
    return linregress(np.arange(len(ordered_powers)), np.log(ordered_powers))[0]
task_single_order = Task(order_slope, "2019-06-16T17:40", "single-order")
res_single_order = task_single_order(res_neuron_info)

def prediction(spike_trajectory: Tuple[DataFrame, SparseRec], single_powers: np.ndarray, svr_params: Dict[str, float],
               neuron_no: int = 20) -> np.ndarray:
    """Predict the trajectory from the top <neuron_no> informative neurons."""
    spike, trajectory = spike_trajectory
    y = trajectory.values
    X = spike.values[single_powers >= np.sort(single_powers)[-neuron_no], :]
    svr = SVR('rbf', **svr_params)
    # predicted = list()
    # count = 0
    predicted = svr.fit(X.T, y).predict(X.T)
    # for X_tr, y_tr, X_te, y_te in split_time_series(X, y, 10):
    #     count += 1
    #     predicted.append(svr.fit(X_tr.T, y_tr).predict(X_te.T))
    return np.hstack(predicted)
task_predict = Task(prediction, "2019-06-16T16:31", "predict-trajectory", extra_args=(SVR_PARAMS, 20))
res_predict = task_predict([res_align_xy, res_neuron_info])

def decode_power(spike_trajectory: Tuple[DataFrame, SparseRec], y_hat: np.ndarray) -> float:
    y = spike_trajectory[1].values[0: y_hat.shape[0]]
    return mutual_info(y, y_hat)
task_decode_power = Task(decode_power, "2019-05-23T18:46", "decode-power")
res_decode_power = task_decode_power([res_align_xy, res_predict])

##
def main():
    logger = getLogger("astrocyte", "log-decode.log")
    pool = Pool(max(1, cpu_count() - 5))
    params = [(item.name, logger) for item in mice]
    result = pool.starmap(res_decode_power.run, params)
    return result

def merge(result: List[np.ndarray]):
    grouping = read_group(proj_folder, 'grouping')
    merged = pd.DataFrame(group(result, mice, grouping), columns=("mutual_info", "case_id", "group"))
    merged.to_csv(proj_folder.joinpath("data", "analysis", "decoder_power.csv"))

def cno_merge(result: List[np.ndarray]):
    grouping = read_group(proj_folder, 'cno-schedule')
    merged = pd.DataFrame(group(result, mice, grouping), columns=("mutual_info", "case_id", "treat"))
    merged.to_csv(proj_folder.joinpath("data", "analysis", "decoder_cno.csv"))

def order_slope_merge():
    logger = getLogger("astrocyte", "log-decode.log")
    pool = Pool(max(1, cpu_count() - 2))
    params = [(item.name, logger) for item in mice]
    result = pool.starmap(res_single_order.run, params)
    grouping = read_group(proj_folder, 'grouping')
    merged = pd.DataFrame(group(result, mice, grouping), columns=("slope", "case_id", "group"))
    merged.to_csv(proj_folder.joinpath("data", "analysis", "single_power_slope.csv"))

def single_power_merge():
    logger = getLogger("astrocyte", "log-decode.log")
    pool = Pool(max(1, cpu_count() - 2))
    params = [(item.name, logger) for item in mice]
    result = pool.starmap(res_neuron_info.run, params)
    grouping = read_group(proj_folder, "grouping")
    lookup = {(value.id, value.session): key for key, values in grouping.items() for value in values}
    table = list()
    for mi_power, mouse in zip(result, mice):
        group_name = lookup.get((mouse.id, mouse.session), None)
        if group_name is not None:
            for idx, power in enumerate(sorted(mi_power, reverse=True)):
                table.append((power, mouse.name, group_name, idx))
    mi_table = pd.DataFrame(table, columns=("mi", "case_id", "group", "order"))
    mi_table.to_csv(proj_folder.joinpath("data", "analysis", "single_power.csv"))

if __name__ == '__main__':
    # cno_merge(main())
    # single_power_merge()
    # main()
    order_slope_merge()
