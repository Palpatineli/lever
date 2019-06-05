##
from typing import Tuple, Dict
from pathlib import Path
from warnings import simplefilter
from multiprocessing import cpu_count, Pool
import numpy as np
import toml
from scipy.interpolate import InterpolatedUnivariateSpline
from thundersvm import SVR
from entropy_estimators import mutual_info
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec
from algorithm.stats import split_time_series, scale_features
from pypedream import Task, getLogger
from log import res_filter_log, res_spike

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
mice = [dict(x) for x in toml.load(proj_folder.joinpath("data", "index", "index.toml"))["recordings"]]
simplefilter(action='ignore', category=FutureWarning)
SVR_PARAMS = {"gamma": 3E-9, "C": 12, "epsilon": 1E-3, "cache_size": 1E3}

def align_XY(spike_sample_rate: Tuple[DataFrame, float], filterd_log: SparseRec) -> Tuple[DataFrame, SparseRec]:
    spike, sample_rate = spike_sample_rate
    resampled_trace = InterpolatedUnivariateSpline(filterd_log.axes[0], filterd_log.values[0])(spike['y'])
    y = filterd_log.create_like(scale_features(resampled_trace), [spike['y']])
    spike_df = DataFrame(scale_features(spike['data'], axes=1), [spike['x'], spike['y']])
    return spike_df, y
task_align_xy = Task(align_XY, "2019-05-23T18:37", "align-xy")
res_align_xy = task_align_xy([res_spike, res_filter_log])

def neuron_info(spike_trajectory: Tuple[DataFrame, SparseRec], svr_params: Dict[str, float]) -> np.ndarray:
    """Give the prediction power of individual neurosn on push trajectory predicted in a rbf SVR."""
    spike, trajectory = spike_trajectory
    y = trajectory.values
    X = spike.values
    svr = SVR('rbf', gamma=3E-9, C=12, epsilon=1E-3, cache_size=1E3)
    single_powers = list()
    for X_tr, y_tr, X_te, y_te in split_time_series(X, y, 10):
        single_powers.append([mutual_info(y_te, svr.fit(X_tr.T, y_tr).predict(X_te.T)) for neuron in X])
    single_powers = np.array(single_powers).mean(axis=0)
    return single_powers
task_neuron_info = Task(neuron_info, "2019-05-23T18:37", "neuron-info", extra_args=(SVR_PARAMS,))
res_neuron_info = task_neuron_info(res_align_xy)

def prediction(spike_trajectory: Tuple[DataFrame, SparseRec], single_powers: np.ndarray, svr_params: Dict[str, float],
               neuron_no: int = 20) -> np.ndarray:
    """Predict the trajectory from the top <neuron_no> informative neurons."""
    spike, trajectory = spike_trajectory
    y = trajectory.values
    X = spike.values[single_powers >= np.sort(single_powers)[-neuron_no], :]
    svr = SVR('rbf', **svr_params)
    predicted = list()
    count = 0
    for X_tr, y_tr, X_te, y_te in split_time_series(X, y, 10):
        count += 1
        predicted.append(svr.fit(X_tr.T, y_tr).predict(X_te.T))
    return np.hstack(predicted)
task_predict = Task(prediction, "2019-06-05T13:11", "predict-trajectory", extra_args=(SVR_PARAMS, 20))
res_predict = task_predict([res_align_xy, res_neuron_info])

def decode_power(spike_trajectory: Tuple[DataFrame, SparseRec], y_hat: np.ndarray) -> float:
    y = spike_trajectory[1].values[0: y_hat.shape[0]]
    return mutual_info(y, y_hat)
task_decode_power = Task(decode_power, "2019-05-23T18:46", "decode-power")
res_decode_power = task_decode_power([res_align_xy, res_predict])

def neuron_order():
    pass

##
def main():
    logger = getLogger("astrocyte", "log-decode.log")
    pool = Pool(max(1, cpu_count() - 2))
    params = [(item['name'], logger) for item in mice]
    result = pool.starmap(res_decode_power.run, params)
    return result

if __name__ == '__main__':
    main()
