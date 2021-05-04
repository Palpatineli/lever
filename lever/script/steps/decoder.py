##
from typing import Tuple, Dict, List
from pathlib import Path
from warnings import simplefilter
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import linregress
import pandas as pd
from thundersvm import SVR
from entropy_estimators import mutual_info
from algorithm.array import DataFrame
from algorithm.time_series import SparseRec
from algorithm.time_series.utils import take_segment  # type: ignore
from algorithm.stats import scale_features
from algorithm.optimize import PieceLinear2
from pypedream import Task, get_result
from lever.script.steps.log import res_filter_log
from lever.script.steps.trial_neuron import res_spike

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
mice: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "index.csv"))  # type: ignore
grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", 'grouping.csv')).set_index(["id", "session"])  # type: ignore
simplefilter(action='ignore', category=FutureWarning)
SVR_PARAMS = {"gamma": 1E-4, "C": 12, "epsilon": 1E-3, "cache_size": 1E3}

def align_XY(spike_sample_rate: Tuple[DataFrame, int], filterd_log: SparseRec) -> Tuple[DataFrame, SparseRec]:
    """
    Returns:
        X: spikes scaled
        y: lever trajectory resampled to the sample rate of spikes
    """
    spike, sample_rate = spike_sample_rate
    resampled_trace = InterpolatedUnivariateSpline(filterd_log.axes[1], filterd_log.values[0])(spike['y'])
    y = filterd_log.create_like(scale_features(resampled_trace), [spike['y']])  # type: ignore
    y.sample_rate = sample_rate
    spike_df = DataFrame(scale_features(spike['data'], axes=1), [spike['x'], spike['y']])  # type: ignore
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

def order_slope2(single_powers: np.ndarray) -> PieceLinear2:
    ordered_powers = np.sort(single_powers[single_powers > 0])
    linear = PieceLinear2.fit(np.arange(len(ordered_powers)), np.log(ordered_powers), ([10, -8, -np.inf, -np.inf], [np.inf, np.inf, np.inf, 0]))
    return linear
task_single_order2 = Task(order_slope2, "2019-07-11T19:52", "single-order-piece2")
res_single_order2 = task_single_order2(res_neuron_info)

def prediction(spike_trajectory: Tuple[DataFrame, SparseRec], single_powers: np.ndarray, svr_params: Dict[str, float],
               neuron_no: int = 20) -> np.ndarray:
    """Predict the trajectory from the top <neuron_no> informative neurons."""
    spike, trajectory = spike_trajectory
    y = trajectory.values
    X = spike.values[single_powers >= np.sort(single_powers)[-min(len(single_powers), neuron_no)], :]
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

def splice_trial_xy(xy: Tuple[DataFrame, SparseRec], sample_rate: float):
    spike, trace = xy
    onset = np.rint(trace.timestamps * 5 / 256).astype(int)
    trial_samples = int(round(sample_rate * 5.0))
    neurons = np.array([take_segment(neuron, onset, trial_samples).ravel() for neuron in spike.values])
    trajectory = take_segment(trace.values, onset, trial_samples).ravel()
    timestamps = take_segment(trace.axes[0], onset, trial_samples).ravel()
    return spike.create_like(neurons, [spike.axes[0], timestamps]), trace.create_like(trajectory, [timestamps])
task_trial_xy = Task(splice_trial_xy, "2020-12-17T22:38", "splice_trial_xy", extra_args=(5,))
res_trial_xy = task_trial_xy([res_align_xy])

task_neuron_info_trial = Task(neuron_info, "2019-06-16T16:35", "neuron-info-trial", extra_args=(SVR_PARAMS,))
res_neuron_info_trial = task_neuron_info_trial(res_trial_xy)
task_predict_trial = Task(prediction, "2019-06-16T16:31", "predict-trajectory-trial", extra_args=(SVR_PARAMS, 20))
res_predict_trial = task_predict_trial([res_trial_xy, res_neuron_info_trial])
task_trial_decode_power = Task(decode_power, "2019-05-23T18:46", "decode-power-trial")
res_decode_power_trial = task_trial_decode_power([res_trial_xy, res_predict_trial])
##
def merge(result: List[np.ndarray]):
    grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "grouping.csv")).set_index(["id", "session"])  # type: ignore
    merged = pd.DataFrame(result, index=mice.set_index(["id", "session"]).index, columns=["mutual_info"]).join(grouping, how="inner")
    merged.to_csv(proj_folder.joinpath("data", "analysis", "decoder_power.csv"))
    grouping: pd.DataFrame = pd.read_csv(proj_folder.joinpath("data", "index", "cno-schedule.csv")).set_index(["id", "session"])  # type: ignore
    merged = pd.DataFrame(result, index=mice.set_index(["id", "session"]).index, columns=["mutual_info"]).join(grouping, how="inner")
    merged.to_csv(proj_folder.joinpath("data", "analysis", "decoder_cno.csv"))

def merge_trial(result: List[np.ndarray]):
    merged = pd.DataFrame(result, index=mice.set_index(["id", "session"]).index, columns=["mutual_info"]).join(grouping, how="inner")
    merged.to_csv(proj_folder.joinpath("data", "analysis", "decoder_power_trial.csv"))

def order_slope_merge(result: List[np.ndarray]):
    merged = pd.DataFrame(result, index=mice.set_index(["id", "session"]).index, columns=["slope"]).join(grouping, how="inner")
    merged.to_csv(proj_folder.joinpath("data", "analysis", "single_power_slope.csv"))

def order_slope2_merge(result: List[PieceLinear2]):
    merged = pd.DataFrame(result, index=mice.set_index(["id", "session"]).index, columns=["x0", "y0", "k0", "k1"]).join(grouping, how="inner")
    merged.to_csv(proj_folder.joinpath("data", "analysis", "single_power_slope2.csv"))

def single_power_merge(result: List[np.ndarray]):
    pd.DataFrame([x for y in result for x in y], columns=["mutual_info"],
                 index=np.repeat(mice.set_index(['id', 'session']).index, list(map(len, result))))\
        .join(grouping, how="inner")\
        .to_csv(proj_folder.joinpath("data", "analysis", "single_power.csv"))
##
if __name__ == '__main__':
    names = mice.name.to_list()
    # result = get_result(names, [res_neuron_info])[0]
    # single_power_merge(result)
    # order_slope_merge(get_result(names, [res_single_order])[0])
    # order_slope2_merge(get_result(names, [res_single_order2])[0])
    merge(get_result(names, [res_decode_power])[0])
    merge_trial(get_result(names, [res_decode_power_trial])[0])
##
