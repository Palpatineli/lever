import numpy as np
import pandas as pd
from pypedream import get_result
from lever.script.steps.decoder import res_predict, res_align_xy
from lever.script.steps.decoder import mice, proj_folder
from mplplot import Figure

def example_curve():
    xy, predicts = get_result([x.name for x in mice[1: 2]], [res_align_xy, res_predict], "astrocyte")
    with Figure() as ax:
        ax = ax[0]
        ax.plot(xy[0][1].values, color='blue')
        ax.plot(predicts[0], color='orange')

def save_example():
    xy, predicts = get_result([x.name for x in mice[1: 2]], [res_align_xy, res_predict], "astrocyte")
    dataframe = list()
    for (idx, trace), predict in zip(enumerate(xy[0][1].values), predicts[0]):
        dataframe.append((idx / 5.0, trace, predict))
    df = pd.DataFrame(dataframe, columns=["time", "trajectory", "predicted"])
    df.to_csv(proj_folder.joinpath("data", "analysis", "decoder_example.csv"))
    with Figure() as ax:
        ax = ax[0]
        ax.plot(df["time"], df["trajectory"], color="blue")
        ax.plot(df["time"], df["predicted"], color="orange")

def example_validation(xy, predicts):
    xy, predicts = get_result([x.name for x in mice[1: 2]], [res_align_xy, res_predict], "astrocyte")
    with Figure(proj_folder.joinpath("report", "fig", "decoder_validation.svg"), (9, 6)) as ax:
        ax = ax[0]
        ax.plot(np.arange(1800), xy[0][1].values[0: 1800], color='blue')
        ax.plot(np.arange(1800, 2100), xy[0][1].values[1800: 2100] - 5.0, color='blue')
        ax.plot(np.arange(2100, 3000), xy[0][1].values[2100: 3000], color='blue')
        ax.plot(np.arange(1800, 2100), predicts[0][1800: 2100] - 10.0, color='orange')
        for idx, neuron in enumerate(xy[0][0].values[:20, :]):
            scaled = neuron * 2 / neuron.max()
            ax.plot(np.arange(1800), scaled[:1800] + 5.0 + idx, color='red')
            ax.plot(np.arange(1800, 2100), scaled[1800: 2100] + 10.0 + idx, color='red')
            ax.plot(np.arange(2100, 3000), scaled[2100: 3000] + 5.0 + idx, color='red')


