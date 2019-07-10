import numpy as np
from mplplot import Figure
from mplplot.plots import heatmap, annotate_heatmap, Array
from pypedream import get_result
from lever.script.steps.encoding import res_encoding, mice, proj_folder

def r2imshow():
    r2 = get_result([x.name for x in mice], [res_encoding], "r2")[0]
    r2 = [x[0] for x in r2]
    fig_path = proj_folder.joinpath("report", "fig", "encoding")
    for single_r2, mouse in zip(r2, mice):
        with Figure(fig_path.joinpath(mouse.name + ".svg"), (2, 4)) as axes:
            ax = axes[0]
            heatmap(ax, Array(single_r2, [np.arange(single_r2.shape[0]), ['start', 'reward', 'delay', 'hit', 'amplitude', 'max speed', 'delay lenght', 'trajectory', 'speed']]))
            ax.imshow(single_r2)
