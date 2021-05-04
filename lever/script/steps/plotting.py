from warnings import filterwarnings
from pathlib import Path
from pypedream import draw_nx, to_nx
from mplplot import Figure
from lever.script.steps.classifier import res_classifier_power
from lever.script.steps.behavior import res_behavior
from lever.script.steps.decoder import res_decode_power, res_neuron_info
from lever.script.steps.encoding import res_encoding

filterwarnings("ignore")
proj_folder = Path.home().joinpath("Sync", "project", "2018-leverpush-chloe", "report", "fig")

def main():
    graph = to_nx([res_classifier_power, res_behavior, res_decode_power, res_neuron_info, res_encoding])
    with Figure(proj_folder.joinpath("flowchar.svg"), (9, 9), despine={'bottom': True, 'left': True}) as (ax, ):
        draw_nx(graph, ax, prog='fdp', args="-GK=0.9")

if __name__ == '__main__':
    main()
