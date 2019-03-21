from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from matplotlib import Axes
from fastdtw import fastdtw
from noformat import File
from mplplot import tsplot
from .reader import load_mat
from .cluster import threshold_big_cluster

__all__ = ['TraceTemplate']

class TraceTemplate(object):
    """manages the template of typical lever push traces"""
    def __init__(self, expert_pushes: np.ndarray, alpha=0.05) -> None:
        self.traces = expert_pushes
        self.template = expert_pushes.mean(axis=0)
        distribution = np.sort([fastdtw(self.template, trace)[0] for trace in expert_pushes])
        self.threshold = distribution[int((1 - alpha) * len(distribution)) - 1]

    @classmethod
    def loads(cls, traces: List[np.ndarray], alpha: float = 0.05, threshold: float = 16000.) -> TraceTemplate:
        """Load a template from list of traces in already folded Record."""
        typical_trials = threshold_big_cluster(traces, threshold)[0]
        expert_pushes = traces[typical_trials]
        return cls(expert_pushes, alpha)

    @classmethod
    def load(cls, lever_file: File, motion_params: Dict[str, float], threshold: float = 16000.) -> TraceTemplate:
        """Load a template from lever record file, given motion parameters and cluster threshold"""
        lever = load_mat(lever_file['response'])
        lever.center_on("motion", **motion_params)
        lever.fold_trials()
        cluster_main = threshold_big_cluster(lever.values, threshold)[0]
        return cls(cluster_main)

    @classmethod
    def load_dual(cls, lever_file: File, motion_params: Dict[str, float], threshold: float = 16000.)\
            -> Tuple[TraceTemplate, TraceTemplate]:
        """Load a template from lever record file, given motion parameters and cluster threshold"""
        lever = load_mat(lever_file['response'])
        lever.center_on("motion", **motion_params)
        lever.fold_trials()
        cluster_main = threshold_big_cluster(lever.values, threshold)[0]
        cluster_rest = np.setdiff1d(cluster_main, range(lever.shape[1]))
        return cls(cluster_main), cls(cluster_rest)

    def filter(self, trace_list):
        distance = np.array([fastdtw(self.template, trace)[0] for trace in trace_list])
        return distance <= self.threshold

    def plot(self, ax: Axes) -> Axes:
        return tsplot(ax, self.traces.values, color='red', ci=0.32)
