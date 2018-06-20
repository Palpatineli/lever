import numpy as np
from fastdtw import fastdtw


class TraceTemplate(object):
    """manages the template of typical lever push traces"""
    def __init__(self, expert_pushes: np.ndarray, alpha=0.05) -> None:
        self.traces = expert_pushes
        self.template = expert_pushes.mean(axis=0)
        distribution = np.sort([fastdtw(self.template, trace)[0] for trace in expert_pushes])
        self.threshold = distribution[int((1 - alpha) * len(distribution)) - 1]

    def filter(self, trace_list):
        distance = np.array([fastdtw(self.template, trace)[0] for trace in trace_list])
        return distance <= self.threshold
