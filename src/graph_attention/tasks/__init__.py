"""Scientific learning objectives and task semantics."""

from .diffusion import DiffusionDenoisingTask
from .flow_matching import FlowMatchingTask
from .regression import NodeRegressionBatch, NodeRegressionTask

__all__ = [
    "DiffusionDenoisingTask",
    "FlowMatchingTask",
    "NodeRegressionBatch",
    "NodeRegressionTask",
]
