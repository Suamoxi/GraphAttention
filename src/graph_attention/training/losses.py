"""Backward-compatible training imports for task-neutral sample losses."""

from graph_attention.objectives import SampleLossAggregate, sample_reduced_mse

__all__ = ["SampleLossAggregate", "sample_reduced_mse"]
