"""Geometry ownership layer: how spatial entities are related."""

from .connectivity import hex_connectivity_to_edge_index, mesh_with_hex_edge_index

__all__ = ["hex_connectivity_to_edge_index", "mesh_with_hex_edge_index"]
