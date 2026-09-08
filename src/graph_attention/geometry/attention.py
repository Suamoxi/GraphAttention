"""Named sparse attention topologies owned by the geometry layer."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .connectivity import exact_two_hop_edge_index


def build_attention_edge_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
    specifications: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build named model-independent sparse attention topologies.

    ``edge_index`` remains the canonical local mesh topology. ``specifications``
    requests additional named topologies by construction rule. Models consume the
    resulting names but do not construct or sparsify connectivity themselves.

    The initial M12 implementation intentionally supports only
    ``"exact_two_hop"``. Future reduced-A^2, random-jumper, or global-node rules
    belong here rather than in model code.
    """

    if specifications is None:
        return {}
    if not isinstance(specifications, Mapping):
        raise TypeError("attention topology specifications must be a mapping")

    result: dict[str, torch.Tensor] = {}
    for name, kind in specifications.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("attention topology names must be non-empty strings")
        if name in result:
            raise ValueError(f"duplicate attention topology name: {name}")
        if kind == "exact_two_hop":
            result[name] = exact_two_hop_edge_index(edge_index, num_nodes)
            continue
        raise ValueError(f"unsupported attention topology construction: {kind!r}")
    return result
