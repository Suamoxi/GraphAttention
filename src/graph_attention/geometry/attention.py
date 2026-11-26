"""Named sparse attention topologies owned by the geometry layer."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .connectivity import exact_two_hop_edge_index, _exact_three_hop_from_two


def build_attention_edge_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
    specifications: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build named model-independent sparse attention topologies.

    ``edge_index`` remains the canonical local mesh topology. ``specifications``
    requests additional named topologies by construction rule. Models consume the
    resulting names but do not construct or sparsify connectivity themselves.

    Exact two- and three-hop neighborhoods are shortest-path shells.
    Reuse the two-hop shell when constructing the three-hop shell.
    """

    if specifications is None:
        return {}
    if not isinstance(specifications, Mapping):
        raise TypeError("attention topology specifications must be a mapping")

    result: dict[str, torch.Tensor] = {}
    two = three = None
    for name, kind in specifications.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("attention topology names must be non-empty strings")
        if name in result:
            raise ValueError(f"duplicate attention topology name: {name}")
        if kind in ("exact_two_hop", "exact_three_hop"):
            if two is None:
                two = exact_two_hop_edge_index(edge_index,num_nodes)
            if kind == "exact_two_hop":
                result[name] = two
            else:
                if three is None:
                    three = _exact_three_hop_from_two(edge_index,two,num_nodes)
                result[name] = three
            continue
        raise ValueError(f"unsupported attention topology construction: {kind!r}")
    return result
