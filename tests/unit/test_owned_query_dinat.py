"""M41.2 owned-query DiNAT numerical equivalence and topology tests."""

from dataclasses import replace

import pytest
import torch

from graph_attention.geometry import build_attention_edge_indices
from graph_attention.geometry.partitioning import coordinate_block_partition_ids
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.models.owned_query_dinat import (
    build_owned_query_plan,
    forward_owned_query_dinat,
)
from graph_attention.models.partitioned_dinat import (
    build_dinat_partition_plan,
    forward_partitioned_dinat,
)


def _mesh(nx=3, ny=3, nz=2):
    points, edges = [], []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                here = (x * ny + y) * nz + z
                points.append((float(x), float(y), float(z)))
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    a, b, c = x + dx, y + dy, z + dz
                    if a < nx and b < ny and c < nz:
                        there = (a * ny + b) * nz + c
                        edges.extend(((here, there), (there, here)))
    return (
        torch.tensor(points, dtype=torch.float32),
        torch.tensor(edges, dtype=torch.long).T.contiguous(),
    )


def _model(channels=5, self_mode="none"):
    torch.manual_seed(4102)
    model = AlternatingDilatedGeometricDiT(
        in_channels=channels,
        out_channels=channels,
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        spatial_dim=3,
        conditioning_channels=1,
        condition_embed_dim=32,
        sparse_attention_backend="torch_sparse",
        include_self_attention=self_mode == "all",
        include_local_self_attention=self_mode == "local",
        coordinate_normalization="centered_bbox",
    ).eval()
    # Exercise actual attention and MLP instead of zero-initialized adaLN gates.
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.015)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.075)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.07)
    return model


def _compare_all_layers(model, coords, edges, topology, owner, *, device="cpu"):
    model = model.to(device)
    coords, edges, owner = coords.to(device), edges.to(device), owner.to(device)
    topology = {key: value.to(device) for key, value in topology.items()}
    reference_plan = build_dinat_partition_plan(
        model, coords=coords, edge_index=edges,
        attention_edge_indices=topology, partition_ids=owner,
    )
    optimized_plan = build_owned_query_plan(
        model, coords=coords, edge_index=edges,
        attention_edge_indices=topology, partition_ids=owner,
    )
    assert all(
        part.nodes.local.sparse_topology is None
        and part.nodes.dilated.sparse_topology is None
        for part in optimized_plan.partitions
    )
    assert not optimized_plan.partitions[0].local.pattern.crow_indices().is_inference()
    for optimized_part in optimized_plan.partitions:
        for name, original_edges in (
            ("local", edges),
            ("dilated", topology["dilated"]),
        ):
            original_edges = model._attention_edges_with_optional_self(
                name, original_edges, num_nodes=len(coords)
            )
            subgraph = getattr(optimized_part.nodes, name)
            global_edges = optimized_part.nodes.read_global[subgraph.edge_index]
            expected = original_edges[
                :, owner[original_edges[1]] == optimized_part.nodes.partition_id
            ]
            assert torch.equal(global_edges, expected)
            rect = getattr(optimized_part, name)
            assert rect.pattern.shape == (
                len(optimized_part.nodes.owned_global),
                len(optimized_part.nodes.read_global),
            )
            assert torch.equal(
                rect.pattern.crow_indices().diff(),
                torch.bincount(rect.row_indices, minlength=rect.num_queries),
            )
            assert int(rect.pattern.crow_indices()[-1]) == expected.shape[1]

    torch.manual_seed(1702)
    inputs = torch.randn(len(coords), model.in_channels, device=device)
    condition = torch.tensor([[463.0]], device=device)
    layer_refs = []
    hooks = [
        block.register_forward_hook(
            lambda _module, _args, result: layer_refs.append(result.detach().clone())
        )
        for block in model.blocks
    ]
    try:
        with torch.inference_mode():
            original = model(
                inputs, edge_index=edges, coords=coords,
                attention_edge_indices=topology, conditioning=condition,
            )
    finally:
        for hook in hooks:
            hook.remove()
    with torch.inference_mode():
        previous, previous_layers = forward_partitioned_dinat(
            model, inputs, plan=reference_plan, conditioning=condition,
            return_layer_states=True,
        )
        actual, layers = forward_owned_query_dinat(
            model, inputs, plan=optimized_plan, conditioning=condition,
            return_layer_states=True,
        )
    tol = 2e-4 if device == "cuda" else 3e-5
    for expected, full, owned in zip(layer_refs, previous_layers, layers, strict=True):
        torch.testing.assert_close(owned, expected, atol=tol, rtol=tol)
        torch.testing.assert_close(owned, full, atol=tol, rtol=tol)
    torch.testing.assert_close(actual, original, atol=tol, rtol=tol)
    torch.testing.assert_close(actual, previous, atol=tol, rtol=tol)


@pytest.mark.parametrize("self_mode", ["none", "local", "all"])
@pytest.mark.parametrize("ownership", ["one", "blocks", "interleaved"])
def test_optimized_matches_m37_and_m41_each_layer(ownership, self_mode):
    coords, edges = _mesh()
    topology = build_attention_edge_indices(edges, len(coords), {"dilated": "exact_two_hop"})
    if ownership == "one":
        owner = torch.zeros(len(coords), dtype=torch.long)
    elif ownership == "blocks":
        owner = coordinate_block_partition_ids(coords, blocks=(2, 2, 2))
    else:
        owner = torch.arange(len(coords)) % 3
    _compare_all_layers(_model(self_mode=self_mode), coords, edges, topology, owner)


@pytest.mark.parametrize("self_mode", ["none", "local", "all"])
def test_directed_empty_rows_and_existing_self_loop(self_mode):
    # Asymmetric graph with empty incoming rows, isolated node and an existing
    # self edge; catches swapped source/query indices and NaN softmax rows.
    coords = torch.tensor(
        [[0., 0., 0.], [1., 0., 1.], [2., 1., 0.], [3., 2., 1.], [4., 3., 2.]]
    )
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 2, 3]], dtype=torch.long)
    topology = build_attention_edge_indices(edges, len(coords), {"dilated": "exact_two_hop"})
    _compare_all_layers(
        _model(self_mode=self_mode), coords, edges, topology,
        torch.tensor([0, 1, 0, 1, 2]),
    )


def test_inference_created_plan_reusable_and_weights_unchanged():
    coords, edges = _mesh()
    topology = build_attention_edge_indices(edges, len(coords), {"dilated": "exact_two_hop"})
    model = _model()
    weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.inference_mode():
        plan = build_owned_query_plan(
            model, coords=coords, edge_index=edges,
            attention_edge_indices=topology, partition_ids=torch.arange(len(coords)) % 3,
        )
    for part in plan.partitions:
        for name in ("local", "dilated"):
            rect = getattr(part, name)
            assert not rect.pattern.is_inference()
            assert not rect.row_indices.is_inference()
            assert not rect.col_indices.is_inference()
            assert not rect.edge_order.is_inference()
    with torch.no_grad():
        out = forward_owned_query_dinat(
            model, torch.randn(len(coords), model.in_channels),
            plan=plan, conditioning=torch.tensor([[120.]]),
        )
    assert torch.isfinite(out).all()
    assert all(torch.equal(model.state_dict()[name], tensor) for name, tensor in weights.items())
    model.train()
    with pytest.raises(ValueError, match="inference-only"):
        forward_owned_query_dinat(
            model, torch.randn(len(coords), model.in_channels),
            plan=plan, conditioning=torch.tensor([[120.]]),
        )


def test_fifty_step_heun_sampler_identical_initial_noise():
    from graph_attention.data import SyntheticMeshDataset
    from graph_attention.tasks import FlowMatchingTask

    dataset = SyntheticMeshDataset(num_samples=1, spatial_dim=3, seed=41)
    task = FlowMatchingTask(state_fields=("rho", "momentum"), time_embedding_scale=1000.0)
    batch = task.pack_and_prepare([dataset[0]], dataset.field_catalog)
    topology = build_attention_edge_indices(
        batch.edge_index, len(batch.inputs), {"dilated": "exact_two_hop"}
    )
    batch = replace(batch, attention_edge_indices=topology)
    model = _model(channels=batch.inputs.shape[1])
    plan = build_owned_query_plan(
        model, coords=batch.coords, edge_index=batch.edge_index,
        attention_edge_indices=topology, partition_ids=torch.arange(len(batch.inputs)) % 3,
    )

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model
            self.calls = 0

        def forward(self, inputs, *, conditioning, **_ignored):
            self.calls += 1
            return forward_owned_query_dinat(
                self.inner, inputs, plan=plan, conditioning=conditioning
            )

    adapter = Adapter().eval()
    source = []
    hook = model.register_forward_pre_hook(
        lambda _module, args: source.append(args[0].detach().clone()) if not source else None
    )
    with torch.inference_mode():
        baseline = task.sample_standardized(model, batch, steps=50, sampling_seed=97)
    hook.remove()
    captured = []
    hook = adapter.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
        if not captured else None
    )
    with torch.inference_mode():
        optimized = task.sample_standardized(adapter, batch, steps=50, sampling_seed=97)
    hook.remove()
    assert adapter.calls == 100
    assert torch.equal(source[0], captured[0])
    torch.testing.assert_close(optimized, baseline, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_owned_query_cuda_sparse_parity():
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        coords, edges = _mesh(4, 3, 2)
        topology = build_attention_edge_indices(
            edges, len(coords), {"dilated": "exact_two_hop"}
        )
        owners = coordinate_block_partition_ids(coords, blocks=(2, 2, 2))
        _compare_all_layers(
            _model(self_mode="local"), coords, edges, topology, owners, device="cuda"
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn


def test_nodewise_layers_are_evaluated_once_not_per_partition():
    coords, edges = _mesh(4, 3, 2)
    topology = build_attention_edge_indices(
        edges, len(coords), {"dilated": "exact_two_hop"}
    )
    model = _model()
    plan = build_owned_query_plan(
        model, coords=coords, edge_index=edges,
        attention_edge_indices=topology,
        partition_ids=coordinate_block_partition_ids(coords, blocks=(2, 2, 2)),
    )
    assert len(plan.partitions) > 1
    counts = {name: [] for name in ("norm1", "qkv", "out_proj", "norm2", "mlp")}
    hooks = []
    try:
        for block in model.blocks:
            for name, module in (
                ("norm1", block.norm1),
                ("qkv", block.attention.qkv),
                ("out_proj", block.attention.out_proj),
                ("norm2", block.norm2),
                ("mlp", block.mlp),
            ):
                hooks.append(
                    module.register_forward_pre_hook(
                        lambda _module, args, key=name: counts[key].append(
                            len(args[0])
                        )
                    )
                )
        with torch.inference_mode():
            forward_owned_query_dinat(
                model,
                torch.randn(len(coords), model.in_channels),
                plan=plan,
                conditioning=torch.tensor([[50.0]]),
            )
    finally:
        for hook in hooks:
            hook.remove()
    assert all(
        calls == [len(coords)] * len(model.blocks)
        for calls in counts.values()
    )


def test_no_edges_matches_original_torch_sparse():
    coords = torch.tensor(
        [[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]]
    )
    edges = torch.empty((2, 0), dtype=torch.long)
    topology = {"dilated": edges.clone()}
    _compare_all_layers(
        _model(), coords, edges, topology, torch.arange(len(coords))
    )


def test_backend_and_fp32_contract_are_enforced():
    coords, edges = _mesh()
    topology = build_attention_edge_indices(
        edges, len(coords), {"dilated": "exact_two_hop"}
    )
    owners = torch.zeros(len(coords), dtype=torch.long)
    model = _model().double()
    double_coords = coords.double()
    plan = build_owned_query_plan(
        model, coords=double_coords, edge_index=edges,
        attention_edge_indices=topology, partition_ids=owners,
    )
    with pytest.raises(TypeError, match="requires float32"):
        forward_owned_query_dinat(
            model,
            torch.randn(len(coords), model.in_channels, dtype=torch.float64),
            plan=plan,
            conditioning=torch.tensor([[23.]], dtype=torch.float64),
        )
    model.sparse_attention_backend = "scatter"
    with pytest.raises(ValueError, match="requires torch_sparse"):
        build_owned_query_plan(
            model, coords=double_coords, edge_index=edges,
            attention_edge_indices=topology, partition_ids=owners,
        )
