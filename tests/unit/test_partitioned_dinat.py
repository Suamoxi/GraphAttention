"""M41.1: exact layerwise halo inference regression tests."""

import pytest
import torch

from graph_attention.geometry import build_attention_edge_indices, edge_relative_displacement
from graph_attention.geometry.partitioning import (
    coordinate_block_partition_ids,
    validate_partition_ids,
)
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.models.partitioned_dinat import (
    build_dinat_partition_plan,
    forward_partitioned_dinat,
)


def make_graph(nx=3, ny=3, nz=2):
    coords, links = [], []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                node = (x * ny + y) * nz + z
                coords.append((x, y, z))
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    a, b, c = x + dx, y + dy, z + dz
                    if a < nx and b < ny and c < nz:
                        other = (a * ny + b) * nz + c
                        links.extend(((node, other), (other, node)))
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(links, dtype=torch.long).T.contiguous(),
    )


def make_model(backend="scatter", self_mode="none"):
    torch.manual_seed(41)
    model = AlternatingDilatedGeometricDiT(
        in_channels=5,
        out_channels=5,
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        spatial_dim=3,
        conditioning_channels=1,
        condition_embed_dim=32,
        sparse_attention_backend=backend,
        include_self_attention=self_mode == "all",
        include_local_self_attention=self_mode == "local",
        coordinate_normalization="centered_bbox",
        dropout=0.0,
    ).eval()
    # Default adaLN-Zero initializes zero attention gates and output weights.
    # Make them nonzero so differences in halo attention are observable.
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.015)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.075)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.07)
    return model


@pytest.mark.parametrize("policy", ["one", "blocks", "interleaved"])
@pytest.mark.parametrize("self_mode", ["none", "local", "all"])
def test_exact_forward_and_all_layers(policy, self_mode):
    coords, edges = make_graph()
    n = coords.shape[0]
    topologies = build_attention_edge_indices(edges, n, {"dilated": "exact_two_hop"})
    model = make_model(self_mode=self_mode)
    if policy == "one":
        owner = torch.zeros(n, dtype=torch.long)
    elif policy == "blocks":
        owner = coordinate_block_partition_ids(coords, blocks=(2, 2, 2))
    else:
        owner = (torch.arange(n) * 3 % 4).long()
    plan = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edges,
        attention_edge_indices=topologies,
        partition_ids=owner,
    )

    torch.manual_seed(42)
    inputs = torch.randn(n, 5)
    condition = torch.tensor([[431.0]])
    snapshots = []
    hooks = [
        block.register_forward_hook(
            lambda _module, _args, result: snapshots.append(result.detach().clone())
        )
        for block in model.blocks
    ]
    try:
        with torch.no_grad():
            reference = model(
                inputs,
                edge_index=edges,
                coords=coords,
                attention_edge_indices=topologies,
                conditioning=condition,
            )
    finally:
        for hook in hooks:
            hook.remove()

    actual, layer_states = forward_partitioned_dinat(
        model,
        inputs,
        plan=plan,
        conditioning=condition,
        return_layer_states=True,
    )
    assert len(snapshots) == len(layer_states) == 4
    for old, new in zip(snapshots, layer_states, strict=True):
        torch.testing.assert_close(new, old, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-5)
    assert sum(part.owned_global.numel() for part in plan.partitions) == n
    if policy != "one":
        assert any(part.num_halo_nodes > 0 for part in plan.partitions)


def test_full_incoming_edges_and_validation():
    coords, edges = make_graph(4, 3, 2)
    n = coords.shape[0]
    topology = build_attention_edge_indices(edges, n, {"dilated": "exact_two_hop"})
    owner = coordinate_block_partition_ids(coords, blocks=(2, 1, 1))
    model = make_model()
    plan = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edges,
        attention_edge_indices=topology,
        partition_ids=owner,
    )
    for part in plan.partitions:
        for name, global_edges in (
            ("local", edges),
            ("dilated", topology["dilated"]),
        ):
            local_edges = getattr(part, name).edge_index
            reconstructed = part.read_global[local_edges]
            expected = global_edges[:, owner[global_edges[1]] == part.partition_id]
            assert set(map(tuple, reconstructed.T.tolist())) == set(map(tuple, expected.T.tolist()))
    with pytest.raises(ValueError, match="contiguous"):
        validate_partition_ids(torch.ones(n, dtype=torch.long), n)
    with pytest.raises(ValueError, match="blocks"):
        coordinate_block_partition_ids(coords, blocks=(2, 2))
    model.train()
    with pytest.raises(ValueError, match="inference-only"):
        forward_partitioned_dinat(
            model,
            torch.randn(n, 5),
            plan=plan,
            conditioning=torch.tensor([[0.0]]),
        )


def test_repeated_heun_evaluations_match():
    coords, edges = make_graph()
    n = len(coords)
    topology = build_attention_edge_indices(edges, n, {"dilated": "exact_two_hop"})
    model = make_model(self_mode="local")
    owners = coordinate_block_partition_ids(coords, blocks=(2, 2, 1))
    plan = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edges,
        attention_edge_indices=topology,
        partition_ids=owners,
    )
    torch.manual_seed(43)
    initial = torch.randn(n, 5)

    def integrate(use_partitions):
        state = initial.clone()
        dt = 1.0 / 3
        for step in range(3):

            def predict(state_at_time, t):
                condition = torch.tensor([[1000.0 * t]])
                if use_partitions:
                    return forward_partitioned_dinat(
                        model, state_at_time, plan=plan, conditioning=condition
                    )
                return model(
                    state_at_time,
                    edge_index=edges,
                    coords=coords,
                    attention_edge_indices=topology,
                    conditioning=condition,
                )

            t = step * dt
            k1 = predict(state, t)
            k2 = predict(state + dt * k1, t + dt)
            state = state + (dt / 2) * (k1 + k2)
        return state

    with torch.no_grad():
        reference = integrate(False)
        partitioned = integrate(True)
    torch.testing.assert_close(partitioned, reference, rtol=3e-5, atol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_torch_sparse_gpu_parity():
    coords, edges = make_graph()
    n = len(coords)
    topology = build_attention_edge_indices(edges, n, {"dilated": "exact_two_hop"})
    model = make_model(backend="torch_sparse").cuda()
    coords, edges = coords.cuda(), edges.cuda()
    topology = {name: value.cuda() for name, value in topology.items()}
    owners = coordinate_block_partition_ids(coords, blocks=(2, 2, 2))
    plan = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edges,
        attention_edge_indices=topology,
        partition_ids=owners,
    )
    torch.manual_seed(44)
    inputs = torch.randn(n, 5, device="cuda")
    condition = torch.tensor([[251.0]], device="cuda")
    with torch.inference_mode():
        reference = model(
            inputs,
            coords=coords,
            edge_index=edges,
            attention_edge_indices=topology,
            conditioning=condition,
        )
        partitioned = forward_partitioned_dinat(model, inputs, plan=plan, conditioning=condition)
    torch.testing.assert_close(partitioned, reference, atol=2e-4, rtol=2e-4)


def test_partitioning_is_invariant_to_mesh_node_numbering():
    coords, edges = make_graph(4, 3, 2)
    n = len(coords)
    torch.manual_seed(45)
    permutation = torch.randperm(n)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n)
    shuffled_coords = coords[permutation]
    shuffled_edges = inverse[edges]
    topology = build_attention_edge_indices(shuffled_edges, n, {"dilated": "exact_two_hop"})
    model = make_model()
    owners = coordinate_block_partition_ids(shuffled_coords, blocks=(2, 2, 2))
    plan = build_dinat_partition_plan(
        model,
        coords=shuffled_coords,
        edge_index=shuffled_edges,
        attention_edge_indices=topology,
        partition_ids=owners,
    )
    state = torch.randn(n, 5)
    condition = torch.tensor([[157.0]])
    with torch.no_grad():
        reference = model(
            state,
            coords=shuffled_coords,
            edge_index=shuffled_edges,
            attention_edge_indices=topology,
            conditioning=condition,
        )
        result = forward_partitioned_dinat(model, state, plan=plan, conditioning=condition)
    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("backend", ["scatter", "torch_sparse"])
@pytest.mark.parametrize("self_mode", ["none", "local", "all"])
def test_directed_empty_rows_and_inference_plan(backend, self_mode):
    # Asymmetric directed graph, existing loop, isolated node and empty CSR rows.
    # Symmetric mesh graphs alone cannot detect reversed source/target selection.
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [4.0, 3.0, 2.0]]
    )
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 2, 3]])
    topology = build_attention_edge_indices(edges, 5, {"dilated": "exact_two_hop"})
    model = make_model(backend, self_mode)
    owners = torch.tensor([0, 1, 0, 1, 2])
    with torch.inference_mode():
        plan = build_dinat_partition_plan(
            model,
            coords=coords.clone(),
            edge_index=edges.clone(),
            attention_edge_indices=topology,
            partition_ids=owners,
        )
    assert not plan.model_coords.is_inference()
    for part in plan.partitions:
        assert not part.owned_global.is_inference()
        assert not part.read_global.is_inference()
        assert not part.owned_local.is_inference()
        assert not part.local.edge_index.is_inference()
        assert not part.dilated.edge_index.is_inference()
        assert not part.local.displacement.is_inference()
        assert not part.dilated.displacement.is_inference()
        for name in ("local", "dilated"):
            attention = getattr(part, name)
            effective = model._attention_edges_with_optional_self(
                name,
                edges if name == "local" else topology["dilated"],
                num_nodes=5,
            )
            assert torch.equal(
                part.read_global[attention.edge_index],
                effective[:, owners[effective[1]] == part.partition_id],
            )
            if attention.sparse_topology is not None:
                csr = attention.sparse_topology
                assert not csr.crow_indices.is_inference()
                assert all(
                    not tensor.is_inference()
                    for tensor in (
                        csr.crow_indices, csr.col_indices, csr.row_indices,
                        csr.edge_order, csr.transpose_crow_indices,
                        csr.transpose_col_indices, csr.transpose_order,
                    )
                )
                assert csr.crow_indices.numel() == len(part.read_global) + 1
                assert int(csr.crow_indices[-1]) == attention.edge_index.shape[1]
                assert torch.equal(
                    csr.crow_indices.diff(),
                    torch.bincount(attention.edge_index[1], minlength=len(part.read_global)),
                )
    x, condition = torch.randn(5, 5), torch.tensor([[123.0]])
    with torch.inference_mode():
        reference = model(
            x,
            coords=coords,
            edge_index=edges,
            attention_edge_indices=topology,
            conditioning=condition,
        )
    # Also reuse inference-created plan outside inference_mode.
    result = forward_partitioned_dinat(model, x, plan=plan, conditioning=condition)
    torch.testing.assert_close(result, reference, atol=3e-5, rtol=3e-5)
    assert torch.isfinite(result).all()


def test_empty_spatial_bins_and_plan_dtype_guard():
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    owners = coordinate_block_partition_ids(coords, blocks=(4, 4, 4))
    assert owners.tolist() == [0, 1]
    edges = torch.empty((2, 0), dtype=torch.long)
    model = make_model()
    plan = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edges,
        attention_edge_indices={"dilated": edges},
        partition_ids=owners,
    )
    model.double()
    with pytest.raises(ValueError, match="dtype differs"):
        forward_partitioned_dinat(
            model,
            torch.zeros(2, 5, dtype=torch.float64),
            plan=plan,
            conditioning=torch.zeros(1, 1).double(),
        )


def test_actual_fifty_step_heun_sampler():
    from dataclasses import replace

    from scripts.benchmark_m41_partitioned import _errors, _PartitionedModel

    from graph_attention.data import SyntheticMeshDataset
    from graph_attention.tasks import FlowMatchingTask

    dataset = SyntheticMeshDataset(num_samples=1, spatial_dim=3, seed=41)
    task = FlowMatchingTask(state_fields=("rho", "momentum"), time_embedding_scale=1000.0)
    batch = task.pack_and_prepare([dataset[0]], dataset.field_catalog)
    batch = replace(
        batch,
        attention_edge_indices=build_attention_edge_indices(
            batch.edge_index, len(batch.inputs), {"dilated": "exact_two_hop"}
        ),
    )
    model = make_model()
    model.in_channels = model.out_channels = batch.inputs.shape[1]
    model.input_projection = torch.nn.Linear(model.in_channels, model.hidden_dim)
    model.final_layer.projection = torch.nn.Linear(model.hidden_dim, model.out_channels)
    plan = build_dinat_partition_plan(
        model,
        coords=batch.coords,
        edge_index=batch.edge_index,
        attention_edge_indices=batch.attention_edge_indices,
        partition_ids=torch.arange(len(batch.inputs)) % 3,
    )
    calls = []
    hook = model.register_forward_pre_hook(lambda _m, args: calls.append(args[0].detach().clone()))
    with torch.inference_mode():
        reference = task.sample_standardized(model, batch, steps=50, sampling_seed=123)
    hook.remove()
    adapter = _PartitionedModel(model, plan).eval()
    partition_calls = []
    hook = adapter.register_forward_pre_hook(
        lambda _m, args: partition_calls.append(args[0].detach().clone())
    )
    with torch.inference_mode():
        actual = task.sample_standardized(adapter, batch, steps=50, sampling_seed=123)
    hook.remove()
    assert len(calls) == len(partition_calls) == 100
    assert torch.equal(calls[0], partition_calls[0])
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
    assert _errors(actual, reference, atol=3e-5, rtol=3e-5)["passed"]
    assert not _errors(reference + 1, reference, atol=3e-5, rtol=3e-5)["passed"]


def test_global_geometry_and_model_parameters_are_unchanged():
    coords, edges = make_graph(4, 3, 2)
    coords = coords * torch.tensor([0.25, 2.0, 4.0]) + torch.tensor([10.0, -5.0, 3.0])
    n = len(coords)
    topology = build_attention_edge_indices(edges, n, {"dilated": "exact_two_hop"})
    model = make_model(self_mode="local")
    before = {key: value.clone() for key, value in model.state_dict().items()}
    owner = coordinate_block_partition_ids(coords, blocks=(2, 2, 1))
    plan = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edges,
        attention_edge_indices=topology,
        partition_ids=owner,
    )
    low = coords.amin(dim=0, keepdim=True)
    high = coords.amax(dim=0, keepdim=True)
    center = 0.5 * (low + high)
    span = (high - low).amax(dim=1, keepdim=True).clamp_min(
        model.coordinate_normalization_eps
    )
    assert torch.equal(plan.model_coords, (coords - center) / span)
    assert any(part.num_halo_nodes > 0 for part in plan.partitions)
    for part in plan.partitions:
        for attention in (part.local, part.dilated):
            global_edges = part.read_global[attention.edge_index]
            assert torch.equal(
                attention.displacement,
                edge_relative_displacement(coords, global_edges),
            )
    inputs = torch.randn(n, model.in_channels)
    condition = torch.tensor([[123.0]])
    with torch.no_grad():
        baseline = model(
            inputs,
            edge_index=edges,
            coords=coords,
            batch_index=torch.zeros(n, dtype=torch.long),
            attention_edge_indices=topology,
            conditioning=condition,
        )
        changed = forward_partitioned_dinat(
            model, inputs, plan=plan, conditioning=condition
        )
    torch.testing.assert_close(changed, baseline, atol=3e-5, rtol=3e-5)
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_sampler_adapter_rejects_reused_plan_on_different_graph():
    from types import SimpleNamespace

    from scripts.benchmark_m41_partitioned import _PartitionedModel

    coords, edges = make_graph()
    n = len(coords)
    topology = build_attention_edge_indices(edges, n, {"dilated": "exact_two_hop"})
    batch_index = torch.zeros(n, dtype=torch.long)
    model = make_model()
    plan = build_dinat_partition_plan(
        model,
        coords=coords,
        edge_index=edges,
        attention_edge_indices=topology,
        partition_ids=coordinate_block_partition_ids(coords, blocks=(2, 2, 1)),
    )
    original = SimpleNamespace(
        coords=coords, edge_index=edges,
        attention_edge_indices=topology, batch_index=batch_index,
    )
    adapter = _PartitionedModel(model, plan, reference_batch=original).eval()
    kwargs = dict(
        conditioning=torch.tensor([[17.0]]),
        coords=coords,
        edge_index=edges,
        batch_index=batch_index,
        attention_edge_indices=topology,
    )
    with torch.no_grad():
        prediction = adapter(torch.randn(n, model.in_channels), **kwargs)
    assert prediction.shape == (n, model.out_channels)
    with pytest.raises(ValueError, match="different edge_index"):
        adapter(torch.randn(n, model.in_channels), **{**kwargs, "edge_index": edges.clone()})
    with pytest.raises(ValueError, match="different dilated topology"):
        adapter(
            torch.randn(n, model.in_channels),
            **{**kwargs, "attention_edge_indices": {"dilated": topology["dilated"].clone()}},
        )
