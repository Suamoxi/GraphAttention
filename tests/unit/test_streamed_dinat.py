"""M41.3: CPU-resident global states, partition-local attention, M37 parity."""

from __future__ import annotations

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
from graph_attention.models.streamed_dinat import (
    build_streamed_dinat_plan,
    forward_streamed_dinat,
)


def _mesh(nx=3, ny=3, nz=2):
    points = []
    edges = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                here = (i * ny + j) * nz + k
                points.append((float(i), float(j), float(k)))
                for a, b, c in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    if i + a < nx and j + b < ny and k + c < nz:
                        other = ((i + a) * ny + j + b) * nz + k + c
                        edges.extend(((here, other), (other, here)))
    return (
        torch.tensor(points, dtype=torch.float32),
        torch.tensor(edges, dtype=torch.long).T.contiguous(),
    )


def _model(self_mode="none", channels=5):
    torch.manual_seed(13041)
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
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.015)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.075)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.07)
    return model


def _parity(coords, edges, attention, owner, model, *, device="cpu"):
    # The topology/partition planner must NOT require a CUDA graph or model.
    model = model.to(device).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    plan = build_streamed_dinat_plan(
        model, coords=coords, edge_index=edges,
        attention_edge_indices=attention, partition_ids=owner,
    )
    assert plan.base.device.type == "cpu"
    assert plan.base.model_coords.device.type == "cpu"
    for part in plan.partitions:
        assert part.nodes.owned_global.device.type == "cpu"
        assert part.nodes.read_global.device.type == "cpu"
        for name in ("local", "dilated"):
            topology = getattr(part, name)
            assert topology.pattern.device.type == "cpu"
            assert topology.row_indices.device.type == "cpu"
            assert getattr(part.nodes, name).displacement.device.type == "cpu"

    reference_plan = build_owned_query_plan(
        model, coords=coords.to(device), edge_index=edges.to(device),
        attention_edge_indices={k: v.to(device) for k, v in attention.items()},
        partition_ids=owner.to(device),
    )

    torch.manual_seed(13042)
    features = torch.randn((len(coords), model.in_channels), device=device)
    condition = torch.tensor([[80.0]], dtype=torch.float32, device=device)
    original_states = []
    hooks = [
        block.register_forward_hook(
            lambda _model, _args, out: original_states.append(out.detach().cpu().clone())
        )
        for block in model.blocks
    ]
    try:
        with torch.inference_mode():
            original = model(
                features, coords=coords.to(device), edge_index=edges.to(device),
                attention_edge_indices={k: v.to(device) for k, v in attention.items()},
                conditioning=condition,
            )
    finally:
        for hook in hooks:
            hook.remove()
    with torch.inference_mode():
        reference, reference_layers = forward_owned_query_dinat(
            model, features, plan=reference_plan, conditioning=condition,
            return_layer_states=True,
        )
        actual, streamed_layers = forward_streamed_dinat(
            model, features, plan=plan, conditioning=condition,
            return_layer_states=True,
        )
    # Small partition GEMMs can use different FP32 reduction groupings;
    # the real GH200 checkpoint gate additionally uses atol=rtol=2e-4.
    tol = 2e-4 if device == "cuda" else 5e-5
    for expected, owned, streamed in zip(
        original_states, reference_layers, streamed_layers, strict=True
    ):
        torch.testing.assert_close(streamed, expected, atol=tol, rtol=tol)
        torch.testing.assert_close(streamed, owned.cpu(), atol=tol, rtol=tol)
    torch.testing.assert_close(actual, original, atol=tol, rtol=tol)
    torch.testing.assert_close(actual, reference, atol=tol, rtol=tol)
    assert all(torch.equal(model.state_dict()[name], tensor) for name, tensor in before.items())
    return model, plan


@pytest.mark.parametrize("self_mode", ["none", "local", "all"])
@pytest.mark.parametrize("blocks", [(1, 1, 1), (2, 2, 1), (2, 2, 2)])
def test_streamed_layers_match_m37_and_m41_2(self_mode, blocks):
    coords, edges = _mesh()
    attention = build_attention_edge_indices(
        edges, len(coords), {"dilated": "exact_two_hop"}
    )
    owners = coordinate_block_partition_ids(coords, blocks=blocks)
    _parity(coords, edges, attention, owners, _model(self_mode))


def test_nonspatial_interleaved_ownership_and_empty_incoming_rows():
    coords = torch.tensor(
        [[0., 0., 0.], [1., 1., 0.], [2., 0., 1.],
         [3., 1., 1.], [4., 2., 0.]]
    )
    edges = torch.tensor(
        [[0, 1, 2, 0], [1, 2, 2, 3]], dtype=torch.long
    )
    attention = build_attention_edge_indices(
        edges, len(coords), {"dilated": "exact_two_hop"}
    )
    _parity(
        coords, edges, attention, torch.tensor([0, 1, 0, 1, 2]),
        _model("local"),
    )


def test_empty_graph_edges_and_self_attention():
    coords = torch.tensor([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    edges = torch.empty((2, 0), dtype=torch.long)
    for self_mode in ("none", "local", "all"):
        _parity(
            coords, edges, {"dilated": edges.clone()},
            torch.arange(len(coords)), _model(self_mode),
        )


def test_plan_builder_requires_cpu_graph_and_does_not_mutate_model():
    coords, edges = _mesh()
    attention = build_attention_edge_indices(
        edges, len(coords), {"dilated": "exact_two_hop"}
    )
    owner = torch.arange(len(coords)) % 3
    model = _model()
    with pytest.raises(ValueError, match="CPU coordinates"):
        build_streamed_dinat_plan(
            model, coords=coords.to("meta"), edge_index=edges,
            attention_edge_indices=attention, partition_ids=owner,
        )
    plan = build_streamed_dinat_plan(
        model, coords=coords, edge_index=edges,
        attention_edge_indices=attention, partition_ids=owner,
    )
    original = torch.randn((len(coords), model.in_channels), dtype=torch.float32)
    original_copy = original.clone()
    with torch.inference_mode():
        result = forward_streamed_dinat(
            model, original, plan=plan, conditioning=torch.tensor([[40.0]])
        )
    assert result.shape == original.shape
    assert result.device.type == "cpu"
    assert torch.equal(original, original_copy)
    model.train()
    with pytest.raises(ValueError, match="inference-only"):
        forward_streamed_dinat(
            model, original, plan=plan, conditioning=torch.tensor([[40.0]])
        )


def test_inputs_can_reside_on_cpu_independently_of_model_device():
    coords, edges = _mesh()
    topology = build_attention_edge_indices(
        edges, len(coords), {"dilated": "exact_two_hop"}
    )
    model = _model()
    plan = build_streamed_dinat_plan(
        model, coords=coords, edge_index=edges,
        attention_edge_indices=topology,
        partition_ids=torch.arange(len(coords)) % 3,
    )
    with torch.inference_mode():
        result = forward_streamed_dinat(
            model, torch.randn(len(coords), model.in_channels),
            plan=plan, conditioning=torch.tensor([[110.0]]),
        )
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_cuda_parity_and_no_global_gpu_qkv():
    before = torch.backends.cuda.matmul.allow_tf32
    before_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        coords, edges = _mesh(4, 3, 2)
        topology = build_attention_edge_indices(
            edges, len(coords), {"dilated": "exact_two_hop"}
        )
        owners = coordinate_block_partition_ids(coords, blocks=(2, 2, 2))
        _parity(coords, edges, topology, owners, _model("local"), device="cuda")

        model = _model("local").cuda()
        plan = build_streamed_dinat_plan(
            model, coords=coords, edge_index=edges,
            attention_edge_indices=topology, partition_ids=owners,
            pin_memory=False,
        )
        qkv_batch_sizes = []
        hooks = [
            block.attention.qkv.register_forward_pre_hook(
                lambda _module, args: qkv_batch_sizes.append(args[0].shape[0])
            )
            for block in model.blocks
        ]
        try:
            with torch.inference_mode():
                forward_streamed_dinat(
                    model, torch.randn(len(coords), model.in_channels, device="cuda"),
                    plan=plan, conditioning=torch.tensor([[50.]], device="cuda"),
                )
        finally:
            for hook in hooks:
                hook.remove()
        assert qkv_batch_sizes == [
            len(p.nodes.read_global)
            for _layer in model.blocks
            for p in plan.partitions
        ]
        assert max(qkv_batch_sizes) <= len(coords)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before
        torch.backends.cudnn.allow_tf32 = before_cudnn


def test_actual_fifty_step_heun_task_reuses_cpu_streamed_plan():
    from graph_attention.data import SyntheticMeshDataset
    from graph_attention.tasks import FlowMatchingTask

    dataset = SyntheticMeshDataset(num_samples=1, spatial_dim=3, seed=41)
    task = FlowMatchingTask(
        state_fields=("rho", "momentum"), time_embedding_scale=1000.0
    )
    batch = task.pack_and_prepare([dataset[0]], dataset.field_catalog)
    topology = build_attention_edge_indices(
        batch.edge_index, len(batch.inputs), {"dilated": "exact_two_hop"}
    )
    batch = replace(batch, attention_edge_indices=topology)
    model = _model(channels=batch.inputs.shape[1])
    plan = build_streamed_dinat_plan(
        model, coords=batch.coords, edge_index=batch.edge_index,
        attention_edge_indices=topology,
        partition_ids=torch.arange(len(batch.inputs)) % 3,
    )

    class StreamedAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model
            self.calls = 0

        def forward(self, inputs, *, conditioning, **_ignored):
            self.calls += 1
            return forward_streamed_dinat(
                self.inner, inputs, plan=plan, conditioning=conditioning
            )

    adapter = StreamedAdapter().eval()
    original_noise = []
    hook = model.register_forward_pre_hook(
        lambda _module, args: original_noise.append(args[0].detach().clone())
        if not original_noise else None
    )
    with torch.inference_mode():
        original = task.sample_standardized(model, batch, steps=50, sampling_seed=97)
    hook.remove()

    streamed_noise = []
    hook = adapter.register_forward_pre_hook(
        lambda _module, args: streamed_noise.append(args[0].detach().clone())
        if not streamed_noise else None
    )
    with torch.inference_mode():
        generated = task.sample_standardized(
            adapter, batch, steps=50, sampling_seed=97
        )
    hook.remove()
    assert adapter.calls == 100
    assert torch.equal(original_noise[0], streamed_noise[0])
    torch.testing.assert_close(generated, original, atol=2e-4, rtol=2e-4)
