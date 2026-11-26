from itertools import product

import pytest
import torch

from graph_attention.data.contracts import Mesh, Sample
from graph_attention.data.periodic_endpoints import (
    lift_periodic_endpoints, reduce_cartesian_endpoints, reduce_sample,
)
from graph_attention.geometry.connectivity import hex_connectivity_to_edge_index


def mesh_and_field(side=5):
    coords = torch.tensor(list(product(range(side), repeat=3)), dtype=torch.float64)
    local = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
             (0,0,1),(1,0,1),(1,1,1),(0,1,1)]
    cells = torch.tensor([
        [((i+x)*side+j+y)*side+k+z for x,y,z in local]
        for i,j,k in product(range(side-1),repeat=3)
    ])
    # Deliberately non-Cartesian exporter order.
    perm = torch.randperm(len(coords), generator=torch.Generator().manual_seed(42))
    inverse = torch.argsort(perm)
    coords = coords[perm]
    field = (coords.remainder(side-1)*torch.tensor([1.,7.,19.])).sum(1)
    return Mesh(coords,torch.empty((2,0),dtype=torch.long),
                mesh_id="test",cell_connectivity=inverse[cells]), field


def test_periodic_endpoint_roundtrip_and_induced_open_graph():
    mesh, field = mesh_and_field()
    reduction = reduce_cartesian_endpoints(mesh,side=5)
    reduced = reduce_sample(Sample("test",mesh,{"rho":field}),reduction)
    assert reduced.mesh.num_nodes == 64
    assert reduced.mesh.cell_connectivity.shape == (27,8)
    torch.testing.assert_close(lift_periodic_endpoints(reduced.fields["rho"][:,None],reduction),
                               field[:,None],rtol=0,atol=0)
    edge=hex_connectivity_to_edge_index(reduced.mesh.cell_connectivity,64)
    original_edge=hex_connectivity_to_edge_index(mesh.cell_connectivity,125)
    old_to_new=torch.full((125,),-1,dtype=torch.long)
    old_to_new[reduction.retained]=torch.arange(64)
    induced=old_to_new[original_edge]
    induced=induced[:,(induced>=0).all(0)]
    assert set(map(tuple,edge.T.tolist()))==set(map(tuple,induced.T.tolist()))
    assert edge.shape[1]==2*3*3*4**2
    distance=(reduced.mesh.coords[edge[0]]-reduced.mesh.coords[edge[1]]).abs().sum(1)
    assert torch.equal(distance,torch.ones_like(distance))


def test_nonidentical_endpoint_rejected():
    mesh, field=mesh_and_field()
    reduction=reduce_cartesian_endpoints(mesh,side=5)
    field[torch.nonzero((mesh.coords==4).any(1),as_tuple=True)[0][0]] += 1
    with pytest.raises(ValueError,match="nonidentical periodic endpoint"):
        reduce_sample(Sample("bad",mesh,{"rho":field}),reduction)


def test_noncartesian_mesh_rejected():
    mesh,_=mesh_and_field()
    mesh.coords[0,0] += .1
    with pytest.raises(ValueError,match="uniform Cartesian"):
        reduce_cartesian_endpoints(mesh,side=5)


# M49.6: qualify the quotient without using raw exporter numbering.
from dataclasses import replace
from graph_attention.data.avbp import AVBP_FIELD_CATALOG
from graph_attention.data.contracts import ReferenceScale, ReferenceScales
from graph_attention.geometry import build_attention_edge_indices
from graph_attention.geometry.relative import edge_relative_displacement
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.tasks import FlowMatchingTask
from graph_attention.tasks.flow_matching import _flow_problem, _model_velocity
from graph_attention.training.data_pipeline import GraphTaskCollator, task_batch_to_device
from scripts.train_generative import _forward_model


def periodic_fixture():
    mesh, field = mesh_and_field(side=9)
    reduction = reduce_cartesian_endpoints(mesh, side=9, policy="periodic_quotient")
    names = ("rho", "rhou", "rhov", "rhow", "rhoE")
    references = ReferenceScales(
        scales=tuple(ReferenceScale(name, value, "synthetic qualification",
                                    units=units, provenance="analytic test fixture",
                                    inference_available=True)
                     for name, value, units in (("rho_ref", 1., "kg/m^3"),
                                                ("U_ref", 2., "m/s"), ("L_ref", 4., "m"))),
        scheme="synthetic_periodic",
    )
    original = Sample("periodic-a", mesh,
                      {name: field + i + 1 for i, name in enumerate(names)},
                      reference_scales=references)
    sample = reduce_sample(original, reduction)
    task = FlowMatchingTask(state_fields=names, physical_nondimensionalization=True,
                            time_embedding_scale=1000.)
    collator = GraphTaskCollator(task, AVBP_FIELD_CATALOG, {
        "attention_edge_indices": {"dilated": "exact_two_hop",
                                   "dilated_3": "exact_three_hop"}
    })
    return original, reduction, sample, task, collator


def test_quotient_preserves_all_native_cells_and_periodic_shells():
    original, reduction, sample, _, collator = periodic_fixture()
    n = 8**3
    assert sample.mesh.num_nodes == n
    assert sample.mesh.cell_connectivity.shape == (n, 8)
    assert sample.mesh.metadata["periodic_wrap_edges"] is True
    assert sample.mesh.metadata["periodic_box_lengths"] == (8., 8., 8.)
    assert sample.metadata["periodic_endpoint_policy"] == "periodic_quotient"
    for name, values in sample.fields.items():
        assert torch.equal(lift_periodic_endpoints(values[:, None], reduction)[:, 0],
                           original.fields[name])
    native_edges = hex_connectivity_to_edge_index(original.mesh.cell_connectivity, 9**3)
    mapped = reduction.full_to_unique[native_edges]
    mapped = mapped[:, mapped[0] != mapped[1]]
    expected = torch.unique(mapped.T, dim=0).T.contiguous()
    batch = collator([sample])
    assert torch.equal(batch.edge_index, expected)
    assert batch.source.mesh_metadata[0]["periodic_length_nondimensional"] == 2.
    assert (batch.coords.amax(0) - batch.coords.amin(0)).tolist() == [1.75]*3
    for hop, degree, edges in (
        (1, 6, batch.edge_index),
        (2, 18, batch.attention_edge_indices["dilated"]),
        (3, 38, batch.attention_edge_indices["dilated_3"]),
    ):
        assert edges.shape == (2, n * degree)
        assert torch.equal(torch.bincount(edges[1], minlength=n), torch.full((n,), degree))
        delta = edge_relative_displacement(batch.coords, edges, periodic_length=2.)
        assert torch.equal((delta/.25).abs().sum(dim=1), torch.full((n*degree,),float(hop),dtype=delta.dtype))


def test_periodic_reduction_rejects_bad_native_cells_and_non_cubic_box():
    mesh, _ = mesh_and_field(side=9)
    # Duplicate one boundary cell, leaving the historical induced interior
    # count unchanged. Periodic qualification must still reject it.
    cells = mesh.cell_connectivity.clone()
    cells[-1] = cells[-2]
    with pytest.raises(ValueError, match="distinct elementary"):
        reduce_cartesian_endpoints(replace(mesh, cell_connectivity=cells), side=9,
                                   policy="periodic_quotient")
    coords = mesh.coords.clone()
    coords[:,0] *= 2
    with pytest.raises(ValueError, match="cubic box"):
        reduce_cartesian_endpoints(replace(mesh, coords=coords), side=9,
                                   policy="periodic_quotient")


@pytest.mark.parametrize("backend", ["scatter", "torch_sparse_triton"])
def test_periodic_relative_only_equivariance_training_and_sampling(backend):
    if backend == "torch_sparse_triton" and not torch.cuda.is_available():
        pytest.skip("Grace CUDA required")
    device = torch.device("cuda" if backend == "torch_sparse_triton" else "cpu")
    torch.manual_seed(42)
    _, _, sample, task, collator = periodic_fixture()
    single = task_batch_to_device(collator([sample]), device=device, dtype=torch.float32)
    n = single.inputs.shape[0]
    model = AlternatingDilatedGeometricDiT(
        in_channels=5, out_channels=5, hidden_dim=128, num_heads=4, num_layers=3,
        spatial_dim=3, mlp_ratio=2, conditioning_channels=1, condition_embed_dim=128,
        sparse_attention_backend=backend, use_absolute_coordinate_embedding=False,
        attention_hop_schedule=(1, 2, 3),
    ).to(device).eval()
    # DiT starts as the zero map. Activate each attention branch and the
    # output projection; the equivariance tests must be nontrivial.
    with torch.no_grad():
        model.final_layer.projection.weight.normal_(std=.1)
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.05)
            block.adaLN_modulation[-1].bias.normal_(std=.05)
    times = torch.tensor([.4], device=device)
    source = torch.randn_like(single.inputs)
    problem = _flow_problem(single, times, source, time_embedding_scale=1000.)
    x = torch.randn_like(problem.inputs)
    problem = replace(problem, inputs=x)
    with torch.no_grad():
        expected = _forward_model(model, problem)
        assert torch.isfinite(expected).all() and expected.abs().max() > 1e-5
        sampled = _model_velocity(model, single, x, times, time_embedding_scale=1000.)
        torch.testing.assert_close(sampled, expected, rtol=1e-4, atol=3e-5)
        # Translation of data on fixed torus coordinates (not a renumbering).
        ijk = (sample.mesh.coords - sample.mesh.coords.amin(0)).round().long()
        linear = (ijk[:,0]*8+ijk[:,1])*8+ijk[:,2]
        inverse = torch.argsort(linear)
        shifted = (ijk + torch.tensor([1,3,5])).remainder(8)
        perm = inverse[(shifted[:,0]*8+shifted[:,1])*8+shifted[:,2]].to(device)
        translated = _forward_model(model, replace(problem, inputs=x[perm]))
        torch.testing.assert_close(translated, expected[perm], rtol=1e-4, atol=3e-5)

        # Arbitrary consistent renumbering: preserve coordinates and edges.
        p = torch.randperm(n, device=device)
        inv = torch.argsort(p)
        reordered = replace(problem, inputs=x[p], coords=problem.coords[p],
                            source=replace(problem.source, edge_index=inv[problem.edge_index]),
                            attention_edge_indices={k: inv[e] for k,e in problem.attention_edge_indices.items()})
        torch.testing.assert_close(_forward_model(model, reordered)[inv], expected, rtol=1e-4, atol=3e-5)

        # Disconnected packed graphs must not exchange information.
        packed = task_batch_to_device(collator([sample, replace(sample, sample_id="periodic-b")]),
                                      device=device, dtype=torch.float32)
        packed_problem = _flow_problem(packed, times.repeat(2), source.repeat(2,1),
                                       time_embedding_scale=1000.)
        packed_problem = replace(packed_problem, inputs=torch.cat((x,x[perm])))
        torch.testing.assert_close(_forward_model(model,packed_problem),
                                   torch.cat((expected,expected[perm])),rtol=1e-4,atol=3e-5)
        torch.testing.assert_close(_model_velocity(model,packed,packed_problem.inputs,
                                                   times.repeat(2),time_embedding_scale=1000.),
                                   torch.cat((expected,expected[perm])),rtol=1e-4,atol=3e-5)
        # Check full Heun sampling, not just the forward call.
        generated = task.sample_standardized(model,single,steps=2,sampling_keys=("a",))
        generated_packed = task.sample_standardized(model,packed,steps=2,sampling_keys=("a","b"))
        torch.testing.assert_close(generated_packed[:n],generated,rtol=1e-4,atol=5e-5)
    model.train()
    output = _forward_model(model,replace(problem,inputs=x.detach().requires_grad_(True)))
    output.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in model.coordinate_projection.parameters())
    for block in model.blocks:
        assert any(p.grad is not None and torch.count_nonzero(p.grad)
                   for p in block.attention.geometry_mlp.parameters())
    model.use_absolute_coordinate_embedding = True
    with pytest.raises(ValueError, match="relative-only"):
        _forward_model(model, problem)
    with pytest.raises(ValueError, match="relative-only"):
        _model_velocity(model, single, x, times, time_embedding_scale=1000.)


def test_real_m49_periodic_mesh_when_requested():
    # This test is opt-in for the Calypso launcher; ordinary unit tests
    # have no dependency on an external dataset.
    import json
    import os
    from pathlib import Path
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    split_path = os.environ.get("M49_PERIODIC_PREFLIGHT_SPLIT")
    if split_path is None:
        pytest.skip("real HIT mesh qualification is a Grace preflight")
    manifest = json.loads(Path(split_path).read_text())
    assert len(manifest["train_ids"]) == 800 and len(manifest["validation_ids"]) == 180
    cfg = OmegaConf.load("configs/data/hit_3d_avbp.yaml")
    cfg.case_file = str(Path("cases/HIT_LES_FORCED.yaml").resolve())
    cfg.snapshot_pattern = "solut_hit_*.h5"
    cfg.periodic_endpoint_policy = "periodic_quotient"
    dataset = instantiate(cfg)
    index = {sid: i for i,sid in enumerate(dataset.sample_ids)}
    task = FlowMatchingTask(state_fields=tuple(cfg.field_names),physical_nondimensionalization=True)
    collator = GraphTaskCollator(task,dataset.field_catalog,{
        "attention_edge_indices": {"dilated":"exact_two_hop","dilated_3":"exact_three_hop"}
    })
    # No validation/test field access: first frozen training snapshot only.
    sample = dataset[index[manifest["train_ids"][0]]]
    batch = collator([sample])
    n = 32768
    assert batch.inputs.shape == (n,5)
    assert sample.mesh.cell_connectivity.shape == (n,8)
    length = batch.source.mesh_metadata[0]["periodic_length_nondimensional"]
    for hop,degree,edges in ((1,6,batch.edge_index),
            (2,18,batch.attention_edge_indices["dilated"]),
            (3,38,batch.attention_edge_indices["dilated_3"])):
        assert edges.shape == (2,n*degree)
        assert torch.equal(torch.bincount(edges[1],minlength=n),torch.full((n,),degree))
        delta = edge_relative_displacement(batch.coords.double(),edges,periodic_length=length)
        steps = delta/(length/32)
        rounded = steps.round()
        torch.testing.assert_close(steps,rounded,rtol=0.,atol=5e-4)
        assert torch.all(rounded.abs().sum(1) == hop)
    print(f"M49 periodic mesh qualified: N={n}, L*={length}, 6/18/38 exact-hop degrees")
