"""Matched-initialization and execution-path gate for the 3-D ablation."""
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from graph_attention.geometry.connectivity import exact_two_hop_edge_index
from graph_attention.training.model_factory import instantiate_controlled_model


def test_relative_only_3d_preserves_weights_and_geometric_attention():
    # Four connected 3-D nodes; both one-hop and exact-two-hop edges exist.
    probe = SimpleNamespace(inputs=torch.randn(4,5), targets=torch.randn(4,5),
                            coords=torch.tensor([[0.,0.,0.],[1.,0.,0.],
                                                 [1.,1.,0.],[1.,1.,1.]]),
                            conditioning=torch.tensor([[.5]]))
    cfg = OmegaConf.create(dict(
        _target_="graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT",
        hidden_dim=16, num_heads=4, num_layers=2, mlp_ratio=2,
        condition_embed_dim=16, dropout=0., use_coord_mlp=True,
        coordinate_normalization="centered_bbox", sparse_attention_backend="scatter",
        use_sdpa=False, include_self_attention=False,
    ))
    baseline, base_meta = instantiate_controlled_model(cfg,probe,seed=42)
    cfg.use_absolute_coordinate_embedding = False
    relative, rel_meta = instantiate_controlled_model(cfg,probe,seed=42)
    assert base_meta["absolute_coordinate_embedding"] is True
    assert rel_meta["absolute_coordinate_embedding"] is False
    assert rel_meta["coordinate_conditioning"] == "relative_displacement_only"
    assert baseline.state_dict().keys() == relative.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value,relative.state_dict()[key],rtol=0,atol=0)

    local = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    dilated = exact_two_hop_edge_index(local,4)
    calls = {"absolute":0, "relative":0, "geometric":0}
    def count(name):
        def hook(_module, _inputs, _output):
            calls[name] += 1
        return hook
    baseline.coordinate_projection.register_forward_hook(count("absolute"))
    relative.coordinate_projection.register_forward_hook(count("relative"))
    for block in relative.blocks:
        block.attention.geometry_mlp.register_forward_hook(count("geometric"))
    kwargs = dict(edge_index=local, coords=probe.coords,
                  batch_index=torch.zeros(4,dtype=torch.long),
                  conditioning=probe.conditioning,
                  attention_edge_indices={"dilated":dilated})
    baseline(probe.inputs,**kwargs)
    result = relative(probe.inputs,**kwargs)
    assert result.shape == probe.targets.shape and torch.isfinite(result).all()
    assert calls == {"absolute":1, "relative":0, "geometric":2}
    result.square().sum().backward()
    assert all(p.grad is None for p in relative.coordinate_projection.parameters())
