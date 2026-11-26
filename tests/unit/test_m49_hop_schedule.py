"""Full-mesh hop schedule and matched initialization."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from graph_attention.geometry import build_attention_edge_indices
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.training.model_factory import instantiate_controlled_model


def make_model(schedule=(1,2,3),layers=6,checkpointing=False):
    return AlternatingDilatedGeometricDiT(in_channels=3,out_channels=3,hidden_dim=8,
        num_heads=2,num_layers=layers,spatial_dim=3,mlp_ratio=2,conditioning_channels=1,
        condition_embed_dim=8,sparse_attention_backend="scatter",activation_checkpointing=checkpointing,
        attention_hop_schedule=schedule)


def kwargs_for_chain(n=6):
    edges=torch.tensor([(i,j) for i in range(n) for j in (i-1,i+1) if 0<=j<n]).T.contiguous()
    topology=build_attention_edge_indices(edges,n,{"dilated":"exact_two_hop","dilated_3":"exact_three_hop"})
    return dict(coords=torch.arange(n,dtype=torch.float32)[:,None].expand(n,3).contiguous(),
                edge_index=edges,attention_edge_indices=topology,
                batch_index=torch.zeros(n,dtype=torch.long),conditioning=torch.tensor([[.5]]))


@pytest.mark.parametrize("checkpointing",[False,True])
def test_schedule_and_backward(checkpointing):
    model=make_model(checkpointing=checkpointing)
    kwargs=kwargs_for_chain(7)
    calls=[]
    for block in model.blocks:
        block.attention.register_forward_pre_hook(
            lambda _mod,_args,kw:calls.append(kw["edge_index"].clone()),with_kwargs=True)
    x=torch.randn(7,3,requires_grad=True)
    with torch.no_grad():
        model.final_layer.projection.weight.normal_()
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.05)
    result=model(x,**kwargs)
    expected=[kwargs["edge_index"],kwargs["attention_edge_indices"]["dilated"],
              kwargs["attention_edge_indices"]["dilated_3"]]*2
    assert len(calls)==6
    for got,want in zip(calls,expected,strict=True):
        assert torch.equal(got,want)
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0
               for p in model.blocks[2].attention.geometry_mlp.parameters())
    del kwargs["attention_edge_indices"]["dilated_3"]
    with pytest.raises(ValueError,match="dilated_3"):
        model(x,**kwargs)


def test_default_schedule_and_factory_metadata():
    probe=SimpleNamespace(inputs=torch.zeros(6,3),targets=torch.zeros(6,3),
                          coords=torch.zeros(6,3),conditioning=torch.zeros(1,1))
    cfg=OmegaConf.create(dict(_target_="graph_attention.models.dinat_dit.AlternatingDilatedGeometricDiT",
        hidden_dim=8,num_heads=2,num_layers=3,mlp_ratio=2,condition_embed_dim=8))
    old,_=instantiate_controlled_model(cfg,probe,seed=42)
    assert old.attention_hop_schedule==(1,2)
    cfg.attention_hop_schedule=[1,2,3]
    new,meta=instantiate_controlled_model(cfg,probe,seed=42)
    assert meta["attention_hop_schedule"]==[1,2,3]
    for key,value in old.state_dict().items():
        torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match="attention_hop_schedule"):
        make_model((1,4))


def test_mixed_packed_graph_equivalence_and_renumbering():
    model=make_model(layers=3).eval()
    with torch.no_grad():
        model.final_layer.projection.weight.normal_()
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.05)
        a,b=kwargs_for_chain(6),kwargs_for_chain(7)
        x,y=torch.randn(6,3),torch.randn(7,3)
        expected=torch.cat((model(x,**a),model(y,**b)))
        packed=dict(coords=torch.cat((a["coords"],b["coords"])),
                    edge_index=torch.cat((a["edge_index"],b["edge_index"]+6),dim=1),
                    attention_edge_indices={k:torch.cat((v,b["attention_edge_indices"][k]+6),dim=1)
                                            for k,v in a["attention_edge_indices"].items()},
                    conditioning=torch.cat((a["conditioning"],b["conditioning"])),
                    batch_index=torch.tensor([0]*6+[1]*7),graph_sizes=(6,7))
        z=torch.cat((x,y))
        torch.testing.assert_close(model(z,**packed),expected,rtol=1e-5,atol=2e-5)
        perm=torch.cat((torch.randperm(6),torch.randperm(7)+6))
        inverse=torch.argsort(perm)
        permuted=dict(packed,coords=packed["coords"][perm],
                      edge_index=inverse[packed["edge_index"]],
                      attention_edge_indices={k:inverse[e] for k,e in packed["attention_edge_indices"].items()})
        torch.testing.assert_close(model(z[perm],**permuted)[inverse],expected,rtol=1e-5,atol=2e-5)
