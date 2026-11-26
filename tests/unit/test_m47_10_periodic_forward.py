"""A3 minimum-image displacements reach both training and generation forwards."""
from types import SimpleNamespace

import torch

from scripts.train_m43_b_multires import _forward
from graph_attention.tasks.flow_matching import _model_velocity


class Recorder(torch.nn.Module):
    sparse_attention_backend = "scatter"

    def __init__(self):
        super().__init__()
        self.call_kwargs = None

    def forward(self, inputs, **kwargs):
        self.call_kwargs = kwargs
        return torch.zeros_like(inputs)


def problem(periodic=False):
    return SimpleNamespace(
        inputs=torch.zeros(4,5),
        coords=torch.zeros(4,3),
        edge_index=torch.tensor([[0,1],[1,0]]),
        batch_index=torch.zeros(4,dtype=torch.long),
        conditioning=torch.zeros(1,2),
        attention_edge_indices={"dilated":torch.tensor([[0,2],[2,0]])},
        num_graphs=1, ptr=torch.tensor([0,4]),
        source=SimpleNamespace(mesh_metadata=(
            {"periodic_length_nondimensional":2.0} if periodic else {},
        )),
    )


def test_training_forwards_periodic_metadata_only_for_a3():
    model=Recorder()
    _forward(model,problem(periodic=True))
    assert model.call_kwargs["periodic_length"]==2.0
    _forward(model,problem(periodic=False))
    assert "periodic_length" not in model.call_kwargs


def test_generation_forwards_periodic_metadata_only_for_a3():
    model=Recorder()
    batch=problem(periodic=True)
    _model_velocity(model,batch,batch.inputs,torch.tensor([.5]),time_embedding_scale=1000.0)
    assert model.call_kwargs["periodic_length"]==2.0
    batch=problem(periodic=False)
    _model_velocity(model,batch,batch.inputs,torch.tensor([.5]),time_embedding_scale=1000.0)
    assert "periodic_length" not in model.call_kwargs
