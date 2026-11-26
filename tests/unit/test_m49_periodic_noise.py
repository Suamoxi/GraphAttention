"""Source tying preserves the legacy RNG and does not project the ODE."""
from types import SimpleNamespace
import pytest
import torch

from graph_attention.tasks.flow_matching import FlowMatchingTask, _deterministic_sampling_source


def test_initial_source_tying_only(monkeypatch):
    import graph_attention.tasks.flow_matching as fm
    # Isolate the sampler without constructing an unrelated CFD data fixture.
    batch = SimpleNamespace(inputs=torch.zeros(6,2), ptr=torch.tensor([0,3,6]),
        num_graphs=2,batch_index=torch.tensor([0,0,0,1,1,1]),
        source=SimpleNamespace(sample_ids=("a","b")))
    monkeypatch.setattr(fm,"_validate_state_batch",lambda b:None)
    velocity=torch.arange(6,dtype=torch.float32)[:,None].expand(6,2)
    monkeypatch.setattr(fm,"_model_velocity",lambda *a,**k:velocity)
    task=FlowMatchingTask(state_fields=("rho",))
    keys=("a","b")
    source=_deterministic_sampling_source(batch,5678,keys)
    idx=torch.tensor([0,1,0,3,4,3])
    states={}
    result=task.sample_standardized(torch.nn.Identity(),batch,steps=2,
        sampling_keys=keys,initial_source_index=idx,
        step_observer=lambda step,state:states.update({step:state.clone()}))
    torch.testing.assert_close(states[0],source[idx],rtol=0,atol=0)
    torch.testing.assert_close(result,source[idx]+velocity)
    assert torch.equal(states[0][0],states[0][2])
    assert not torch.equal(result[0],result[2])
    legacy=task.sample_standardized(torch.nn.Identity(),batch,steps=2,sampling_keys=keys)
    torch.testing.assert_close(legacy,source+velocity)
    with pytest.raises(ValueError,match="within each graph"):
        task.sample_standardized(torch.nn.Identity(),batch,steps=2,
                                 initial_source_index=torch.tensor([0,1,3,3,4,3]))
