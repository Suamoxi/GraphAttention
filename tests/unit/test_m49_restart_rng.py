"""Checkpointed shuffle state must reproduce uninterrupted epoch ordering."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from graph_attention.training.data_pipeline import restore_training_loader_generator


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_restart_keeps_legacy_shuffle_stream(workers):
    ds = TensorDataset(torch.arange(32))
    def loader():
        return DataLoader(ds, batch_size=8, shuffle=True,
                          num_workers=workers, persistent_workers=workers > 0,
                          generator=torch.Generator().manual_seed(42))
    def epoch(dl):
        return torch.cat([b[0] for b in dl])
    reference = loader()
    epoch(reference)
    epoch(reference)
    state = reference.generator.get_state().clone()
    expected = [(epoch(reference), reference.generator.get_state().clone()) for _ in range(2)]
    restarted = loader()
    shuffle_rng = restore_training_loader_generator(restarted, state)
    for ids, rng in expected:
        assert torch.equal(epoch(restarted), ids)
        assert torch.equal(shuffle_rng.get_state(), rng)
    # A second allocation must be restartable using the same checkpoint field.
    state = shuffle_rng.get_state().clone()
    next_ids = epoch(reference)
    again = loader()
    restore_training_loader_generator(again, state)
    assert torch.equal(epoch(again), next_ids)
