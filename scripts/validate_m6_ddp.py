"""Validate M6 equal-sample DDP weighting with uneven rank/microbatch composition."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from graph_attention.data import SyntheticMeshDataset
from graph_attention.models import NodeLinearBaseline
from graph_attention.tasks import NodeRegressionTask
from graph_attention.training import sample_reduced_mse, train_equal_sample_optimizer_step


def main() -> None:
    dist.init_process_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError("validate_m6_ddp.py must be launched with exactly 2 processes")

        dataset = SyntheticMeshDataset(num_samples=5, spatial_dim=2, seed=101)
        samples = [dataset[index] for index in range(5)]
        task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))

        torch.manual_seed(17)
        model = NodeLinearBaseline(in_channels=2, out_channels=1)
        initial_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        ddp_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.05)

        if rank == 0:
            local_samples = samples[:3]
            microbatches = [
                task.pack_and_prepare(local_samples[:1], dataset.field_catalog),
                task.pack_and_prepare(local_samples[1:], dataset.field_catalog),
            ]
        else:
            local_samples = samples[3:]
            microbatches = [
                task.pack_and_prepare(local_samples, dataset.field_catalog),
            ]

        result = train_equal_sample_optimizer_step(
            ddp_model,
            optimizer,
            microbatches,
            local_sample_count=len(local_samples),
        )

        dist.barrier()
        if rank == 0:
            reference = NodeLinearBaseline(in_channels=2, out_channels=1)
            reference.load_state_dict(initial_state)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
            full_batch = task.pack_and_prepare(samples, dataset.field_catalog)

            reference_optimizer.zero_grad(set_to_none=True)
            predictions = reference(
                full_batch.inputs,
                batch_index=full_batch.batch_index,
                conditioning=full_batch.conditioning,
            )
            aggregate = sample_reduced_mse(
                predictions,
                full_batch.targets,
                full_batch.ptr,
                node_weights=full_batch.node_weights,
            )
            aggregate.mean.backward()
            reference_optimizer.step()

            torch.testing.assert_close(
                ddp_model.module.linear.weight,
                reference.linear.weight,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            torch.testing.assert_close(
                ddp_model.module.linear.bias,
                reference.linear.bias,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            torch.testing.assert_close(
                result.objective,
                aggregate.mean.detach().to(torch.float64),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            print(
                "M6 DDP validation PASS: equal-sample update matches global reference "
                "with rank sample counts 3 and 2 and unequal microbatch counts."
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
