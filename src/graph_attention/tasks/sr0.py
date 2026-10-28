"""M44 SR-0: conditional residual Flow Matching without a coarse neural encoder.

All physical restriction, baseline and parent-element transfer operators are
provided by the paired-data producer (dnstoles); this module does not define
or infer a DNS-to-LES coarse-graining operator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import nn

from graph_attention.data.nondimensionalization import ConvectiveNondimensionalizer
from graph_attention.objectives import SampleLossAggregate

from .flow_matching import FlowMatchingTask
from .sr0_evaluation import point_sample_anchors
from .regression import NodeRegressionBatch

_CONSERVATIVE = ("rho", "rhou", "rhov", "rhow", "rhoE")
_RESIDUAL_CHANNELS = tuple(f"sr_residual:{field}" for field in _CONSERVATIVE)
_INPUT_CHANNELS = (
    *_RESIDUAL_CHANNELS,
    *(f"sr_interpolated:{field}" for field in _CONSERVATIVE),
    *(f"sr_baseline:{field}" for field in _CONSERVATIVE),
)


@dataclass(frozen=True, slots=True)
class CoarseToFineCSR:
    """Externally supplied parent-element feature transfer, stored as CSR.

    Columns index coarse nodes, rows index fine nodes. Weights are dimensionless
    and must form a partition of unity on every fine node. This is NOT the
    conservative physical lifting operator or the fine-to-coarse restriction.
    """

    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    weights: torch.Tensor
    num_coarse_nodes: int

    def __post_init__(self) -> None:
        rowptr, cols, weights = self.crow_indices, self.col_indices, self.weights
        if rowptr.ndim != 1 or rowptr.numel() < 2 or rowptr.dtype != torch.long:
            raise ValueError("transfer crow_indices must be int64 [N_f+1]")
        if cols.ndim != 1 or cols.dtype != torch.long:
            raise ValueError("transfer col_indices must be int64 [nnz]")
        if weights.ndim != 1 or not weights.is_floating_point():
            raise ValueError("transfer weights must be floating-point [nnz]")
        if cols.numel() != weights.numel():
            raise ValueError("transfer indices and weights have different lengths")
        if not isinstance(self.num_coarse_nodes, int) or self.num_coarse_nodes <= 0:
            raise ValueError("num_coarse_nodes must be a positive integer")
        if not (rowptr.device == cols.device == weights.device):
            raise ValueError("all transfer arrays must be on the same device")
        if int(rowptr[0]) != 0 or int(rowptr[-1]) != cols.numel():
            raise ValueError("transfer CSR row offsets must span all weights")
        counts = rowptr[1:] - rowptr[:-1]
        if torch.any(counts <= 0):
            raise ValueError("every fine node must have at least one coarse parent")
        if torch.any(cols < 0) or torch.any(cols >= self.num_coarse_nodes):
            raise ValueError("transfer contains an out-of-range coarse index")
        if not torch.isfinite(weights).all() or torch.any(weights < -1.0e-7):
            raise ValueError("transfer weights must be finite and nonnegative")
        row_ids = torch.repeat_interleave(
            torch.arange(self.num_fine_nodes, device=rowptr.device), counts
        )
        sums = torch.zeros(self.num_fine_nodes, dtype=weights.dtype, device=weights.device)
        sums.index_add_(0, row_ids, weights)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1.0e-5, rtol=1.0e-5):
            raise ValueError("coarse-to-fine interpolation must preserve constants")

    @property
    def num_fine_nodes(self) -> int:
        return self.crow_indices.numel() - 1

    def apply(self, coarse_features: torch.Tensor) -> torch.Tensor:
        """Interpolate any number of coarse feature channels without a dense matrix."""
        if coarse_features.ndim != 2 or coarse_features.shape[0] != self.num_coarse_nodes:
            raise ValueError("coarse feature node count does not match CSR columns")
        if coarse_features.device != self.weights.device:
            raise ValueError("coarse features and CSR transfer must share a device")
        if coarse_features.dtype != self.weights.dtype:
            raise TypeError("coarse features and CSR weights must have the same dtype")
        operator = torch.sparse_csr_tensor(
            self.crow_indices,
            self.col_indices,
            self.weights,
            size=(self.num_fine_nodes, self.num_coarse_nodes),
        )
        return torch.sparse.mm(operator, coarse_features)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> CoarseToFineCSR:
        return CoarseToFineCSR(
            self.crow_indices.to(device=device),
            self.col_indices.to(device=device),
            self.weights.to(device=device, dtype=dtype),
            self.num_coarse_nodes,
        )


@dataclass(frozen=True, slots=True)
class SR0PairBatch:
    """Paired *dimensionless* M3.3 conservative fields on two mesh resolutions.

    ``fine`` supplies the existing packed fine graph, fine coordinates (already
    divided by the case L_ref), graph metadata, and global regime conditioning.
    Its input/target field values are ignored. The fine DNS target is optional
    so inference never needs an unavailable high-resolution physical field.
    """

    fine: NodeRegressionBatch
    coarse_state: torch.Tensor
    fine_baseline: torch.Tensor
    transfer: CoarseToFineCSR
    coarse_ptr: torch.Tensor
    fine_target: torch.Tensor | None = None

    def __post_init__(self) -> None:
        n_f = self.fine.coords.shape[0]
        n_c = self.transfer.num_coarse_nodes
        if self.coarse_state.shape != (n_c, 5):
            raise ValueError("coarse_state must have shape [N_c, 5]")
        if self.fine_baseline.shape != (n_f, 5):
            raise ValueError("fine_baseline must have shape [N_f, 5]")
        if self.fine_target is not None and self.fine_target.shape != (n_f, 5):
            raise ValueError("fine_target must have shape [N_f, 5]")
        if self.transfer.num_fine_nodes != n_f:
            raise ValueError("transfer row count does not match fine mesh")
        if self.coarse_ptr.dtype != torch.long or self.coarse_ptr.ndim != 1:
            raise TypeError("coarse_ptr must be int64 [B+1]")
        if self.coarse_ptr.shape != self.fine.ptr.shape:
            raise ValueError("coarse_ptr and fine graph ptr must have the same shape")
        if int(self.coarse_ptr[0]) != 0 or int(self.coarse_ptr[-1]) != n_c:
            raise ValueError("coarse_ptr must cover all coarse nodes")
        if torch.any(self.coarse_ptr[1:] <= self.coarse_ptr[:-1]):
            raise ValueError("coarse_ptr must describe nonempty coarse graphs")
        if self.fine.coords.shape[1] != 3:
            raise ValueError("M44 SR-0 currently supports three-dimensional fine meshes")
        tensors = [self.coarse_state, self.fine_baseline, self.fine.coords]
        if self.fine_target is not None:
            tensors.append(self.fine_target)
        for value in tensors:
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("SR-0 physical fields and coordinates must be finite floats")
            if value.dtype != self.fine.coords.dtype or value.device != self.fine.coords.device:
                raise TypeError("SR-0 physical fields must share fine coordinate dtype/device")
        if self.transfer.weights.device != self.fine.coords.device:
            raise ValueError("CSR transfer and fine mesh must be on the same device")
        if self.transfer.weights.dtype != self.fine.coords.dtype:
            raise TypeError("CSR weights and fine coordinates must have the same dtype")
        if self.coarse_ptr.device != self.fine.coords.device:
            raise ValueError("coarse_ptr and fine graph must be on the same device")
        if "dilated" not in self.fine.attention_edge_indices:
            raise ValueError("SR-0 requires fine exact-two-hop ('dilated') attention edges")
        if len(self.fine.source.reference_scales) != self.fine.num_graphs:
            raise ValueError("every fine graph must carry physical reference scales")
        for references in self.fine.source.reference_scales:
            converter = ConvectiveNondimensionalizer(references)
            converter.coordinate_scale()
            for field in _CONSERVATIVE:
                converter.field_scale(field)
        # No interpolation edge may connect different physical samples.
        for graph in range(self.fine.num_graphs):
            f0, f1 = (int(v) for v in self.fine.ptr[graph:graph + 2])
            c0, c1 = (int(v) for v in self.coarse_ptr[graph:graph + 2])
            start = int(self.transfer.crow_indices[f0])
            stop = int(self.transfer.crow_indices[f1])
            cols = self.transfer.col_indices[start:stop]
            if torch.any(cols < c0) or torch.any(cols >= c1):
                raise ValueError("coarse-to-fine transfer crosses physical sample boundaries")

    def to(self, *, device: torch.device, dtype: torch.dtype) -> SR0PairBatch:
        from graph_attention.training.data_pipeline import task_batch_to_device

        return SR0PairBatch(
            fine=task_batch_to_device(self.fine, device=device, dtype=dtype),
            coarse_state=self.coarse_state.to(device=device, dtype=dtype),
            fine_baseline=self.fine_baseline.to(device=device, dtype=dtype),
            transfer=self.transfer.to(device=device, dtype=dtype),
            coarse_ptr=self.coarse_ptr.to(device=device),
            fine_target=(
                self.fine_target.to(device=device, dtype=dtype)
                if self.fine_target is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class SR0Scales:
    """Train-only channel statistics after M3.3 physical nondimensionalization."""

    coarse_mean: torch.Tensor
    coarse_std: torch.Tensor
    residual_rms: torch.Tensor
    train_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("coarse_mean", "coarse_std", "residual_rms"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != (5,):
                raise ValueError(f"{name} must be a tensor with shape [5]")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if torch.any(self.coarse_std <= 0) or torch.any(self.residual_rms <= 0):
            raise ValueError("SR-0 training scales must be strictly positive")
        if not self.train_sample_ids or len(set(self.train_sample_ids)) != len(self.train_sample_ids):
            raise ValueError("SR-0 scales require unique training sample IDs")


def fit_sr0_scales(
    pairs: Iterable[SR0PairBatch], *, train_sample_ids: Iterable[str]
) -> SR0Scales:
    """Equal-physical-sample moments, accepting only the declared train split.

    Coarse conditions use centered standardization. Conservative residuals use
    an uncentered RMS so zero remains the exact physical zero residual.
    """
    expected = tuple(train_sample_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("train_sample_ids must be a nonempty sequence of unique IDs")
    remaining = set(expected)
    coarse_mean = torch.zeros(5, dtype=torch.float64)
    coarse_second = torch.zeros(5, dtype=torch.float64)
    residual_second = torch.zeros(5, dtype=torch.float64)
    count = 0
    for pair in pairs:
        if pair.fine_target is None:
            raise ValueError("fitting SR-0 scales requires the fine DNS target")
        for graph, sample_id in enumerate(pair.fine.source.sample_ids):
            if sample_id not in remaining:
                raise ValueError(f"SR-0 train split contains duplicate or unapproved ID: {sample_id}")
            remaining.remove(sample_id)
            c0, c1 = (int(v) for v in pair.coarse_ptr[graph:graph + 2])
            f0, f1 = (int(v) for v in pair.fine.ptr[graph:graph + 2])
            uc = pair.coarse_state[c0:c1].detach().to(device="cpu", dtype=torch.float64)
            r = (pair.fine_target[f0:f1] - pair.fine_baseline[f0:f1]).detach().to(
                device="cpu", dtype=torch.float64
            )
            coarse_mean += uc.mean(dim=0)
            coarse_second += uc.square().mean(dim=0)
            residual_second += r.square().mean(dim=0)
            count += 1
    if remaining:
        raise ValueError(f"SR-0 training data missing expected samples: {sorted(remaining)}")
    mean = coarse_mean / count
    std = torch.clamp(coarse_second / count - mean.square(), min=0).sqrt()
    rms = (residual_second / count).sqrt()
    if torch.any(std <= 1.0e-12) or torch.any(rms <= 1.0e-12):
        raise ValueError("cannot fit SR-0 statistics with zero-variance condition or residual")
    return SR0Scales(mean, std, rms, expected)


@dataclass(frozen=True, slots=True)
class SR0FlowProblem:
    """Time-dependent five-channel M11 problem plus static spatial condition."""

    flow: NodeRegressionBatch
    spatial_condition: torch.Tensor
    free_node_mask: torch.Tensor | None = None


class _StaticSpatialCondition(nn.Module):
    """Adapter that lets the unchanged M11 Heun solver call a 15-input M37."""

    def __init__(
        self,
        backbone: nn.Module,
        spatial_condition: torch.Tensor,
        free_node_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.spatial_condition = spatial_condition
        self.free_node_mask = free_node_mask

    def forward(self, inputs: torch.Tensor, **model_kwargs: object) -> torch.Tensor:
        if inputs.shape != (self.spatial_condition.shape[0], 5):
            raise ValueError("SR-0 Flow Matching state must have shape [N_f, 5]")
        velocity = self.backbone(
            torch.cat((inputs, self.spatial_condition), dim=1), **model_kwargs
        )
        if self.free_node_mask is not None:
            velocity = velocity.masked_fill(~self.free_node_mask[:, None], 0.0)
        return velocity


class SR0FlowMatching:
    """Keep M11 noise, loss and Heun numerics; change only spatial conditioning."""

    def __init__(
        self,
        scales: SR0Scales,
        *,
        validation_seed: int = 1234,
        time_embedding_scale: float = 1.0,
        spatial_condition_mode: str = "interpolated_plus_baseline",
        point_sample_constraint: bool = False,
    ) -> None:
        if spatial_condition_mode not in {"interpolated_only", "interpolated_plus_baseline"}:
            raise ValueError("unsupported SR-0 spatial_condition_mode")
        self.spatial_condition_mode = spatial_condition_mode
        if not isinstance(point_sample_constraint, bool):
            raise TypeError("point_sample_constraint must be a bool")
        self.point_sample_constraint = point_sample_constraint
        self.scales = scales
        self.flow = FlowMatchingTask(
            state_fields=_RESIDUAL_CHANNELS,
            validation_seed=validation_seed,
            time_embedding_scale=time_embedding_scale,
        )

    def spatial_condition(self, pair: SR0PairBatch) -> torch.Tensor:
        """Compute once per coarse field; reuse across all 100 Heun evaluations."""
        mean = self.scales.coarse_mean.to(
            device=pair.coarse_state.device, dtype=pair.coarse_state.dtype
        )
        std = self.scales.coarse_std.to(
            device=pair.coarse_state.device, dtype=pair.coarse_state.dtype
        )
        coarse = (pair.coarse_state - mean) / std
        lifted = pair.transfer.apply(coarse)
        if self.spatial_condition_mode == "interpolated_only":
            return lifted
        baseline = (pair.fine_baseline - mean) / std
        return torch.cat((lifted, baseline), dim=1)

    def residual_batch(self, pair: SR0PairBatch, *, require_target: bool) -> NodeRegressionBatch:
        if require_target and pair.fine_target is None:
            raise ValueError("SR-0 training/validation requires a paired fine DNS target")
        scale = self.scales.residual_rms.to(
            device=pair.fine_baseline.device, dtype=pair.fine_baseline.dtype
        )
        residual = (
            (pair.fine_target - pair.fine_baseline) / scale
            if pair.fine_target is not None and require_target
            else torch.zeros_like(pair.fine_baseline)
        )
        return replace(
            pair.fine,
            inputs=residual,
            targets=residual,
            input_channels=_RESIDUAL_CHANNELS,
            target_channels=_RESIDUAL_CHANNELS,
        )


    def point_sample_free_mask(
        self, pair: SR0PairBatch, *, require_target: bool = True
    ) -> torch.Tensor | None:
        """Unobserved fine nodes for the verified Q_sample point-sampling proxy.

        Must NOT be used as a physical LES restriction; incompatible source
        metadata, baseline/observation mismatch or target mismatches fail closed.
        """
        if not self.point_sample_constraint:
            return None
        if any(
            meta.get("sr0_pilot") != "point_sampled_dns_not_les"
            for meta in pair.fine.source.sample_metadata
        ):
            raise ValueError(
                "point-sampling nullspace requires explicitly tagged SR-0 DNS proxy data"
            )
        anchor, coarse_id = point_sample_anchors(pair.transfer)
        observed = pair.coarse_state[coarse_id]
        if not torch.allclose(
            pair.fine_baseline[anchor], observed, atol=1e-5, rtol=1e-6
        ):
            raise ValueError("point-sample interpolation must exactly reproduce coarse anchors")
        if require_target:
            if pair.fine_target is None:
                raise ValueError("point-sample training requires a fine target")
            if not torch.allclose(
                pair.fine_target[anchor], observed, atol=1e-5, rtol=1e-6
            ):
                raise ValueError("point-sample fine target differs at observed coarse anchors")
        free = torch.ones(
            pair.fine.coords.shape[0], dtype=torch.bool, device=pair.fine.coords.device
        )
        free[anchor] = False
        for graph in range(pair.fine.num_graphs):
            f0, f1 = (int(x) for x in pair.fine.ptr[graph:graph + 2])
            if not bool(free[f0:f1].any()):
                raise ValueError("point-sample constraint leaves a graph with no free nodes")
        return free

    def _constrained_problem(
        self, pair: SR0PairBatch, flow: NodeRegressionBatch
    ) -> SR0FlowProblem:
        free = self.point_sample_free_mask(pair)
        if free is not None:
            # M(z_t) and M(v*) are the exact straight FM path between
            # M(epsilon) and the zero-at-anchor data residual. The source
            # Gaussian/time RNG draws are unchanged.
            weights = free.to(dtype=flow.inputs.dtype)
            if flow.node_weights is not None:
                weights = weights * flow.node_weights
            flow = replace(
                flow,
                inputs=flow.inputs.masked_fill(~free[:, None], 0.0),
                targets=flow.targets.masked_fill(~free[:, None], 0.0),
                source=replace(flow.source, node_weights=weights),
            )
        return SR0FlowProblem(flow, self.spatial_condition(pair), free)

    def make_training_problem(
        self, pair: SR0PairBatch, *, generator: torch.Generator | None = None
    ) -> SR0FlowProblem:
        residual = self.residual_batch(pair, require_target=True)
        return self._constrained_problem(
            pair, self.flow.make_training_problem(residual, generator=generator)
        )

    def make_validation_problem(self, pair: SR0PairBatch) -> SR0FlowProblem:
        residual = self.residual_batch(pair, require_target=True)
        return self._constrained_problem(pair, self.flow.make_validation_problem(residual))

    def model_probe(self, problem: SR0FlowProblem) -> NodeRegressionBatch:
        """Produce a [N_f,10 or 15]->[N_f,5] controlled M37 factory probe."""
        channels = (
            _INPUT_CHANNELS[:10]
            if self.spatial_condition_mode == "interpolated_only"
            else _INPUT_CHANNELS
        )
        return replace(
            problem.flow,
            inputs=torch.cat((problem.flow.inputs, problem.spatial_condition), dim=1),
            input_channels=channels,
        )

    def training_loss(
        self, model: nn.Module, problem: SR0FlowProblem
    ) -> SampleLossAggregate:
        predicted = _StaticSpatialCondition(
            model, problem.spatial_condition, problem.free_node_mask
        )(
            problem.flow.inputs,
            edge_index=problem.flow.edge_index,
            coords=problem.flow.coords,
            batch_index=problem.flow.batch_index,
            conditioning=problem.flow.conditioning,
            attention_edge_indices=problem.flow.attention_edge_indices,
        )
        return self.flow.training_loss(predicted, problem.flow)

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        pair: SR0PairBatch,
        *,
        steps: int = 50,
        solver: str = "heun",
        sampling_seed: int = 5678,
        sampling_keys: Iterable[str] | None = None,
    ) -> torch.Tensor:
        """Generate dimensionless fine conservative state from LES-only inputs."""
        residual_batch = self.residual_batch(pair, require_target=False)
        free = self.point_sample_free_mask(pair, require_target=False)
        adapter = _StaticSpatialCondition(model, self.spatial_condition(pair), free)
        z = self.flow.sample_standardized(
            adapter,
            residual_batch,
            steps=steps,
            solver=solver,
            sampling_seed=sampling_seed,
            sampling_keys=sampling_keys,
            state_mask=(free[:, None] if free is not None else None),
        )
        scale = self.scales.residual_rms.to(device=z.device, dtype=z.dtype)
        return pair.fine_baseline + z * scale
