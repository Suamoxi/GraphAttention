# M6 Training Correctness

## Status

M6 implements the reference-correct training mechanics required before performance benchmarking or sparse-attention work. The implementation covers train-only statistical scaling, per-sample regression loss reduction, optimizer-step accumulation across variable computational microbatches, DDP global sample weighting, and an autocast-compatible loss path.

Target validation on Calypso is required before M6 is marked complete.

## 1. Scope

M6 freezes the first optimization semantics for the deterministic node-regression baseline introduced in M5.

The required preprocessing order is:

$$
\boxed{
q_{\mathrm{dim}}
\rightarrow
q^*
\rightarrow
\hat q
}
$$

where physical nondimensionalization is defined by M3.3 and statistical standardization is fitted only from the M5 training split.

M6 does not add a new CFD model. The node-linear model remains a null geometric baseline.

## 2. Sample-balanced train statistics

For one task channel $c$ and $G$ physical training samples, sample $g$ contains $N_g$ nodes.

M6 uses equal physical-sample weighting:

$$
\mu_c
=
\frac{1}{G}
\sum_{g=1}^{G}
\left[
\frac{1}{N_g}
\sum_{i=1}^{N_g}
q^*_{gic}
\right].
$$

The variance is the equal-weight mixture variance:

$$
\sigma_c^2
=
\frac{1}{G}
\sum_{g=1}^{G}
\left[
\frac{1}{N_g}
\sum_{i=1}^{N_g}
(q^*_{gic}-\mu_c)^2
\right].
$$

The standardized value is:

$$
\hat q_{gic}
=
\frac{q^*_{gic}-\mu_c}{\sigma_c}.
$$

A mesh with more nodes therefore does not receive more influence on the scaler solely because of resolution.

The implementation uses stable online mixture-moment accumulation in float64. It does not concatenate the full training dataset in memory.

## 3. Train-only anti-leakage contract

`fit_train_standardizers` requires:

- an explicit `SplitManifest`;
- an iterable containing exactly the declared training sample IDs;
- each training sample exactly once;
- no validation or test sample in the fit iterable.

A validation/test sample is rejected rather than silently ignored. Missing or duplicated training samples also fail.

The resulting `TaskStandardizers` records:

- ordered input channel names;
- ordered target channel names;
- fitted means and scales;
- exact training sample IDs;
- whether the source task used M3.3 physical nondimensionalization;
- the `sample_balanced` weighting convention.

## 4. Near-zero variance policy

M6 does not silently replace a near-zero standard deviation with one.

The default threshold is:

$$
\sigma_{\min}=10^{-12}.
$$

If any fitted channel satisfies:

$$
\sigma_c \le \sigma_{\min},
$$

fitting fails and identifies the affected channel names.

This is a fail-fast numerical policy. A future task that legitimately contains a constant channel must define an explicit alternative rather than inheriting an implicit clamp.

## 5. Regression loss

For node $i$ in physical sample $g$, with $C$ target channels, the baseline pointwise loss is:

$$
\ell_{gi}
=
\frac{1}{C}
\sum_{c=1}^{C}
(\hat y_{gic}-\hat y^{\mathrm{target}}_{gic})^2.
$$

Channels are equally weighted. For physically heterogeneous targets, M6 therefore expects the task's frozen statistical standardization to make channel magnitudes comparable before this MSE is interpreted as a multi-channel objective.

### 5.1 Unweighted spatial reduction

If no node quadrature weights are supplied:

$$
L_g
=
\frac{1}{N_g}
\sum_{i=1}^{N_g}
\ell_{gi}.
$$

### 5.2 Weighted spatial reduction

If explicit non-negative node weights $w_{gi}$ are supplied:

$$
L_g
=
\frac{
\sum_i w_{gi}\ell_{gi}
}{
\sum_i w_{gi}
}.
$$

Each graph must have strictly positive total weight.

M6 uses `Mesh.node_weights` when present. The scientific meaning of those weights remains a data/geometry responsibility. The real AVBP path does not yet claim that a physical node quadrature has been established.

## 6. Optimizer objective

With equal statistical weight for each physical sample, one optimizer step over $G_{\mathrm{opt}}$ samples is:

$$
L_{\mathrm{opt}}
=
\frac{1}{G_{\mathrm{opt}}}
\sum_{g=1}^{G_{\mathrm{opt}}} L_g.
$$

A computational microbatch contributes a **sum** of its physical-sample losses. The backward scale is determined by the complete optimizer-step sample count, not by the number of graphs in that particular microbatch.

Therefore changing:

```text
4 samples in one microbatch
```

into:

```text
1 sample + 2 samples + 1 sample
```

must produce the same optimizer gradient within numerical tolerance.

`train_equal_sample_optimizer_step` is the M6 reference implementation of this rule.

The caller supplies `local_sample_count` before backward starts. A mismatch between the declared statistical count and observed microbatches fails before `optimizer.step()` and clears accumulated gradients.

## 7. DDP global weighting

PyTorch DDP averages synchronized gradients over `world_size = R` ranks.

Let the global optimizer step contain:

$$
G_{\mathrm{global}}
=
\sum_r G_r
$$

physical samples.

Each rank backpropagates its rank-local **summed** sample loss with scale:

$$
\boxed{
\alpha_{\mathrm{DDP}}
=
\frac{R}{G_{\mathrm{global}}}
}
$$

so after DDP gradient averaging:

$$
\frac{1}{R}
\sum_r
\alpha_{\mathrm{DDP}}
\sum_{g\in r}\nabla L_g
=
\frac{1}{G_{\mathrm{global}}}
\sum_g \nabla L_g.
$$

This remains correct when ranks contain different physical-sample counts.

M6 performs one distributed reduction of the optimizer-step sample counts before backward.

## 8. Unequal microbatch counts across DDP ranks

Ranks may require different numbers of computational microbatches because graph sizes differ.

Synchronizing every rank-local backward call would require identical microbatch counts and can create incorrect synchronization behavior.

M6 therefore uses `DistributedDataParallel.no_sync()` for every local microbatch except the final one. Each rank performs exactly one synchronized backward per optimizer step.

This assumes every participating rank owns at least one microbatch in the optimizer step. If any rank has none, all ranks fail explicitly before backward.

## 9. Autocast and loss precision

M6 allows an explicit `autocast_dtype` in the reference optimizer step.

Model forward execution occurs under native `torch.autocast`. The loss path explicitly promotes prediction and target dtypes before subtraction. This supports cases such as a BF16 autocast model output compared with FP32 frozen targets without silently forcing the stored target tensor to lower precision.

For CUDA FP16, M6 requires an explicit `torch.amp.GradScaler`. BF16 does not require gradient scaling by default.

The default repository trainer configuration remains `32-true`; CUDA AMP/BF16 performance and device-specific numerical behavior are not yet target-validated.

## 10. DDP validation executable

M6 adds:

```bash
torchrun --standalone --nproc_per_node=2 scripts/validate_m6_ddp.py
```

The validation intentionally gives:

- rank 0: three physical samples split across two microbatches;
- rank 1: two physical samples in one microbatch.

It compares the resulting DDP SGD update against a single-process global five-sample reference update.

The test uses CPU/Gloo so the global weighting and uneven-microbatch synchronization rule can be validated without requiring a GPU allocation. NCCL/GPU performance is not claimed by this test.

## 11. Assumptions introduced or inherited

M6 introduces or inherits the following assumptions:

- statistical scaling uses equal total weight per physical training sample;
- nodes inside one sample are equally weighted for scaler fitting;
- statistical scaling is fitted after any enabled M3.3 physical nondimensionalization;
- input and target scalers are separate and preserve exact semantic channel order;
- the baseline regression loss is channel-mean squared error;
- physical samples have equal optimizer-level statistical weight by default;
- `Mesh.node_weights`, when supplied to the loss, are valid spatial integration weights for the task using them;
- every DDP rank has at least one microbatch for an optimizer step;
- the M5 model interface accepts `inputs`, `batch_index`, and `conditioning`;
- the caller knows the optimizer-step local physical-sample count before backward begins.

## 12. Handled edge cases

M6 explicitly handles:

- variable node counts during scaler fitting;
- training samples supplied in any iteration order;
- validation/test leakage attempts;
- duplicated or missing training samples;
- zero/near-zero variance channels;
- exact channel-order mismatches when applying a scaler;
- variable graph sizes in one loss batch;
- weighted and unweighted per-sample spatial reductions;
- negative or zero-total node weights;
- NaN/Inf in scaler data, predictions, targets, or weights;
- different computational microbatch partitions for the same optimizer batch;
- unequal physical-sample counts across DDP ranks;
- unequal rank-local microbatch counts under DDP;
- CPU BF16 autocast mechanics;
- optimizer-count mismatch without applying an optimizer step.

## 13. Deliberately unsupported or deferred cases

The following remain outside M6:

- physical/quadrature-weighted statistical scaler fitting;
- a validated AVBP node-control-volume interpretation;
- arbitrary per-sample optimizer weights other than equal physical-sample weighting;
- learned uncertainty or channel-specific loss weights;
- gradient clipping;
- learning-rate schedulers;
- checkpoint orchestration for complete training runs;
- resuming mid-optimizer-accumulation window;
- a full Lightning training module/data module;
- zero-sample DDP ranks;
- single-graph partitioning;
- CUDA/NCCL AMP performance claims;
- sparse-attention model training.

## 14. Failure behavior

M6 fails explicitly when:

- scaler fitting sees a non-training, duplicate, or missing training sample;
- channel semantics do not match the fitted scaler;
- a channel has standard deviation at or below the configured threshold;
- prediction/target layouts are incompatible;
- spatial weights are invalid;
- the declared optimizer sample count disagrees with supplied microbatches;
- a distributed process group is active without a DDP-wrapped model;
- a DDP rank has no microbatch;
- CUDA FP16 is requested without an explicit GradScaler.

No optimizer step is applied after a detected local sample-count mismatch.

## 15. Efficiency and numerical trade-offs

Scaler fitting is streaming over physical samples and stores only channel-sized moment state, but the current correctness path computes float64 per-sample moments. This is intentionally conservative and is not a throughput-optimized preprocessing implementation.

The reference loss uses a Python loop over graphs for transparent per-sample reduction. Its asymptotic work is still linear in the number of nodes, but a future benchmark may justify a fused/index-reduction implementation.

Finite-value and weight-validity checks can synchronize when tensors are already on a GPU. They are retained in the M6 correctness path. M7 may benchmark whether some validation should move to CPU/data-loading boundaries.

`DDP.no_sync()` reduces synchronization to one backward per optimizer step. The implementation is intended to establish the correct objective first; communication/throughput optimization remains benchmark-driven.

## 16. Validation gate

Before M6 is marked complete on Calypso, run:

```bash
pytest
ruff check .
ruff format --check .
python scripts/inspect_config.py
torchrun --standalone --nproc_per_node=2 scripts/validate_m6_ddp.py
```

Performance evidence remains `ANALYTICAL`. Passing this gate establishes numerical/software correctness for the M6 reference path, not GPU throughput or scaling claims.
