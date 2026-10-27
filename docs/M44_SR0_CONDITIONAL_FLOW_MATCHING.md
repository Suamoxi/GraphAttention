# M44 SR-0 — Conditional residual Flow Matching baseline

## Scope

**Status:** first reusable model/task core and CPU unit gates, not a production
DNS/LES paired-data reader or a validated unstructured-mesh SR run. Existing M37
backbone, unconditional FlowMatchingTask, Heun solver and checkpoints remain
unchanged. No scientific operator is redefined in GraphAttention.

SR-0 uses **no coarse neural encoder**. The only learned model is the existing
`AlternatingDilatedGeometricDiT` with 15 input channels and 5 output channels.

## Authoritative input contract

The paired-data provider (dnstoles) must supply, for every physical sample:

- A coarse LES mesh and conservative state `Uc` with 5 channels in order
  `(rho, rhou, rhov, rhow, rhoE)`.
- The hierarchically refined fine mesh, its original one-hop graph and exact-two-hop
  attention graph, and a packed `NodeRegressionBatch` representing that **fine** graph.
- `T` as sparse CSR interpolation weights from parent coarse nodes to each fine node.
  Its rows must preserve constants; `T` is **not** a physical conservative lifting.
- A fine physical baseline `Ubase` from the chosen dnstoles lifting, and an
  optional DNS training target `Uf` from the SAME DNS-derived source sample.
- Explicit M3.3 physical reference scales for every sample, especially `L_ref`.
  Supply both coarse and fine coordinates divided by the **same** case `L_ref`.
  Supply all 5 conservative fields **after** the same M3.3 nondimensionalization.
- Splits grouped by source DNS snapshot/trajectory, not by the derived mesh pair.

`SR0PairBatch.fine_target=None` is valid for inference. It does not request a
fine DNS field. `SR0PairBatch` validates shapes, sample boundaries, finite data,
conservative field reference metadata and required exact-two-hop attention edges.
It cannot independently verify that a producer used the declared scales or
that its baseline satisfies `Q Ubase = Uc`; those are upstream scientific gates.

The implementation deliberately does **not** infer physical operators from
node counts, nearest-neighbor distances or a requested refinement factor.
Construction of hierarchical tetra/hexa refinement, transfer weights, boundary
conditions, coarse restriction `Q`, and exact physical lifting remains in
dnstoles. That producer-to-model interface is the next integration milestone.

## Statistics — train-only

`fit_sr0_scales` requires an explicit list of train sample IDs and rejects
held-out, duplicated or missing IDs. After M3.3 nondimensionalization:

- Coarse conservative inputs use equal-physical-sample mean and standard
  deviation (each sample has equal total statistical weight).
- Fine baseline uses the **same coarse** channel mean and standard deviation.
- The target residual uses train-only **uncentered RMS** per conservative channel:
  `z=(Uf-Ubase)/residual_rms`. Physical zero residual therefore stays zero.
- No target snapshot statistics enter an inference condition.

Statistics and sample IDs must be saved with experiment checkpoints by the
future paired-data runner. The current core does not yet implement checkpoint
serialization or a new generic runner.

## Training — exactly M11 straight-path Flow Matching

For each matched pair:

1. `z=(Uf-Ubase)/residual_rms`.
2. Standardize coarse LES field using the frozen coarse means/scales.
3. Transfer it once to fine nodes: `c_f=T((Uc-mean_c)/std_c)`.
4. Standardize physical baseline: `b_f=(Ubase-mean_c)/std_c`.
5. Draw per-physical-sample `t ~ Uniform(0,1)` and fine-node
   `epsilon ~ Normal(0,I)` using the **existing** `FlowMatchingTask`.
6. Construct `z_t=(1-t)*epsilon+t*z`, `v*=z-epsilon`.
7. Concatenate `[z_t, c_f, b_f]` as 15 fine-node inputs. The existing M37
   DiNAT-DiT predicts 5 velocity channels. Existing DiT adaLN receives global
   physical conditions plus time from FlowMatchingTask.
8. Apply existing `FlowMatchingTask.training_loss` (sample-reduced MSE with
   optional node-volume weights) and backpropagate as usual.

`SR0FlowMatching.model_probe(problem)` produces the expected `[N_f,15]` to
`[N_f,5]` probe used by `instantiate_controlled_model` and the existing
`configs/model/dinat_dit.yaml`. The backbone and its self-attention policy,
local/exact-two-hop schedule, sparse backend and activation checkpointing are
unchanged. Existing 5-channel unconditional checkpoints remain independent.

Sketch (the caller supplies the scientifically prepared SR0PairBatch objects):

```python
from graph_attention.tasks.sr0 import SR0FlowMatching, fit_sr0_scales
from graph_attention.training.model_factory import instantiate_controlled_model

scales = fit_sr0_scales(train_pairs, train_sample_ids=train_sample_ids)
task = SR0FlowMatching(scales, time_embedding_scale=1000.0)
problem = task.make_training_problem(train_pair, generator=train_generator)
model, initialization = instantiate_controlled_model(model_cfg, task.model_probe(problem), seed=seed)
optimizer.zero_grad(set_to_none=True)
task.training_loss(model, problem).mean.backward()
optimizer.step()
```

## Inference — LES only and unchanged Heun

The caller supplies `SR0PairBatch(fine_target=None, ...)` for a coarse LES
condition, plus saved `SR0Scales` and a trained model. The routine computes
`[c_f,b_f]` **once**, then calls the existing
`FlowMatchingTask.sample_standardized` through a lightweight input adapter:

```python
model.eval()
U_sr_dimensionless = task.generate(
    model, inference_pair, steps=50, solver="heun", sampling_seed=5678,
)
```

M11 generates Gaussian initial `z_0`, performs 50 Heun steps (100 model
calls), and returns `z_1`. SR-0 reconstructs
`U_sr = Ubase + residual_rms * z_1`. Distinct `sampling_keys` give distinct
reproducible draws without changing the supplied coarse condition.

This first baseline does **not** enforce `Q U_sr = Uc`, positive density,
positive pressure, or SGS fidelity. Final projection/limiting, when used,
requires explicitly validated dnstoles operators and a separate controlled
experiment. The direct-transferred LES input is spatial conditioning, not
physical restriction or lifting.

## Tests and next integration gate

Run:

```bash
python -m pytest tests/unit/test_m44_sr0.py -q
python -m pytest tests/unit/test_flow_matching_task.py tests/unit/test_dinat_dit.py -q
```

The unit tests cover transfer, partition-of-unity, mandatory length-reference
metadata, train-only statistics, M11 path identity, a real small M37 gradient
update, and inference with no fine target plus deterministic Heun.

**Not yet implemented:** a paired hierarchical-mesh loader, source-snapshot
split manifests and saved scaling/checkpoints in a full training runner,
cluster/GPU benchmark, M41 halo executor integration, dnstoles `Q/B` projection,
or scientifically validated generation on CFD data. These are blocked on
explicit paired-data/operator contracts and should not be claimed complete.
