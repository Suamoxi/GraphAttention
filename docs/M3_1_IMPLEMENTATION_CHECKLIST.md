# M3.1 Implementation Checklist

M3.1 is a software/infrastructure milestone. It introduces no physical model, real CFD reader, preprocessing rule, batching policy, or performance claim.

## Implemented

- deterministic `SyntheticMeshDataset`;
- variable node counts;
- variable edge counts;
- variable graph topology (`chain`, `cycle`, `star`);
- variable random geometry from a sample-local generator;
- named scalar/vector fields using the M2 contracts;
- uniform normalized synthetic node weights;
- Hydra construction through the `data: synthetic` group;
- tests for contract validation, variability, determinism, RNG isolation, configuration validation, and Hydra instantiation.

## Cluster validation

After merging, validate from `main` with:

```bash
git pull --rebase origin main
pytest
ruff check .
ruff format --check .
python scripts/inspect_config.py
```

Do not mark M3.1 complete until these checks pass on the target development environment.
