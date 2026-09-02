"""Minimal M1 run-provenance collection.

Scientific data, preprocessing, and split provenance are added in later milestones.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_runtime_provenance(repo_root: str | Path = ".") -> dict[str, Any]:
    """Collect the reproducibility metadata available at M1.

    This deliberately records only repository/runtime state. Dataset manifests,
    physical preprocessing, field semantics, and statistical scalers belong to
    later milestones and must not be fabricated as placeholders.
    """

    root = Path(repo_root).resolve()
    sha = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output(root, "status", "--porcelain")

    cuda_version = torch.version.cuda
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return {
        "git": {
            "sha": sha,
            "branch": branch,
            "dirty": None if status is None else bool(status),
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": cuda_version,
            "cuda_available": torch.cuda.is_available(),
        },
        "hardware": {
            "gpu_count": torch.cuda.device_count(),
            "gpus": gpu_names,
        },
    }
