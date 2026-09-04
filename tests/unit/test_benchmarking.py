import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from graph_attention.utils.benchmarking import measure_callable


def test_measure_callable_reports_cpu_timing_without_cuda_memory() -> None:
    tensor = torch.ones(16)

    measurement = measure_callable(
        lambda: tensor.square().sum(),
        device="cpu",
        warmup=1,
        repetitions=3,
    )

    assert measurement.timing.warmup == 1
    assert measurement.timing.repetitions == 3
    assert measurement.timing.median_ms >= 0.0
    assert measurement.timing.min_ms >= 0.0
    assert measurement.timing.max_ms >= measurement.timing.min_ms
    assert measurement.cuda_memory is None


@pytest.mark.parametrize(
    ("warmup", "repetitions", "error"),
    [
        (-1, 1, ValueError),
        (0, 0, ValueError),
        (True, 1, TypeError),
        (0, False, TypeError),
    ],
)
def test_measure_callable_rejects_invalid_counts(
    warmup: int,
    repetitions: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        measure_callable(lambda: None, device="cpu", warmup=warmup, repetitions=repetitions)


def test_m7_benchmark_script_writes_cpu_synthetic_result(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "m7.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_m7.py"),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--output",
            str(output),
            "synthetic",
            "--graphs",
            "2",
            "--nodes-per-graph",
            "4",
            "--spatial-dim",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "M7_node_linear_baseline"
    assert payload["evidence"] == "LOCAL_BENCHMARK"
    assert payload["workload"]["num_graphs"] == 2
    assert payload["workload"]["num_nodes"] == 8
    assert payload["cuda_device"] is None
    assert payload["measurements"]["training_iteration"]["timing"]["repetitions"] == 1


def test_m8_benchmark_script_writes_cpu_sparse_result(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "m8.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_m8.py"),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--hidden-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--mlp-ratio",
            "2",
            "--output",
            str(output),
            "synthetic",
            "--graphs",
            "2",
            "--nodes-per-graph",
            "4",
            "--spatial-dim",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "M8_sparse_graph_transformer"
    assert payload["evidence"] == "LOCAL_BENCHMARK"
    assert payload["workload"]["num_graphs"] == 2
    assert payload["workload"]["num_nodes"] == 8
    assert payload["execution"]["hidden_dim"] == 16
    assert payload["execution"]["uses_coordinates"] is False
    assert payload["cuda_device"] is None
    assert payload["measurements"]["forward"]["throughput"]["edges_per_second"] is not None
    assert payload["measurements"]["training_iteration"]["timing"]["repetitions"] == 1
