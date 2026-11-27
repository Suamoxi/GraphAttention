from scripts.benchmark_m49_depth import choose_trial


def test_memory_gate_prefers_fastest_safe_configuration():
    def row(b,seconds,allocated,reserved,status="ok"):
        return dict(microbatch=b,median_update_seconds=seconds,
                    peak_allocated_gib=allocated,peak_reserved_gib=reserved,status=status)
    safe=row(4,2.,75.,85.)
    faster_but_unsafe=row(8,1.,86.,91.)
    slow=row(2,3.,40.,50.)
    oom={"status":"cuda_oom"}
    assert choose_trial([safe,faster_but_unsafe,slow,oom],100.)==safe
    assert choose_trial([oom,faster_but_unsafe],100.) is None


def test_reuse_benchmark_on_newer_clean_revision(tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace
    from scripts import benchmark_m49_depth as bench

    runtime = {"git": {"sha": "current", "dirty": False},
               "runtime": {"pytorch": "test"}, "hardware": {"gpu_count": 1}}
    measured = dict(runtime, git={"sha": "older", "dirty": False})
    row = dict(microbatch=8, accumulation=2, activation_checkpointing=False,
               median_update_seconds=1., peak_allocated_gib=50.,
               peak_reserved_gib=55., status="ok")
    report = dict(protocol="m49_depth_real_accumulated_v1", variant="9x123",
                  status="ok", absolute_coordinate_embedding=True,
                  provenance=measured, gpu_total_gib=100., trials=[row],
                  selected=row, layers=9, schedule=[1,2,3])
    path = tmp_path/"benchmark.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(bench,"collect_runtime_provenance",lambda: runtime)
    monkeypatch.setattr(bench.torch.cuda,"get_device_properties",
                        lambda _: SimpleNamespace(total_memory=100*1024**3))
    bench.selection(path,"9x123")
    captured = capsys.readouterr()
    assert captured.out.strip() == "9 exact3 8 2 false"
    assert "historical measurements" in captured.err
