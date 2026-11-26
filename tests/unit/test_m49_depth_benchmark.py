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
