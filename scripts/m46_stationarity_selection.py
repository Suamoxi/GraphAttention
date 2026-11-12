"""M46 eligibility: select reviewed stationary-candidate snapshots, not the full inventory.

This is an analysis-selection policy, not a proof of statistical stationarity.
Unreviewed cases remain in the inventory but cannot enter the primary comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REGULAR = re.compile(r"solut_hit_[0-9]{8}\Z")
REQUIRED = {"first_counter", "last_counter", "transient_end_counter", "require_full_window", "evidence"}


def select_snapshots(inventory: dict, policy: dict) -> tuple[dict, list[dict], dict]:
    if policy.get("version") != "m46_stationary_eligibility_v1":
        raise ValueError("Unsupported M46 eligibility policy version")
    rules = policy.get("cases")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("M46 requires explicit reviewed case rules")
    case_by_id = {case["case_id"]: case for case in inventory["cases"]}
    unknown = set(rules) - set(case_by_id)
    if unknown:
        raise ValueError(f"Reviewed rules refer to absent cases: {sorted(unknown)}")
    for case_id, rule in rules.items():
        if not isinstance(rule, dict) or set(rule) != REQUIRED:
            raise ValueError(f"{case_id}: invalid reviewed eligibility rule keys")
        first, last, transient = (rule[k] for k in ("first_counter", "last_counter", "transient_end_counter"))
        if any(type(value) is not int for value in (first, last, transient)) or not (0 <= transient < first <= last):
            raise ValueError(f"{case_id}: invalid reviewed interval boundaries")
        if type(rule["require_full_window"]) is not bool:
            raise ValueError(f"{case_id}: require_full_window must be boolean")
        if not isinstance(rule["evidence"], str) or len(rule["evidence"].strip()) < 20:
            raise ValueError(f"{case_id}: reviewed evidence provenance required")

    selected, rows = [], []
    accepted_counters = {case_id: [] for case_id in rules}
    for sample in inventory["samples"]:
        case_id = sample["case_id"]
        if case_id not in case_by_id:
            raise ValueError(f"Unknown case for snapshot {sample['sample_id']}")
        counter = sample.get("counter")
        rule = rules.get(case_id)
        if rule is None:
            status = "pending_case_qualification"
        elif counter is None:
            status = "pending_unnumbered_snapshot"
        elif type(counter) is not int:
            raise ValueError(f"{sample['sample_id']}: invalid counter")
        elif counter <= rule["transient_end_counter"]:
            status = "excluded_transient"
        elif counter < rule["first_counter"]:
            status = "excluded_transition_uncertain"
        elif counter > rule["last_counter"]:
            status = "outside_reviewed_interval"
        elif REGULAR.fullmatch(Path(sample["snapshot"]).stem) is None:
            status = "pending_noncanonical_filename"
        else:
            status = "selected_exploratory_stationary"
            selected.append(sample)
            accepted_counters[case_id].append(counter)
        rows.append({
            "sample_id": sample["sample_id"], "case_id": case_id,
            "source": sample["source"], "snapshot": sample["snapshot"],
            "counter": counter, "eligibility": status,
            "included": status == "selected_exploratory_stationary",
        })

    for case_id, rule in rules.items():
        counters = accepted_counters[case_id]
        if not counters:
            raise ValueError(f"{case_id}: reviewed rule selects no snapshots")
        if rule["require_full_window"]:
            expected = set(range(rule["first_counter"], rule["last_counter"] + 1))
            counts = Counter(counters)
            missing = sorted(expected - set(counts))
            repeated = {counter: count for counter, count in counts.items() if count != 1}
            if missing or repeated:
                raise ValueError(f"{case_id}: incomplete reviewed window; missing={missing}, repeated={repeated}")

    qualified_cases = [c for c in inventory["cases"] if c["case_id"] in rules]
    status_counts = Counter(row["eligibility"] for row in rows)
    pending = sorted(set(case_by_id) - set(rules))
    summary = {
        "version": "m46_selection_summary_v1",
        "stationarity_filter": True,
        "interpretation": "M43-reviewed exploratory stationary candidates, not strict stationarity certification",
        "discovered_cases": len(inventory["cases"]),
        "discovered_snapshots": len(inventory["samples"]),
        "selected_cases": len(qualified_cases),
        "selected_snapshots": len(selected),
        "pending_cases": pending,
        "status_counts": dict(sorted(status_counts.items())),
    }
    qualified = {
        "version": "m46_qualified_manifest_v1",
        "stationarity_filter": True,
        "qualification": {"policy_version": policy["version"], "rules": rules, "pending_cases": pending},
        "cases": qualified_cases, "samples": selected,
        "counts": {"cases": len(qualified_cases), "snapshots": len(selected)},
    }
    return qualified, rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    policy = json.loads(args.policy.read_text())
    qualified, rows, summary = select_snapshots(inventory, policy)
    qualified["qualification"]["source_inventory_sha256"] = hashlib.sha256(args.inventory.read_bytes()).hexdigest()
    qualified["qualification"]["policy_sha256"] = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, obj in (("qualified_manifest.json", qualified), ("eligibility_summary.json", summary)):
        dest = args.output / filename
        temporary = dest.with_suffix(dest.suffix + ".tmp")
        temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
        temporary.replace(dest)
    with (args.output / "snapshot_eligibility.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "sample_id", "case_id", "source", "snapshot", "counter", "eligibility", "included"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()