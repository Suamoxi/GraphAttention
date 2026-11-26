"""Read-only representative HDF5 and physical setup audit of archived forced HIT.

Recognize explicitly named primitive fields only. Unknown layouts remain
reported, not converted into training data. Physical measurements use stored
units; unit identity and axis semantics must be reviewed before a reader is
qualified. No exhaustive temporal decorrelation or deduplication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import h5py
import numpy as np

from graph_attention.utils.provenance import collect_runtime_provenance

CASES = ("case1_forced", "case2_forced", "case3_forced")
SPLITS = ("train", "val")
FILENAME = re.compile(r"data_(\d+(?:\.\d+)?)\.h5\Z")
VISCOSITY = re.compile(r"\s*lambda\s+T\s*:\s*([0-9.eE+-]+)\s*\*\s*T\s*\*\*\s*([0-9.eE+-]+)\s*")
GROUPS = ("primes", "primitives", "primitive", "GaseousPhase", "")
MAX_DATASET_BYTES = 16 * 1024**2


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def numeric_time(path):
    match = FILENAME.fullmatch(path.name)
    return float(match.group(1)) if match else None


def select_positions(count, spread, boundary=8):
    if count <= 0:
        raise ValueError("empty stream")
    return sorted(set(np.linspace(0, count-1, min(count, spread), dtype=int).tolist())
                  | set(range(min(boundary, count)))
                  | set(range(max(0, count-boundary), count)))


def read_setup(directory):
    result = {}
    for name in ("HIT.json", "numerical_setup.json"):
        path = directory/name
        raw = path.read_bytes()
        result[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                        "content": json.loads(raw)}
    return result


def value_preview(value):
    arr = np.asarray(value)
    if arr.size > 16:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "omitted": True}
    return str(value)[:2048]


def describe(h):
    datasets, attrs = {}, {}
    def visit(name, obj):
        if obj.attrs:
            attrs[name] = {str(k): value_preview(v) for k,v in obj.attrs.items()}
        if isinstance(obj, h5py.Dataset):
            info = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            if obj.size <= 16 and obj.dtype.itemsize * obj.size <= 65536:
                info["preview"] = value_preview(obj[()])
            # A coordinate axis is small; retain the entire axis for review.
            if obj.ndim == 1 and obj.size <= 512 and obj.dtype.kind in "fiu" and re.search(r"(grid|cell|coord|center|face|axis)", name, re.I):
                arr = np.asarray(obj[()], dtype=np.float64)
                if np.isfinite(arr).all():
                    info["coordinate_candidate_values"] = arr.tolist()
            datasets[name] = info
    visit("/", h)
    h.visititems(visit)
    return {"datasets": datasets, "attributes": attrs}


def resolve(h, names):
    candidates = []
    for group in GROUPS:
        for name in names:
            path = f"{group}/{name}" if group else name
            if isinstance(h.get(path), h5py.Dataset):
                candidates.append(path)
    if len(candidates) > 1:
        raise ValueError(f"ambiguous aliases {candidates}")
    return candidates[0] if candidates else None


def array(h, path):
    obj = h[path]
    if obj.size * obj.dtype.itemsize > MAX_DATASET_BYTES or obj.dtype.kind not in "fiu":
        raise ValueError(f"unsupported size or dtype for {path}")
    return np.asarray(obj[()], dtype=np.float64)


def scalar_stats(value):
    return {"mean": float(value.mean()), "std": float(value.std()),
            "min": float(value.min()), "max": float(value.max())}


def read_primitives(h):
    mapping = {
        "rho": resolve(h, ("density", "rho")),
        "p": resolve(h, ("pressure", "p")),
        "T": resolve(h, ("temperature", "T")),
    }
    component_paths = [resolve(h, names) for names in (
        ("velocityX", "velocity_x", "u"), ("velocityY", "velocity_y", "v"),
        ("velocityZ", "velocity_z", "w"))]
    vector = resolve(h, ("velocity",))
    if not mapping["rho"] or not mapping["p"]:
        raise ValueError("explicitly named density/pressure arrays not found")
    rho, p = array(h, mapping["rho"]), array(h, mapping["p"])
    if rho.ndim != 3 or rho.shape != p.shape:
        raise ValueError(f"expected aligned rank-3 scalar fields: {rho.shape}, {p.shape}")
    temperature = array(h, mapping["T"]) if mapping["T"] else None
    if temperature is not None and temperature.shape != rho.shape:
        raise ValueError("temperature shape differs")
    if all(component_paths):
        v = np.stack([array(h, name) for name in component_paths], axis=-1)
        mapping["velocity"] = component_paths
        mapping["velocity_component_semantics"] = "explicitly named x/y/z"
    elif not any(component_paths) and vector:
        v = array(h, vector)
        axes = [i for i in range(v.ndim) if v.shape[i] == 3
                and v.shape[:i]+v.shape[i+1:] == rho.shape]
        if len(axes) != 1:
            raise ValueError(f"ambiguous velocity vector axis: {v.shape}")
        v = np.moveaxis(v, axes[0], -1)
        mapping["velocity"] = vector
        mapping["velocity_component_axis"] = axes[0]
        mapping["velocity_component_semantics"] = "vector order not independently verified"
    else:
        raise ValueError("velocity components missing or ambiguous")
    if v.shape != (*rho.shape,3):
        raise ValueError("velocity and scalar fields are not aligned")
    return rho, p, temperature, v, mapping


def diagnostics(rho, p, temperature, velocity, setup):
    cfg = setup["HIT.json"]["content"]
    material = cfg["material_properties"]
    gamma = float(material["specific_heat_ratio"])
    gas = float(material["specific_gas_constant"])
    if material["type"] != "IdealGas" or gamma <= 1 or gas <= 0:
        raise ValueError("unqualified thermodynamic model")
    shape = tuple(int(cfg["domain"][axis]["cells"]) for axis in "xyz")
    lengths = [float(cfg["domain"][axis]["range"][1]-cfg["domain"][axis]["range"][0])
               for axis in "xyz"]
    arrays = [rho,p,velocity] + ([temperature] if temperature is not None else [])
    finite = all(bool(np.isfinite(a).all()) for a in arrays)
    record = {"shape": list(rho.shape), "configured_shape_xyz": list(shape),
              "shape_matches_configured_sizes": sorted(rho.shape)==sorted(shape),
              "finite": finite, "nonpositive_density_fraction": float(np.mean(rho<=0)),
              "nonpositive_pressure_fraction": float(np.mean(p<=0)),
              "nonpositive_temperature_fraction": float(np.mean(temperature<=0)) if temperature is not None else None}
    if not finite or np.any(rho<=0) or np.any(p<=0):
        return record
    v = velocity.reshape(-1,3)
    fluct = v-v.mean(axis=0)
    rms_vector = float(np.sqrt(np.mean(np.sum(fluct**2,axis=1))))
    rms_component = rms_vector/math.sqrt(3)
    a_rms = float(np.sqrt(np.mean(gamma*p/rho)))
    t_eos = p/(rho*gas)
    result = {"density":scalar_stats(rho),"pressure":scalar_stats(p),
              "temperature_from_eos":scalar_stats(t_eos),
              "velocity_mean_components":v.mean(axis=0).tolist(),
              "velocity_std_components":v.std(axis=0).tolist(),
              "fluctuation_tke":.5*rms_vector**2,
              "u_rms_one_component":rms_component, "u_rms_vector":rms_vector,
              "rms_sound_speed":a_rms, "measured_turbulent_mach_vector":rms_vector/a_rms,
              "measured_turbulent_mach_one_component":rms_component/a_rms}
    if temperature is not None:
        result["stored_temperature"] = scalar_stats(temperature)
        result["eos_relative_l2_temperature_error"] = float(
            np.linalg.norm(temperature-t_eos)/max(np.linalg.norm(t_eos),1e-30))
    # Do not evaluate executable expressions from the data archive. Recognize
    # only the observed mu=A*T**b form and keep its derived Re exploratory.
    match = VISCOSITY.fullmatch(str(material.get("dynamic_viscosity","")))
    if match and min(lengths)>0:
        coefficient, exponent = map(float,match.groups())
        if coefficient > 0 and np.isfinite([coefficient,exponent]).all():
            mu = coefficient*t_eos**exponent
            if np.isfinite(mu).all() and np.all(mu>0):
                result["mean_dynamic_viscosity_from_eos"] = float(mu.mean())
                result["exploratory_Re_box"] = float(rho.mean()*rms_component*min(lengths)/mu.mean())
    q = np.concatenate([rho[...,None],rho[...,None]*velocity,
         (p/(gamma-1)+.5*rho*np.sum(velocity**2,axis=-1))[...,None]],axis=-1)
    result["reconstructed_conservative_mean"] = q.reshape(-1,5).mean(axis=0).tolist()
    result["reconstructed_conservative_std"] = q.reshape(-1,5).std(axis=0).tolist()
    # Hash only explicitly mapped ordered physical arrays, not filename or time.
    digest = hashlib.sha256()
    for field in arrays:
        a = np.ascontiguousarray(field)
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    record.update(result)
    record["mapped_payload_sha256"] = digest.hexdigest()
    return record


def inspect(path, setup):
    record = {"path":str(path), "filename_time":numeric_time(path)}
    try:
        with h5py.File(path,"r") as h:
            schema = describe(h)
            record["schema"] = schema
            shape_dtype = {k:[v["shape"],v["dtype"]] for k,v in schema["datasets"].items()}
            record["schema_sha256"] = hashlib.sha256(json.dumps(shape_dtype,sort_keys=True).encode()).hexdigest()
            rho,p,T,v,mapping = read_primitives(h)
            record["field_mapping"] = mapping
            record["diagnostics"] = diagnostics(rho,p,T,v,setup)
            record["status"] = "mapped"
    except Exception as exc:
        record.update(status="unqualified", error=repr(exc))
    return record


def audit(root, case):
    report = {"case":case,"root":str(root),"status":"complete", "streams":{}}
    for split in SPLITS:
        directory = root/split/case/"HIT"
        setup = read_setup(directory)
        paths = list((directory/"domain").glob("data_*.h5"))
        unknown = [str(p) for p in paths if numeric_time(p) is None]
        paths = sorted((p for p in paths if numeric_time(p) is not None),key=numeric_time)
        times = np.array([numeric_time(p) for p in paths])
        if len(times)<2:
            raise ValueError(f"insufficient snapshots in {directory}")
        positions = select_positions(len(paths),33 if split=="train" else 17)
        records = []
        for j in positions:
            item = inspect(paths[j], setup)
            item["position"] = j
            records.append(item)
            print(f"{case}/{split} {j}/{len(paths)} {item['status']}", flush=True)
        # Keep full schema for first occurrence of each layout; preserve
        # scalar metadata/coordinates for selected snapshots individually.
        schemas = {}
        for item in records:
            schema = item.pop("schema",None)
            key = item.get("schema_sha256")
            if schema and key:
                schemas.setdefault(key,schema)
                item["small_dataset_metadata"] = {
                    k:v["preview"] for k,v in schema["datasets"].items() if "preview" in v}
        dt = np.diff(times)
        report["streams"][split] = {"directory":str(directory),"setup":setup,
            "snapshot_count":len(paths),"unrecognized_data_names":unknown,
            "filename_time_min":float(times.min()),"filename_time_max":float(times.max()),
            "filename_time_interval_stats":scalar_stats(dt),
            "repeated_filename_time_count":int(np.count_nonzero(dt==0)),
            "schemas":schemas,"status_counts":dict(Counter(r["status"] for r in records)),
            "representative_snapshots":records}
    train, val = report["streams"]["train"], report["streams"]["val"]
    a = train["representative_snapshots"][-1]
    b = val["representative_snapshots"][0]
    ah = a.get("diagnostics",{}).get("mapped_payload_sha256")
    bh = b.get("diagnostics",{}).get("mapped_payload_sha256")
    report["train_val_boundary"] = {
        "filename_time_gap":val["filename_time_min"]-train["filename_time_max"],
        "last_train_first_val_same_mapped_payload":ah==bh if ah and bh else None,
        "trajectory_independence":"not established"}
    report["provenance"] = collect_runtime_provenance()
    report["interpretation"] = {
        "units":"stored units; configuration and HDF5 metadata require review before declaring physical reference units",
        "mach":"rms fluctuation speed divided by sqrt(mean(gamma*p/rho)); both vector and one-component definitions",
        "Re_box":"mean(rho) * one-component fluctuation rms * smallest configured box length / mean(mu(T_eos)); exploratory, not Taylor Reynolds",
        "time":"numeric filename labels, inspect HDF metadata for actual times",
        "mapping":"explicit candidate field names; vector order and spatial axis orientation require review",
        "sampling":"time-spread plus 8 consecutive samples at each endpoint; not a full stationarity or deduplication audit"}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("/archivenfs/phlusim/HIT_LES_FORCED"))
    p.add_argument("--case",choices=CASES,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=audit(args.root,args.case)
    atomic_json(args.output,report)
    print(f"Report={args.output}",flush=True)


if __name__ == "__main__":
    main()
