"""Phase 5C — pure-python analysis helpers for the VoxCPM2 LoRA audit.

Everything here is stdlib-only and importable from the main env so the
logic is unit-testable without torch/CUDA:

- safetensors header parsing (keys/shapes/dtype — no tensor values)
- LoRA key classification (lm/dit/proj, A/B)
- step-to-step weight-diff evaluation rules
- load-integrity verdict rules (loaded/skipped/missing keys)
- Phase 5B manifest hash re-verification
- final verdict enumeration (ADAPTER_*/COMPARABILITY_*/INCONCLUSIVE)

The env-side runner (scripts/lora/audit_voxcpm_lora_effect.py) collects raw
numbers (tensor norms, loaded/skipped key lists, generated WAV shas) and
feeds them here for judgement.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

VERDICTS = (
    "ADAPTER_NOT_EFFECTIVE",
    "ADAPTER_EFFECTIVE_NO_CLEAR_GAIN",
    "COMPARABILITY_LIMITED",
    "INCONCLUSIVE",
)

_SAFE_DTYPES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
    "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def safetensors_header(path: str | Path) -> dict:
    """Parse a .safetensors header without loading tensor data.

    Returns {key: {"dtype": str, "shape": list[int], "numel": int}}.
    """
    p = Path(path)
    with p.open("rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        numel = 1
        for d in meta["shape"]:
            numel *= d
        out[key] = {"dtype": meta["dtype"], "shape": meta["shape"], "numel": numel}
    return out


def classify_lora_key(key: str) -> dict:
    """Classify a LoRA weight key into group (lm/dit/proj/other) and matrix (A/B/other)."""
    if ".lora_A" in key:
        matrix = "A"
    elif ".lora_B" in key:
        matrix = "B"
    else:
        matrix = "other"
    if "base_lm." in key or "residual_lm." in key:
        group = "lm"
    elif "feat_decoder." in key:
        group = "dit"
    elif any(p in key for p in ("_proj", "fusion_concat")):
        group = "proj"
    else:
        group = "other"
    return {"group": group, "matrix": matrix}


def weights_census(header: dict, tensor_stats: dict | None = None) -> dict:
    """Aggregate a safetensors header (+ optional per-key stats) into a census.

    tensor_stats: {key: {"l2": float, "nonzero": float, "nan": int, "inf": int}}
    """
    groups: dict[str, dict] = {}
    for key, meta in header.items():
        cls = classify_lora_key(key)
        g = groups.setdefault(cls["group"], {
            "keys": 0, "numel": 0, "A": 0, "B": 0, "dtypes": set(),
            "l2": 0.0, "nonzero_mean": [], "anomalies": [],
        })
        g["keys"] += 1
        g["numel"] += meta["numel"]
        g["dtypes"].add(meta["dtype"])
        if cls["matrix"] in ("A", "B"):
            g[cls["matrix"]] += 1
        if tensor_stats and (st := tensor_stats.get(key)):
            g["l2"] += st.get("l2", 0.0) ** 2
            g["nonzero_mean"].append(st.get("nonzero", 0.0))
            if st.get("nan") or st.get("inf"):
                g["anomalies"].append(
                    {"key": key, "nan": st.get("nan", 0), "inf": st.get("inf", 0)})
            if st.get("l2", 1.0) == 0.0:
                g["anomalies"].append({"key": key, "all_zero": True})
    for g in groups.values():
        g["dtypes"] = sorted(g["dtypes"])
        g["l2"] = g["l2"] ** 0.5 if g["l2"] else 0.0
        g["nonzero_mean"] = (sum(g["nonzero_mean"]) / len(g["nonzero_mean"])
                             if g["nonzero_mean"] else None)
    return groups


def evaluate_weight_diffs(diffs: dict, key_groups: dict) -> dict:
    """Judge step-to-step diffs collected env-side.

    diffs: {"step_a->step_b": {key: {"max_abs": f, "rel_l2": f}}}
    Returns per-transition summary with distinct/changed/unchanged counts.
    """
    out = {}
    for trans, per_key in diffs.items():
        changed = unchanged = 0
        max_rel = 0.0
        by_group: dict[str, dict] = {}
        for key, d in per_key.items():
            grp = classify_lora_key(key)["group"]
            bg = by_group.setdefault(grp, {"changed": 0, "unchanged": 0, "max_rel": 0.0})
            rel = d.get("rel_l2", 0.0)
            if d.get("identical") or rel == 0.0:
                unchanged += 1
                bg["unchanged"] += 1
            else:
                changed += 1
                bg["changed"] += 1
            max_rel = max(max_rel, rel)
            bg["max_rel"] = max(bg["max_rel"], rel)
        out[trans] = {
            "changed": changed, "unchanged": unchanged,
            "max_rel_l2": round(max_rel, 6), "by_group": by_group,
            "trained": changed > 0,
        }
    return out


def evaluate_load_integrity(ckpt_keys: list[str], loaded: list[str],
                            skipped: list[str],
                            required_groups=("lm", "dit")) -> dict:
    """Apply PLAN 26.4 acceptance rules to a load_lora_weights result."""
    ckpt_set, loaded_set, skipped_set = set(ckpt_keys), set(loaded), set(skipped)
    missing = sorted(ckpt_set - loaded_set - skipped_set)
    unexplained_skipped = sorted(skipped_set & ckpt_set)
    unexpected_skipped = sorted(skipped_set - ckpt_set)
    loaded_groups = {classify_lora_key(k)["group"] for k in loaded}
    missing_groups = [g for g in required_groups if g not in loaded_groups]
    ok = (bool(loaded) and not missing and not unexplained_skipped
          and not missing_groups)
    return {
        "ok": ok,
        "loaded": len(loaded), "skipped": len(skipped),
        "ckpt_keys": len(ckpt_set),
        "missing_keys": missing,
        "unexplained_skipped": unexplained_skipped,
        "unexpected_skipped": unexpected_skipped,
        "loaded_groups": sorted(loaded_groups),
        "missing_required_groups": missing_groups,
        "verdict": "LOAD_OK" if ok else "ADAPTER_LOAD_FAILED",
    }


def evaluate_onoff(a_sha: str | None, b_sha: str | None, c_sha: str | None,
                   crep_sha: str | None, diffs: dict) -> dict:
    """Verdict for the minimal ON/OFF experiment (PLAN 26.5).

    diffs: {"A_vs_B": {...}, "B_vs_C": {...}, "C_vs_Crep": {...}} each with
    keys like corr/rmse/duration_s.
    """
    result = {"checks": {}, "verdict": None}
    result["checks"]["A_eq_B"] = a_sha is not None and a_sha == b_sha
    result["checks"]["C_eq_Crep"] = c_sha is not None and c_sha == crep_sha
    result["checks"]["B_neq_C"] = b_sha is not None and c_sha is not None and b_sha != c_sha
    # Waveform-level evidence where shas alone can't distinguish near-equal.
    bc = diffs.get("B_vs_C") or {}
    ab = diffs.get("A_vs_B") or {}
    result["checks"]["B_vs_C_corr"] = bc.get("corr")
    result["checks"]["A_vs_B_corr"] = ab.get("corr")

    if not result["checks"]["C_eq_Crep"]:
        result["verdict"] = "NONDETERMINISTIC_RUN"
    elif not result["checks"]["A_eq_B"]:
        # Disabled adapter should reproduce base exactly; flag if not.
        corr = ab.get("corr")
        result["verdict"] = ("DISABLED_NOT_EQ_BASE"
                             if (corr is None or corr < 0.9999)
                             else "DISABLED_NEAR_BASE_ONLY")
    elif not result["checks"]["B_neq_C"]:
        result["verdict"] = "ADAPTER_NOT_EFFECTIVE"
    else:
        corr = bc.get("corr")
        if corr is not None and corr > 0.99999:
            result["verdict"] = "ADAPTER_NUMERIC_NOISE_ONLY"
        else:
            result["verdict"] = "ADAPTER_EFFECTIVE"
    return result


def verify_manifest_outputs(manifest_path: str | Path,
                            sha_of) -> dict:
    """Re-hash every output in a Phase 5B manifest vs current disk state.

    sha_of: callable(path) -> sha256 or None.
    """
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = []
    ok = True
    for c in m.get("cases", []):
        out = c.get("output")
        p = Path(manifest_path).parent / out if out else None
        actual = sha_of(p) if p else None
        expect = c.get("output_sha256")
        match = actual is not None and actual == expect
        ok = ok and match
        rows.append({"case_id": c.get("case_id"), "expect": expect,
                     "actual": actual, "match": match,
                     "exists": actual is not None})
    return {"experiment_id": m.get("experiment_id"), "all_match": ok,
            "verified": sum(1 for r in rows if r["match"]),
            "total": len(rows), "rows": rows}


def compare_generation_args(base_args: dict, lora_args: dict) -> dict:
    """Diff effective generation args between base and LoRA cells.

    Keys compared: the frozen Phase 5B condition set.
    """
    keys = ["text", "prompt_text", "prompt_wav_path", "cfg_value",
            "inference_timesteps", "normalize", "denoise", "retry_badcase"]
    diffs = {}
    for k in keys:
        bv, lv = base_args.get(k), lora_args.get(k)
        if k == "prompt_wav_path" and bv and lv:
            # Same file may be spelled differently; compare resolved basename+sha elsewhere.
            bv, lv = Path(str(bv)).name, Path(str(lv)).name
        if bv != lv:
            diffs[k] = {"base": bv, "lora": lv}
    return {"comparable": not diffs, "diffs": diffs}


def final_verdict(load: dict, weights: dict, onoff: dict,
                  comparability: dict) -> str:
    """Reduce all audit sections to one PLAN 26.7 verdict."""
    if not load.get("ok"):
        return "ADAPTER_NOT_EFFECTIVE"
    if not weights.get("trained"):
        return "ADAPTER_NOT_EFFECTIVE"
    ov = (onoff or {}).get("verdict")
    if ov in ("NONDETERMINISTIC_RUN", "DISABLED_NOT_EQ_BASE"):
        return "COMPARABILITY_LIMITED"
    if ov == "ADAPTER_EFFECTIVE":
        return ("ADAPTER_EFFECTIVE_NO_CLEAR_GAIN"
                if comparability.get("ok")
                else "COMPARABILITY_LIMITED")
    if ov in ("ADAPTER_NOT_EFFECTIVE", "ADAPTER_NUMERIC_NOISE_ONLY"):
        return "ADAPTER_NOT_EFFECTIVE"
    return "INCONCLUSIVE"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
