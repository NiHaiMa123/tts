"""Tests for Phase 5C LoRA effect-audit pure logic (no GPU required)."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from character_tts.evaluation.lora_audit import (  # noqa: E402
    classify_lora_key, compare_generation_args, evaluate_load_integrity,
    evaluate_onoff, evaluate_weight_diffs, final_verdict,
    safetensors_header, verify_manifest_outputs, weights_census,
)


def _write_safetensors(path: Path, keys):
    header = {k: {"dtype": "BF16", "shape": [16, 8],
                  "data_offsets": [0, 256]} for k in keys}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_safetensors_header_and_classify(tmp_path):
    p = tmp_path / "w.safetensors"
    keys = ["base_lm.layers.0.self_attn.q_proj.lora_A",
            "base_lm.layers.0.self_attn.q_proj.lora_B",
            "residual_lm.layers.3.self_attn.v_proj.lora_A",
            "feat_decoder.estimator.decoder.layers.2.self_attn.k_proj.lora_B",
            "enc_to_lm_proj.lora_A"]
    _write_safetensors(p, keys)
    h = safetensors_header(p)
    assert h[keys[0]]["shape"] == [16, 8] and h[keys[0]]["numel"] == 128
    assert classify_lora_key(keys[0]) == {"group": "lm", "matrix": "A"}
    assert classify_lora_key(keys[1]) == {"group": "lm", "matrix": "B"}
    assert classify_lora_key(keys[3]) == {"group": "dit", "matrix": "B"}
    assert classify_lora_key(keys[4])["group"] == "proj"


def test_weights_census_groups_and_anomalies(tmp_path):
    p = tmp_path / "w.safetensors"
    keys = ["base_lm.layers.0.self_attn.q_proj.lora_A",
            "base_lm.layers.0.self_attn.q_proj.lora_B",
            "feat_decoder.estimator.decoder.layers.0.self_attn.q_proj.lora_A"]
    _write_safetensors(p, keys)
    stats = {keys[0]: {"l2": 3.0, "nonzero": 1.0, "nan": 0, "inf": 0},
             keys[1]: {"l2": 4.0, "nonzero": 0.5, "nan": 0, "inf": 0},
             keys[2]: {"l2": 0.0, "nonzero": 0.0, "nan": 0, "inf": 0}}
    g = weights_census(safetensors_header(p), stats)
    assert g["lm"]["keys"] == 2 and g["lm"]["A"] == 1 and g["lm"]["B"] == 1
    assert abs(g["lm"]["l2"] - 5.0) < 1e-6
    assert g["dit"]["anomalies"] == [{"key": keys[2], "all_zero": True}]


def test_load_integrity_rules():
    ckpt_keys = ["base_lm.layers.0.self_attn.q_proj.lora_A",
                 "feat_decoder.estimator.decoder.layers.0.self_attn.q_proj.lora_B"]
    ok = evaluate_load_integrity(ckpt_keys, ckpt_keys, [])
    assert ok["ok"] and ok["verdict"] == "LOAD_OK"
    # all skipped -> fail
    bad = evaluate_load_integrity(ckpt_keys, [], ckpt_keys)
    assert not bad["ok"] and bad["verdict"] == "ADAPTER_LOAD_FAILED"
    assert bad["unexplained_skipped"] == sorted(ckpt_keys)
    # lm loaded but dit missing -> fail (required group coverage)
    lm_only = evaluate_load_integrity(ckpt_keys, [ckpt_keys[0]],
                                      [ckpt_keys[1]])
    assert not lm_only["ok"] and "dit" in lm_only["missing_required_groups"]
    # key missing entirely (not loaded, not skipped) -> fail
    missing = evaluate_load_integrity(ckpt_keys, [ckpt_keys[0]], [])
    assert not missing["ok"] and missing["missing_keys"] == [ckpt_keys[1]]


def test_onoff_verdicts():
    eff = evaluate_onoff("a", "a", "b", "b",
                         {"B_vs_C": {"corr": 0.1}, "A_vs_B": {"corr": 1.0}})
    assert eff["verdict"] == "ADAPTER_EFFECTIVE"
    dead = evaluate_onoff("a", "a", "a", "a",
                          {"B_vs_C": {"corr": 1.0}, "A_vs_B": {"corr": 1.0}})
    assert dead["verdict"] == "ADAPTER_NOT_EFFECTIVE"
    nond = evaluate_onoff("a", "a", "b", "c", {})
    assert nond["verdict"] == "NONDETERMINISTIC_RUN"
    drift = evaluate_onoff("a", "b", "c", "c",
                           {"A_vs_B": {"corr": 0.5}})
    assert drift["verdict"] == "DISABLED_NOT_EQ_BASE"
    noise = evaluate_onoff("a", "a", "b", "b",
                           {"B_vs_C": {"corr": 0.9999999},
                            "A_vs_B": {"corr": 1.0}})
    assert noise["verdict"] == "ADAPTER_NUMERIC_NOISE_ONLY"


def test_weight_diffs_and_final_verdict():
    diffs = {"ckpt50->ckpt100": {
        "base_lm.x.lora_A": {"rel_l2": 0.5, "identical": False},
        "feat_decoder.y.lora_B": {"rel_l2": 0.0, "identical": True}}}
    ev = evaluate_weight_diffs(diffs, None)
    e = ev["ckpt50->ckpt100"]
    assert e["changed"] == 1 and e["unchanged"] == 1 and e["trained"]
    assert final_verdict({"ok": True}, {"trained": True},
                         {"verdict": "ADAPTER_EFFECTIVE"},
                         {"ok": True}) == "ADAPTER_EFFECTIVE_NO_CLEAR_GAIN"
    assert final_verdict({"ok": False}, {"trained": True},
                         {"verdict": "ADAPTER_EFFECTIVE"},
                         {"ok": True}) == "ADAPTER_NOT_EFFECTIVE"
    assert final_verdict({"ok": True}, {"trained": True},
                         {"verdict": "NONDETERMINISTIC_RUN"},
                         {"ok": True}) == "COMPARABILITY_LIMITED"
    assert final_verdict({"ok": True}, {"trained": True},
                         {"verdict": "ADAPTER_EFFECTIVE"},
                         {"ok": False}) == "COMPARABILITY_LIMITED"
    assert final_verdict({"ok": True}, {"trained": True},
                         {"verdict": "ADAPTER_NOT_EFFECTIVE"},
                         {"ok": True}) == "ADAPTER_NOT_EFFECTIVE"


def test_manifest_reverify_and_args_compare(tmp_path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFFfake")
    man = {"experiment_id": "e1",
           "cases": [{"case_id": "base/x", "output": "x.wav",
                      "output_sha256": "WRONG"}]}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    import hashlib
    real = hashlib.sha256(b"RIFFfake").hexdigest()
    v = verify_manifest_outputs(mp, lambda p: real if Path(p).is_file() else None)
    assert not v["all_match"] and v["rows"][0]["actual"] == real
    c = compare_generation_args(
        {"text": "t", "cfg_value": 2.0, "inference_timesteps": 10,
         "normalize": True, "denoise": False, "retry_badcase": True},
        {"text": "t", "cfg_value": 2.0, "inference_timesteps": 10,
         "normalize": True, "denoise": False, "retry_badcase": True})
    assert c["comparable"]
    c2 = compare_generation_args(
        {"text": "t", "cfg_value": 2.0, "retry_badcase": True},
        {"text": "t", "cfg_value": 3.0, "retry_badcase": True})
    assert not c2["comparable"] and "cfg_value" in c2["diffs"]
