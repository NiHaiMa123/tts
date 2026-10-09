"""Freeze the Phase 5B LoRA-pilot training set (PLAN.md section 25.3).

Reads the legacy suoming *train* split only, verifies every wav
(existence, duration, sampling, speech sanity), enforces leakage
blocks against frozen eval assets (P0/P2 prompts, ground truth, all
evaluation texts — by exact audio sha256, exact normalized text, and
near-duplicate text similarity), and emits:

    <out>/train.jsonl          official {"audio","text"} rows
    <out>/val_metrics.jsonl    small validation-split set for loss logs
    <out>/audit/audit.jsonl    per-record decision + provenance
    <out>/audit/rejects.json   rejected rows grouped by reason
    <out>/audit/blocklist.json the leakage blocklist actually applied
    <out>/audit/stats.json     totals, duration histogram, decision

Exits non-zero with DATA_BLOCKED when accepted data is too small.
No private paths leak into committed artifacts — the JSONL itself
stays local under outputs/ (gitignored).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from character_tts.audio.io import read_wav  # noqa: E402
from character_tts.evaluation.phase5a import load_phase5a_config  # noqa: E402
from character_tts.registry.loader import load_character, repo_root  # noqa: E402

DATA_V1 = Path("data/characters/suoming/datasets/v1")
MIN_ACCEPT = 30          # rows
MIN_ACCEPT_SECONDS = 120.0
MIN_DUR, MAX_DUR = 0.4, 90.0
SIM_REJECT = 0.85        # near-dup text similarity vs blocklist
VAL_ROWS = 8             # rows reserved for val-loss logging only
# suoming 的 phase5a/5a2 评测有独立冻结配置；其他角色以角色 yaml 的
# evaluation.anchor_texts 为评测资产。
DEFAULT_EVAL_CONFIGS = {
    "suoming": ("suoming_voxcpm_phase5a", "suoming_voxcpm_phase5a2"),
}

_PUNCT = re.compile(r"[\s，。！？、；：「」『』（）《》…—\-,.!?;:\"'~·]")


def _resolve_repo(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else repo_root() / path


def _norm_text(t: str) -> str:
    return _PUNCT.sub("", t or "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wav_stats(path: Path) -> dict:
    info = {"ok": True}
    try:
        data, sr = read_wav(path)
    except Exception as exc:
        return {"ok": False, "reason": f"unreadable: {exc}"}
    import numpy as np
    x = np.asarray(data, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    info.update({
        "sample_rate": int(sr),
        "duration_s": round(len(x) / sr, 3),
        "finite": bool(np.isfinite(x).all()),
        "peak": round(float(np.abs(x).max()), 4) if len(x) else 0.0,
        "rms_dbfs": round(20 * math.log10(
            max(float(np.sqrt((x ** 2).mean())), 1e-12)), 2)
        if len(x) else -999.0,
    })
    if not info["finite"]:
        info.update(ok=False, reason="non-finite samples")
    return info


def build_blocklist(character_id: str = "suoming") -> dict:
    """Leakage blocklist from frozen eval assets (audio sha + text)."""
    shas, texts, fids, labels = set(), set(), set(), []

    def add(label, audio=None, sha=None, text=None, fid=None):
        if sha:
            shas.add(sha.lower())
        if fid:
            fids.add(fid)
        if text:
            texts.add(_norm_text(text))
        labels.append({"label": label, "sha256": sha, "fid": fid,
                       "text": text})

    character = load_character(character_id)
    gt = character.ground_truth or {}
    add("ground_truth_rain", audio=gt.get("audio"),
        sha=gt.get("sha256"), text=gt.get("text"))
    add("prompt_P0", audio=character.reference_audio,
        sha=character.reference_sha256,
        text=character.reference.get("text"))

    # 角色 yaml 的评测锚点（含 ground truth 外的其余锚点）
    for a in character.anchor_texts:
        add(f"anchor_{a.id}", audio=a.audio, sha=a.sha256, text=a.text,
            fid=(a.source or "").split("fid=")[-1] or None)

    for cfg_name in DEFAULT_EVAL_CONFIGS.get(character_id, ()):
        cfg = load_phase5a_config(cfg_name)
        for p in cfg.get("prompts") or []:
            add(f"prompt_{p['id']}", audio=p.get("audio"),
                sha=p.get("sha256"), text=p.get("text"),
                fid=(p.get("source") or "").split("fid=")[-1] or None)
        for t in cfg.get("texts") or []:
            src = t.get("source") or ""
            add(f"eval_text_{t['id']}", audio=t.get("audio"),
                sha=t.get("sha256"), text=t.get("text"),
                fid=src.split("fid=")[-1] if "fid=" in src else None)

    return {"audio_sha256": sorted(shas), "norm_texts": sorted(texts),
            "fids": sorted(fids), "entries": labels}


def prepare(train_jsonl: Path, val_jsonl: Path, blocklist: dict,
            out_dir: Path, log=print) -> dict:
    out_dir = Path(out_dir)
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    block_shas = set(blocklist["audio_sha256"])
    block_texts = set(blocklist["norm_texts"])
    block_fids = set(blocklist["fids"])
    seen_shas, seen_texts = set(), set()

    accepted, rejected, audit_rows = [], [], []
    total_dur = 0.0

    def _reject(row, reason, detail=""):
        rejected.append({"reason": reason, "fid": row.get("fid"),
                         "text": row.get("text"), "detail": detail})
        audit_rows.append({"decision": "reject", "reason": reason,
                           "fid": row.get("fid"), "detail": detail})

    for i, line in enumerate(Path(train_jsonl).read_text(
            encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        audio = _resolve_repo(row["audio"])
        text = (row.get("text") or "").strip()
        fid = row.get("fid")

        if fid and fid in block_fids:
            _reject(row, "leak_fid", "fid matches frozen eval asset")
            continue
        if not audio.is_file():
            _reject(row, "missing_file", str(audio))
            continue
        sha = _sha256(audio)
        if sha.lower() in block_shas:
            _reject(row, "leak_audio_sha", sha[:16])
            continue
        if not text or len(_norm_text(text)) < 2:
            _reject(row, "bad_text", "empty or <2 normalized chars")
            continue
        nt = _norm_text(text)
        if nt in block_texts:
            _reject(row, "leak_text_exact", nt[:30])
            continue
        best = max((difflib.SequenceMatcher(None, nt, bt).ratio()
                    for bt in block_texts), default=0.0)
        if best >= SIM_REJECT:
            _reject(row, "leak_text_similar", f"sim={best:.2f}")
            continue
        if sha in seen_shas:
            _reject(row, "duplicate_audio", sha[:16])
            continue
        if nt in seen_texts:
            _reject(row, "duplicate_text", nt[:30])
            continue

        st = _wav_stats(audio)
        if not st.get("ok"):
            _reject(row, "bad_audio", st.get("reason", ""))
            continue
        if not (MIN_DUR <= st["duration_s"] <= MAX_DUR):
            _reject(row, "bad_duration",
                    f"{st['duration_s']}s outside [{MIN_DUR},{MAX_DUR}]")
            continue
        if st["rms_dbfs"] < -50:
            _reject(row, "silent", f"rms={st['rms_dbfs']}dBFS")
            continue

        accepted.append({"audio": str(audio), "text": text})
        total_dur += st["duration_s"]
        seen_shas.add(sha)
        seen_texts.add(nt)
        audit_rows.append({
            "decision": "accept", "fid": fid, "sha256": sha,
            "duration_s": st["duration_s"],
            "sample_rate": st["sample_rate"],
            "rms_dbfs": st["rms_dbfs"], "text": text,
        })

    (out_dir / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n"
                for r in accepted), encoding="utf-8")

    # Validation rows for loss logging only — same block rules, and
    # never trained on (they come from the untouched val split).
    val_rows, val_seen = [], set()
    for line in Path(val_jsonl).read_text(encoding="utf-8").splitlines():
        if len(val_rows) >= VAL_ROWS or not line.strip():
            continue
        row = json.loads(line)
        audio = _resolve_repo(row["audio"])
        nt = _norm_text(row.get("text") or "")
        fid = row.get("fid")
        if (fid and fid in block_fids) or nt in block_texts \
                or nt in val_seen or not audio.is_file():
            continue
        sha = _sha256(audio)
        if sha.lower() in block_shas:
            continue
        val_seen.add(nt)
        # Same columns as train.jsonl — HF datasets requires matching
        # schema across splits.
        val_rows.append({"audio": str(audio), "text": row["text"].strip()})
    (out_dir / "val_metrics.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n"
                for r in val_rows), encoding="utf-8")

    reason_counts = {}
    for r in rejected:
        reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1
    (audit_dir / "audit.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n"
                for r in audit_rows), encoding="utf-8")
    (audit_dir / "rejects.json").write_text(json.dumps(
        {"counts": reason_counts, "rows": rejected},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (audit_dir / "blocklist.json").write_text(json.dumps(
        {k: v for k, v in blocklist.items() if k != "entries"}
        | {"n_entries": len(blocklist["entries"])},
        ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "source_train_manifest": str(train_jsonl),
        "source_val_manifest": str(val_jsonl),
        "rows_read": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reject_reasons": reason_counts,
        "total_seconds": round(total_dur, 1),
        "mean_seconds": round(total_dur / max(1, len(accepted)), 2),
        "val_metrics_rows": len(val_rows),
        "decision": "ok",
    }
    if len(accepted) < MIN_ACCEPT or total_dur < MIN_ACCEPT_SECONDS:
        stats["decision"] = "DATA_BLOCKED"
    (audit_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[prep] accepted={stats['accepted']} rejected={stats['rejected']} "
        f"total={stats['total_seconds']}s val_rows={len(val_rows)} "
        f"-> {stats['decision']}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="suoming")
    ap.add_argument("--train-manifest", default=None,
                    help="default: 角色 yaml dataset.train_manifest")
    ap.add_argument("--val-manifest", default=None,
                    help="default: 角色 yaml dataset.validation_manifest")
    ap.add_argument("--out-dir", default=None,
                    help="default: outputs/training/<char>_voxcpm_lora_pilot")
    args = ap.parse_args()

    character = load_character(args.character)
    ds = character.dataset or {}
    train_manifest = (args.train_manifest
                      or ds.get("train_manifest")
                      or str(DATA_V1 / "train.jsonl"))
    val_manifest = (args.val_manifest
                    or ds.get("validation_manifest")
                    or str(DATA_V1 / "validation.jsonl"))
    out_dir = Path(args.out_dir
                   or f"outputs/training/{args.character}"
                      "_voxcpm_lora_pilot")
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    blocklist = build_blocklist(args.character)
    stats = prepare(_resolve_repo(train_manifest),
                    _resolve_repo(val_manifest),
                    blocklist, out_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats["decision"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
