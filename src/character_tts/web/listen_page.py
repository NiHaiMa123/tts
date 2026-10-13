"""Listening data builders using the shared local-file review template.

All generated pages use the same widgets and acknowledged autosave.
Serve the experiment with scripts/ingest/serve_review.py; choices are
written directly to listen/listen-ratings*.json. The Agent reads those
files when the user says continue. Subjective scores are never prefilled.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .listen_template import render_listen_page



def _fmt_params(sidecar: dict[str, Any] | None) -> str:
    if not sidecar:
        return ""
    parts: list[str] = []
    wr = sidecar.get("worker_result") or {}
    meta = wr.get("metadata") or {}
    gen = meta.get("generation") or {}
    for key in ("num_steps", "guidance_scale", "speaker_scale", "cfg_value",
                "inference_timesteps", "temperature", "top_k", "top_p",
                "ode_method", "variant", "language", "seed"):
        if key in gen:
            parts.append(f"{key}={gen[key]}")
    if sidecar.get("seed") is not None and "seed" not in gen:
        parts.append(f"seed={sidecar['seed']}")
    if meta.get("variant"):
        parts.append(f"variant={meta['variant']}")
    if meta.get("adapter"):
        parts.append(f"adapter={Path(str(meta['adapter'])).parent.name}")
    return html.escape(", ".join(dict.fromkeys(parts)))


def _top_level_role(name: str) -> tuple[str, str]:
    """(backend, kind) labels for wavs living at the output root."""
    if "ground_truth" in name:
        return "ground_truth", "原声(validation)"
    return "reference", "克隆参考音频"


def build_samples(output_dir: Path) -> dict[str, Any]:
    items = []
    wavs = sorted(
        (p for p in Path(output_dir).rglob("*.wav")
         if p.relative_to(output_dir).parts[0] != "listen"),
        # Originals first, then per-backend dirs — the listening flow
        # starts from the real utterances. Depth is judged on the path
        # *relative* to output_dir (p.parts counts absolute ancestors).
        key=lambda p: (len(p.relative_to(output_dir).parts) > 1,
                       p.as_posix()),
    )
    for wav in wavs:
        rel = wav.relative_to(output_dir)
        sidecar_path = wav.with_suffix(wav.suffix + ".json")
        sidecar = None
        if sidecar_path.is_file():
            try:
                sidecar = json.loads(
                    sidecar_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                sidecar = None
        if len(rel.parts) > 1:
            backend, kind = rel.parts[0], rel.stem
        else:
            backend, kind = _top_level_role(rel.stem)
        try:
            import soundfile as sf
            info = sf.info(str(wav))
            duration = round(info.duration, 2)
            sr = info.samplerate
        except Exception:
            duration, sr = "?", "?"
        items.append({
            "id": rel.as_posix(),
            "src": rel.as_posix(),
            "backend": backend,
            "kind": kind,
            "duration": duration,
            "sample_rate": sr,
            "params": _fmt_params(sidecar),
        })
    return {"items": items}


def write_listen_page(output_dir: str | Path, title: str,
                      meta_text: str = "") -> Path:
    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(output_dir)
    # Audio src paths are relative to output_dir; the page lives in
    # listen/, so prefix ../ for playback to work from the subdirectory.
    for item in samples["items"]:
        item["src"] = "../" + item["src"]
    samples["meta"] = meta_text
    page = render_listen_page(output_dir, title, samples, "listen-ratings.json", "samples")
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Phase 5A page: grouped, anonymized, sha256-bound ratings
# ---------------------------------------------------------------------------

_GROUP_TITLES = {
    "originals": "原声参考（非盲听）",
    "baseline": "基线（现用 Prompt P0）",
    "stability": "稳定性集（同 Prompt，多 seed）",
    "prompt_ab": "Prompt A/B（匿名条件）",
    "codec_diagnostics": "Codec 诊断链",
}





def _duration_sr(path: Path) -> tuple[float | str, int | str]:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return round(info.duration, 2), info.samplerate
    except Exception:
        return "?", "?"


def write_phase5a_listen_page(output_dir: str | Path,
                              manifest: dict[str, Any]) -> Path:
    """Grouped blind-listening page for the Phase 5A experiment.

    Samples are anonymized per group (S-xx); a reveal toggle and
    ``unblind_map.json`` expose the true case ids when the user wants
    them. Saved ratings bind each entry to the output file's sha256.
    """
    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict[str, Any]]] = {
        k: [] for k in _GROUP_TITLES}
    unblind: dict[str, Any] = {}

    def add(group: str, case_id: str, rel_src: str | None,
            label: str, extra: dict[str, Any] | None = None,
            error: str | None = None, sha256: str | None = None):
        item: dict[str, Any] = {
            "case_id": case_id, "label": label,
            "src": f"../{rel_src}" if rel_src else None,
            "sha256": sha256, "error": error,
            "duration": "", "sample_rate": "",
        }
        if rel_src:
            dur, sr = _duration_sr(output_dir / rel_src)
            item["duration"], item["sample_rate"] = dur, sr
        if extra:
            item.update(extra)
        groups[group].append(item)
        unblind[label] = {
            "case_id": case_id, "group": group, "src": rel_src,
            "output_sha256": sha256,
            **({} if extra is None else extra),
        }

    # originals (not blinded — they are the calibration anchors)
    gt = output_dir / "ground_truth_original.wav"
    ref = output_dir / "reference_original.wav"
    if gt.is_file():
        add("originals", "original/ground_truth_rain",
            "ground_truth_original.wav", "原声·validation 午后凉风",
            sha256=None)
    if ref.is_file():
        add("originals", "original/clone_reference_P0",
            "reference_original.wav", "克隆参考·谛天鉴(P0)",
            sha256=None)

    counters: dict[str, int] = {}
    cases = manifest.get("cases") or []
    for c in cases:
        group = "baseline" if (
            c.get("group") == "prompt_ab" and c.get("prompt_id") == "P0") \
            else c.get("group", "stability")
        counters[group] = counters.get(group, 0) + 1
        label = f"{group[0].upper()}-{counters[group]:02d}"
        rel = c.get("output")
        err = None if rel else (c.get("error") or "生成失败")
        add(group, c["case_id"], rel, label,
            extra={"text_id": c.get("text_id"),
                   "prompt_id": c.get("prompt_id"),
                   "seed": c.get("seed"),
                   "reused_from": c.get("reused_from")},
            error=err, sha256=c.get("output_sha256"))

    ccount = 0
    for iid, entry in (manifest.get("codec_diagnosis") or {}).items():
        arts = entry.get("artifacts") or {}
        enc = entry.get("encode_input") or {}
        for tag, node in (("encode_input", enc),
                          ("roundtrip48", arts.get("default") or {}),
                          ("roundtrip_cond16k",
                           arts.get("cond16000") or {})):
            p = node.get("path")
            if not p or not Path(p).is_file():
                continue
            ccount += 1
            rel = Path(p).relative_to(output_dir).as_posix()
            add("codec_diagnostics", f"codec/{iid}:{tag}", rel,
                f"C-{ccount:02d}",
                extra={"input_id": iid, "variant": tag},
                sha256=node.get("sha256"))

    group_list = [{"key": k, "title": _GROUP_TITLES[k], "items": v}
                  for k, v in groups.items() if v]
    payload = {
        "experiment_id": manifest.get("experiment_id"),
        "meta": (
            f"实验 {manifest.get('experiment_id')} · "
            f"backend {manifest.get('backend')} · "
            f"rev {str(manifest.get('model', {}).get('revision'))[:12]}\n"
            "标签默认匿名；点「解盲」查看真实 case_id。"
            "所有主观评分默认 PENDING_USER_LISTENING。"),
        "groups": group_list,
    }
    page = render_listen_page(output_dir, f"Phase 5A 盲听 — {manifest.get('experiment_id')}", payload, "listen-ratings-phase5a.json", "groups")
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (listen_dir / "unblind_map.json").write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


DEFECT_TAGS = ["磨砂/粗糙", "前顶/舌位", "空腔/口型大", "语速/停顿",
               "音色不符", "漏字/错读", "金属感", "其他"]


def write_lora_pilot_listen_page(output_dir: str | Path,
                                 manifest: dict[str, Any],
                                 rng_seed: int = 20261008) -> Path:
    """Low-fatigue paired A/B page for the LoRA pilot (PLAN 25.7).

    - pack1 shown by default (4 pairs), later packs collapsed
    - per pair: two blinded sides (甲/乙), deterministic shuffle per
      pair_id; side order balanced across pairs
    - choice + defect tags + first_impression + optional scores
    - UNCERTAIN_FATIGUE session flag; same-sha pairs flagged as
      duplicate evidence, never counted twice
    """
    import random

    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    cases = {c["case_id"]: c for c in manifest.get("cases") or []}
    sha_count: dict[str, int] = {}
    for c in cases.values():
        if c.get("output_sha256"):
            sha_count[c["output_sha256"]] = \
                sha_count.get(c["output_sha256"], 0) + 1

    def _side(case_id: str) -> dict[str, Any] | None:
        c = cases.get(case_id)
        if not c:
            return None
        rel = c.get("output")
        src = f"../{rel}" if rel else None
        dur, sr = _duration_sr(output_dir / rel) if rel else ("?", "?")
        return {"case_id": case_id, "condition": c.get("condition"),
                "src": src, "sha256": c.get("output_sha256"),
                "duration": dur, "sample_rate": sr,
                "error": None if c.get("status") == "ok"
                else (c.get("error") or c["status"]),
                "duplicate_sha": bool(c.get("output_sha256")
                                      and sha_count.get(
                                          c["output_sha256"], 0) > 1)}

    packs_payload, unblind = [], {}
    for pack in manifest.get("packs") or []:
        pairs = []
        for pid in pack.get("pairs") or []:
            pair = next((p for p in manifest.get("pairs") or []
                         if p["pair_id"] == pid), None)
            if not pair:
                continue
            cells = [s for s in (_side(c) for c in pair["cells"]) if s]
            rng = random.Random(f"{rng_seed}:{pid}")
            order = list(range(len(cells)))
            rng.shuffle(order)
            sides = [cells[i] for i in order]
            pairs.append({"pair_id": pid,
                          "text_id": pair["text_id"],
                          "seed": pair["seed"],
                          "cells": [s["case_id"] for s in cells],
                          "sides": sides})
            for i, s in enumerate(sides):
                unblind[f"{pid}:{'甲乙'[i]}"] = s["case_id"]
        packs_payload.append({"pack_id": pack["pack_id"],
                              "title": pack["title"],
                              "collapsed": pack.get("collapsed", False),
                              "pairs": pairs})

    truth = {c["case_id"]: {"condition": c.get("condition"),
                            "checkpoint": c.get("checkpoint"),
                            "output_sha256": c.get("output_sha256")}
             for c in cases.values()}
    payload = {
        "experiment_id": manifest.get("experiment_id"),
        "defect_tags": DEFECT_TAGS,
        "meta": (
            f"实验 {manifest.get('experiment_id')} · "
            f"backend {manifest.get('backend')} · "
            f"rev {str(manifest.get('model', {}).get('revision'))[:12]}\n"
            "先听参考原声校准 → 每组 3–5 对：甲/乙随机隐藏（同文同 "
            "seed，唯一变量是 checkpoint）。每对选「甲更好/乙更好/无"
            "明显差异/暂无法判断」+ 最突出问题标签；可选标首次印象。"
            "疲劳点橙色按钮停止，未听项保持 PENDING_USER_LISTENING。"),
        "packs": packs_payload,
        "truth": truth,
    }
    page = render_listen_page(output_dir, f"LoRA pilot 低疲劳 A/B — {manifest.get('experiment_id')}", payload, "listen-ratings-lora-pilot.json", "pairs")
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (listen_dir / "unblind_map.json").write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_phase5a2_listen_page(output_dir: str | Path,
                               manifest: dict[str, Any],
                               rng_seed: int = 20261008) -> Path:
    """Paired blind-listening page for Phase 5A2.

    Each pair (same text + seed) is one section; the P0/P2 cells appear
    as anonymous 甲/乙 in a deterministic-but-randomized order so prompt
    identity stays hidden. Reveal toggle + ``unblind_map.json`` expose
    the truth; saved choices bind experiment_id + case_id + output_sha256.
    """
    import random

    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    cases = {c["case_id"]: c for c in manifest.get("cases") or []}
    groups: list[dict[str, Any]] = []
    unblind: dict[str, Any] = {}

    # Calibration originals — not blinded.
    orig_items = []
    for fname, label, cid in (
            ("ground_truth_original.wav", "原声·validation 午后凉风",
             "original/ground_truth_rain"),
            ("prompt_P0_original.wav", "Prompt 音频·甲来源",
             "original/prompt_P0"),
            ("prompt_P2_original.wav", "Prompt 音频·乙来源",
             "original/prompt_P2")):
        p = output_dir / fname
        if not p.is_file():
            continue
        dur, sr = _duration_sr(p)
        orig_items.append({
            "case_id": cid, "label": label, "src": f"../{fname}",
            "sha256": None, "error": None,
            "duration": dur, "sample_rate": sr,
        })
    if orig_items:
        groups.append({"key": "originals", "title": _GROUP_TITLES[
            "originals"], "items": orig_items})

    # One group per pair; side order shuffled deterministically.
    for i, pair in enumerate(manifest.get("pairs") or []):
        cells = [cases[cid] for cid in pair["cells"] if cid in cases]
        rng = random.Random(f"{rng_seed}:{pair['pair_id']}")
        order = list(range(len(cells)))
        rng.shuffle(order)
        items = []
        for side, idx in enumerate(order):
            c = cells[idx]
            label = ("甲", "乙")[side]
            rel = c.get("output")
            src = f"../{rel}" if rel else None
            dur, sr = _duration_sr(output_dir / rel) if rel else ("?", "?")
            items.append({
                "case_id": c["case_id"], "label": label,
                "src": src, "sha256": c.get("output_sha256"),
                "error": None if c.get("status") == "ok"
                else (c.get("error") or c["status"]),
                "duration": dur, "sample_rate": sr,
            })
            unblind[f"{pair['pair_id']}:{label}"] = {
                "case_id": c["case_id"], "prompt_id": c.get("prompt_id"),
                "text_id": pair["text_id"], "seed": pair["seed"],
                "src": rel, "output_sha256": c.get("output_sha256"),
                "reused_from": c.get("reused_from"),
            }
        groups.append({
            "key": pair["pair_id"],
            "title": f"配对 {i + 1}：{pair['text_id']} · "
                     f"seed {pair['seed']}",
            "items": items,
        })

    payload = {
        "experiment_id": manifest.get("experiment_id"),
        "meta": (
            f"实验 {manifest.get('experiment_id')} · "
            f"backend {manifest.get('backend')} · "
            f"rev {str(manifest.get('model', {}).get('revision'))[:12]}\n"
            "每个配对内 甲/乙 为匿名条件（同一文本同一 seed，仅 Prompt "
            "不同）；点「解盲」看真实 case_id。主观评分默认 "
            "PENDING_USER_LISTENING。"),
        "groups": groups,
    }
    page = render_listen_page(output_dir, f"Phase 5A2 配对盲听 — {manifest.get('experiment_id')}", payload, "listen-ratings-phase5a2.json", "groups")
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (listen_dir / "unblind_map.json").write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
