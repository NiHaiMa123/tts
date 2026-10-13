"""Atomic local ratings storage and the Agent's read/continue contract."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"listen-ratings[\w-]*\.json")
RATING_FIELDS = ("clarity", "grit", "likeness", "naturalness",
                 "text_complete", "artifact_type", "stability", "notes")


class ReviewError(ValueError):
    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.code = code


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError("invalid ratings object")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Readers observe one complete snapshot, including its page state."""
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False)
    temp = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def ratings_path(bundle: Path, name: str) -> Path:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ReviewError("bad ratings name")
    if not (bundle / "listen").is_dir():
        raise ReviewError("no listen dir")
    return bundle / "listen" / name


def contract_for(bundle: Path) -> dict[str, Any]:
    return load_json(bundle / "listen" / "review-contract.json")


def verify_assets(bundle: Path, contract: dict[str, Any],
                  cache: dict | None = None) -> None:
    for cid, rel in contract.get("audio_paths", {}).items():
        path = (bundle / rel).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file():
            raise ReviewError(f"audio missing: {cid}", 409)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        sha = cache.get(key) if cache is not None else None
        if sha is None:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            sha = h.hexdigest()
            if cache is not None:
                cache[key] = sha
        if sha != contract["asset_hashes"].get(cid):
            raise ReviewError(f"audio changed: {cid}; rebuild listening page", 409)


def read_snapshot(bundle: Path, name: str) -> dict[str, Any]:
    path = ratings_path(bundle, name)
    data = load_json(path)
    meta = data.get("_review", {})
    # Keep legacy .state files readable; new main JSON is authoritative.
    legacy = load_json(path.with_name(path.name + ".state")) \
        if "_page_state" not in data else {}
    contract = contract_for(bundle)
    migrated = _verified_legacy(data, contract) if not data.get("identity") else None
    if migrated:
        data = migrated
        meta = data["_review"]
    return {"state": data.get("_page_state", legacy.get("state")),
            "identity": data.get("identity"),
            "revision": meta.get("revision"),
            "saved_at": meta.get("saved_at_server", legacy.get("saved_at")),
            "data": data, "path": str(path)}


def _verified_legacy(data: dict, contract: dict) -> dict | None:
    """Restore only historical choices proven by experiment, SHA and order."""
    if not data or not contract or data.get("experiment_id") != contract.get("experiment_id"):
        return None
    pairs = contract.get("mode") == "pairs"
    source = data.get("pairs" if pairs else "ratings") or {}
    valid = {}
    for cid, rating in source.items():
        if not isinstance(rating, dict):
            continue
        if pairs:
            order = contract.get("blind_orders", {}).get(cid)
            truth = data.get("truth", {})
            matches = bool(order) and rating.get("blind_order") == order and all(
                truth.get(case, {}).get("output_sha256") == contract["asset_hashes"].get(case)
                and contract["asset_hashes"].get(case) for case in order.values())
            meaningful = bool(rating.get("choice"))
        else:
            matches = (rating.get("output_sha256") is not None and
                       rating["output_sha256"] == contract.get("asset_hashes", {}).get(cid))
            meaningful = any(str(rating.get(field) or "").strip()
                             for field in RATING_FIELDS)
        if matches and meaningful:
            valid[cid] = rating
    # Fatigue alone must still have a verified audio/ordering context.
    full_context = pairs and bool(contract.get("blind_orders")) and all(
        data.get("pairs", {}).get(pid, {}).get("blind_order") == order and all(
            data.get("truth", {}).get(cid, {}).get("output_sha256") == contract["asset_hashes"].get(cid)
            and contract["asset_hashes"].get(cid) for cid in order.values())
        for pid, order in contract["blind_orders"].items())
    fatigue = bool(full_context and data.get("session", {}).get("fatigue"))
    if not valid and not fatigue:
        return None
    session = {**(data.get("session") or {}), "fatigue": fatigue}
    state = {"identity": contract["identity"], "ratings": valid, "session": session}
    meta = {"revision": None, "saved_at_server": data.get("saved_at"),
            "rated_count": len(valid), "pending_count": len(contract["review_ids"]) - len(valid),
            "fatigue": fatigue, "ready_for_agent": bool(valid or fatigue),
            "legacy_verified": True,
            "unmatched_ids": sorted(set(source) - set(valid))}
    return {**data, "identity": contract["identity"],
            "pairs" if pairs else "ratings": valid,
            "_page_state": state, "_review": meta}


def save_snapshot(bundle: Path, request: dict[str, Any],
                  cache: dict | None = None) -> dict[str, Any]:
    name = request.get("name")
    path = ratings_path(bundle, name)
    data, state = request.get("data"), request.get("state")
    if not isinstance(data, dict) or not isinstance(state, dict):
        raise ReviewError("data and state must be objects")
    contract = contract_for(bundle)
    if contract:
        if name != contract["ratings_name"] or \
                request.get("identity") != contract["identity"] or \
                data.get("identity") != contract["identity"] or \
                state.get("identity") != contract["identity"] or \
                data.get("experiment_id") != contract["experiment_id"]:
            raise ReviewError("experiment changed; reload listening page", 409)
        verify_assets(bundle, contract, cache)
    old = read_snapshot(bundle, name)
    if "expected_revision" in request and \
            request["expected_revision"] != old["revision"]:
        raise ReviewError("ratings changed in another page; reload", 409)
    saved_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    review_ids = contract.get("review_ids", [])
    ratings = state.get("ratings", {})
    session = state.get("session", {})
    if not isinstance(ratings, dict) or not isinstance(session, dict):
        raise ReviewError("invalid page state")
    if any(not isinstance(value, dict) for value in ratings.values()):
        raise ReviewError("invalid rating entry")
    if contract.get("mode") == "pairs":
        rated = sum(bool((ratings.get(cid) or {}).get("choice"))
                    for cid in review_ids)
    else:
        rated = sum(any(str((ratings.get(cid) or {}).get(field) or "").strip()
                        for field in RATING_FIELDS) for cid in review_ids)
    revision = uuid.uuid4().hex
    data = {**data, "_page_state": state, "_review": {
        "revision": revision, "saved_at_server": saved_at,
        "rated_count": rated, "pending_count": len(review_ids) - rated,
        "fatigue": bool(session.get("fatigue")),
        "ready_for_agent": bool(rated or session.get("fatigue")),
        "complete": bool(review_ids) and rated == len(review_ids)}}
    if old["data"] and (old["identity"] != data.get("identity") or
                        old["data"].get("_review", {}).get("legacy_verified")):
        history = path.parent / "history" / path.stem
        history.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        atomic_json(history / f"{suffix}.json", load_json(path))
        if old["state"]:
            atomic_json(history / f"{suffix}.state.json", old["state"])
    atomic_json(path, data)
    # Compatibility only; failures here must not invalidate the primary commit.
    try:
        atomic_json(path.with_name(path.name + ".state"),
                    {"saved_at": saved_at, "state": state})
    except OSError:
        pass
    return {"ok": True, "path": str(path), "revision": revision,
            "saved_at": saved_at, "review": data["_review"]}


def agent_review(bundle: Path) -> dict[str, Any]:
    """Read current disk evidence; never infer fatigue as an A/B tie."""
    bundle = Path(bundle).resolve()
    try:
        contract = contract_for(bundle)
        snapshot = read_snapshot(bundle, contract["ratings_name"]) if contract else {}
    except (OSError, ValueError, KeyError) as e:
        return {"status": "BLOCKED", "reason": f"invalid local review file: {e}"}
    if not contract:
        return {"status": "BLOCKED", "reason": "REBUILD_LISTENING_PAGE"}
    if not snapshot["data"]:
        return {"status": "PENDING_USER_LISTENING", "path": snapshot["path"]}
    if snapshot["identity"] != contract["identity"]:
        return {"status": "BLOCKED", "reason": "STALE_EXPERIMENT"}
    try:
        verify_assets(bundle, contract)
    except (ReviewError, OSError) as e:
        return {"status": "BLOCKED", "reason": str(e)}
    review = snapshot["data"].get("_review", {})
    return {"status": ("UNCERTAIN_FATIGUE" if review.get("fatigue") else
                       "RATED" if review.get("ready_for_agent") else
                       "PENDING_USER_LISTENING"),
            "path": snapshot["path"], "experiment_id": contract["experiment_id"],
            "identity": contract["identity"], "review": review,
            "data": snapshot["data"]}
