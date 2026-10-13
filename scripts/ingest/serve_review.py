#!/usr/bin/env python3
"""Serve a review/refpick bundle locally; clicks write straight to disk.

    .venv\\Scripts\\python.exe scripts\\ingest\\serve_review.py ^
        --bundle outputs\\_tmp\\review_<id> [--port 7865]

Pages: open http://127.0.0.1:<port>/review.html or /refpick.html;
for a gate output dir use --bundle outputs/gates/<exp> and open
/listen/index.html (audio resolves via ../).

Writes:
  POST /api/decisions       -> <bundle>/decisions.json
  POST /api/ref-pick        -> assets/characters/<char>/reference/ref.wav
                             + ref_choice.json
                             + configs/characters/<char>.yaml
                             reference block (audio/sha256/text)
  POST /api/listen-ratings  -> <bundle>/listen/<name>        (export format,
                             consumed by scripts/eval/export_*_ratings.py)
                             + <name>.state                (raw page state,
                             restored on next open)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from character_tts.web.listen_store import (
    ReviewError, contract_for, read_snapshot, save_snapshot, verify_assets,
)
CHAR_RE = re.compile(r"^[a-z0-9_]+$")
LISTEN_NAME_RE = re.compile(r"^listen-ratings[\w-]*\.json$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_character_yaml(char: str, ref_path: Path, sha: str,
                           text: str, root: Path = REPO_ROOT) -> bool:
    """Update reference block of configs/characters/<char>.yaml."""
    import yaml
    cfg = root / "configs" / "characters" / f"{char}.yaml"
    if not cfg.is_file():
        return False
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    rel = ref_path.relative_to(root).as_posix()
    data["reference"] = {
        "audio": f"${{TTS_ROOT}}/{rel}", "sha256": sha,
        "text": text or (data.get("reference") or {}).get("text") or ""}
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True,
                                  sort_keys=False), encoding="utf-8")
    return True


def make_handler(bundle: Path, root: Path = REPO_ROOT):
    bundle = bundle.resolve()
    root = Path(root).resolve()
    ratings_lock = threading.Lock()
    audio_hash_cache: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False)
                       .encode("utf-8"), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            raw = self.rfile.read(n)
            ct = self.headers.get("Content-Type", "")
            enc = ("gbk" if "gbk" in ct.lower() else "utf-8")
            try:
                return json.loads(raw.decode(enc))
            except UnicodeDecodeError:
                return json.loads(raw.decode("gbk" if enc == "utf-8" else "utf-8"))

        def log_message(self, fmt, *args):  # keep console quiet-ish
            sys.stderr.write(f"{fmt % args}\n")

        def do_GET(self):
            url = urlparse(self.path)
            path = unquote(url.path).lstrip("/")
            if not path:
                for cand in ("listen/index.html", "review.html", "refpick.html"):
                    if (bundle / cand).is_file():
                        path = cand
                        break
            if path == "api/decisions":
                p = bundle / "decisions.json"
                self._json(json.loads(p.read_text(encoding="utf-8"))
                           if p.is_file() else {"drop": [], "tags": {}})
                return
            if path == "api/ref-choice":
                char = parse_qs(url.query).get("char", [""])[0]
                p = (root / "assets" / "characters" / char
                     / "reference" / "ref_choice.json")
                self._json(json.loads(p.read_text(encoding="utf-8"))
                           if p.is_file() else {})
                return
            if path == "api/listen-ratings":
                name = parse_qs(url.query).get("name", [""])[0]
                try:
                    with ratings_lock:
                        contract = contract_for(bundle)
                        if contract:
                            if name != contract["ratings_name"]:
                                raise ReviewError("experiment changed; reload", 409)
                            verify_assets(bundle, contract, audio_hash_cache)
                        snapshot = read_snapshot(bundle, name)
                    self._json(snapshot)
                except (ReviewError, json.JSONDecodeError) as e:
                    self._json({"ok": False, "error": str(e)},
                               e.code if isinstance(e, ReviewError) else 400)
                return
            target = (bundle / path).resolve()
            if not target.is_relative_to(bundle) or \
                    not target.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = {"html": "text/html; charset=utf-8",
                     "wav": "audio/wav", "json": "application/json"}.get(
                         target.suffix.lstrip("."), "application/octet-stream")
            self._send(200, target.read_bytes(), ctype)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/api/decisions":
                data = self._body()
                out = {"drop": list(data.get("drop") or []),
                       "tags": data.get("tags") or {},
                       "saved_at": datetime.now(timezone.utc)
                       .isoformat(timespec="seconds")}
                (bundle / "decisions.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                self._json({"ok": True, "path":
                            str(bundle / "decisions.json")})
                return
            if url.path == "/api/listen-ratings":
                try:
                    data = self._body()
                    if not isinstance(data, dict):
                        raise ReviewError("request must be an object")
                    with ratings_lock:
                        result = save_snapshot(bundle, data, audio_hash_cache)
                    self._json(result)
                except (ReviewError, json.JSONDecodeError) as e:
                    self._json({"ok": False, "error": str(e)},
                               e.code if isinstance(e, ReviewError) else 400)
                except OSError:
                    self._json({"ok": False,
                                "error": "local ratings write failed"}, 500)
                return
            if url.path == "/api/ref-pick":
                data = self._body()
                char = data.get("char") or ""
                audio = Path(data.get("audio") or "").name
                if not CHAR_RE.match(char):
                    self._json({"ok": False, "error": "bad char"}, 400)
                    return
                src = (bundle / "audio" / audio).resolve()
                if not str(src).startswith(str(bundle / "audio")) \
                        or not src.is_file():
                    self._json({"ok": False, "error": "audio not found"},
                               400)
                    return
                ref_dir = (root / "assets" / "characters" / char
                           / "reference")
                ref_dir.mkdir(parents=True, exist_ok=True)
                dst = ref_dir / "ref.wav"
                import shutil
                shutil.copy2(src, dst)
                sha = _sha256(dst)
                choice = {"char": char, "audio": audio,
                          "audio_sha256": sha,
                          "source_sha256": data.get("sha"),
                          "text": data.get("text") or "",
                          "picked_at": datetime.now(timezone.utc)
                          .isoformat(timespec="seconds")}
                (ref_dir / "ref_choice.json").write_text(
                    json.dumps(choice, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                yaml_ok = _update_character_yaml(
                    char, dst, sha, data.get("text") or "", root)
                self._json({"ok": True, "ref_path": str(dst),
                            "sha256": sha, "yaml_updated": yaml_ok})
                return
            self._json({"ok": False, "error": "unknown"}, 404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help="dir containing review.html / refpick.html / audio/")
    ap.add_argument("--port", type=int, default=7865)
    ap.add_argument("--open-browser", action="store_true",
                    help="open the local listening/review page after startup")
    args = ap.parse_args()
    bundle = Path(args.bundle)
    if not bundle.is_dir():
        print(f"bundle dir missing: {bundle}", file=sys.stderr)
        return 2
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(bundle))
    print(f"serving {bundle} -> http://127.0.0.1:{args.port}/")
    print("  listen/index.html / review.html / refpick.html; Ctrl+C to stop")
    if args.open_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
