#!/usr/bin/env python3
"""Run a backend gate evaluation.

Usage:
    python scripts/run_gate.py [--evaluation suoming_gate_v1]
                               [--character suoming]
                               [--backend dots_legacy] [--backend voxcpm2]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.backends.manager import BackendManager  # noqa: E402
from character_tts.evaluation.gate import run_gate  # noqa: E402
from character_tts.evaluation.report import finalize  # noqa: E402
from character_tts.registry.loader import (  # noqa: E402
    load_app_config,
    load_character,
    load_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="suoming_gate_v1")
    parser.add_argument("--character", default=None)
    parser.add_argument("--backend", action="append", default=None,
                        help="restrict to these backend ids (repeatable)")
    parser.add_argument("--keep-worker", action="store_true",
                        help="leave the last worker running after the gate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = load_app_config()
    character_id = args.character or app.default_character
    if not character_id:
        print("no character specified and no default in app.yaml")
        return 2
    character = load_character(character_id)
    evaluation = load_evaluation(args.evaluation)

    manager = BackendManager(
        logs_dir=app.logs_dir,
        on_event=lambda ev: logging.getLogger("worker").info(
            "%s", ev.get("message", ev)),
    )
    try:
        report = run_gate(
            evaluation, character, manager=manager, backends=args.backend
        )
    finally:
        if not args.keep_worker:
            manager.stop()

    paths = finalize(
        evaluation.output_dir,
        title=f"TTS backend gate — {evaluation.evaluation_id} "
              f"({character.display_name})",
        meta=(f"character={character.character_id} "
                   f"eval={evaluation.evaluation_id} "
                   f"generated_at={report['generated_at']}"),
    )

    ok = sum(1 for c in report["cases"] if c["status"] == "ok")
    total = len(report["cases"])
    print(f"\ngate done: {ok}/{total} cases ok")
    for c in report["cases"]:
        mark = {"ok": "OK", "unsupported": "UNSUP",
                "blocked": "BLOCKED"}.get(c["status"], "FAIL")
        print(f"  [{mark:>7}] {c['backend']:<12} {c['kind']:<16} "
              f"{c['output']}  {c.get('error') or c.get('reason') or ''}")
    print(f"metrics: {evaluation.output_dir}/metrics.json")
    print(f"report:  {paths['report_md']}")
    print(f"listen:  {paths['listen_page']}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
