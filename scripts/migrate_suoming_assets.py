"""One-off: copy suoming source audio + dataset audio from dotstts into this repo."""
import json
import os
import shutil
import sys
from pathlib import Path

DOTS = Path(r"E:\project\dotstts")
REPO = Path(r"E:\project\tts")


def lp(p):
    """Long-path prefix for Windows (>240 chars)."""
    s = str(p)
    return "\\\\?\\" + s if len(s) > 200 and not s.startswith("\\\\") else s


def copy_tree(src, dst):
    count = skipped = total = 0
    for root, dirs, files in os.walk(lp(src)):
        rel = Path(root.replace("\\\\?\\", "")).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s, d = Path(root) / f, dst / rel / f
            try:
                shutil.copy2(lp(s), lp(d))
                count += 1
                total += os.path.getsize(lp(s))
            except OSError as e:
                skipped += 1
                print("SKIP", f[:50], e)
    print(f"{src.name} -> {dst}: {count} files, {total/1e6:.0f} MB, skipped={skipped}")
    return count, skipped


def main():
    # 1) inbox raw audio
    copy_tree(DOTS / "data" / "inbox" / "锁暝",
              REPO / "data" / "characters" / "suoming" / "inbox")

    # 2) dataset manifests + referenced audio (self-contained copy)
    v1_src = DOTS / "datasets" / "suoming" / "v1"
    v1_dst = REPO / "data" / "characters" / "suoming" / "datasets" / "v1"
    audio_dst = v1_dst / "audio"
    audio_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v1_src / "manifest.json", v1_dst / "manifest.json")

    missing = copied = 0
    for jf in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        lines = []
        for line in (v1_src / jf).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            src_audio = Path(row["audio"])
            local = audio_dst / src_audio.name
            if not local.is_file():
                if src_audio.is_file():
                    shutil.copy2(lp(src_audio), lp(local))
                    copied += 1
                else:
                    missing += 1
                    print("MISSING", src_audio.name)
            row["audio"] = str(local.relative_to(REPO)).replace("\\", "/")
            lines.append(json.dumps(row, ensure_ascii=False))
        (v1_dst / jf).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"dataset audio: copied={copied} missing={missing}")

    # 3) reference wav for zero-shot (from inbox, verify sha)
    import hashlib
    ref_src = None
    for root, dirs, files in os.walk(lp(DOTS / "data" / "inbox" / "锁暝")):
        for f in files:
            if "谛天鉴除了处理岁主事务" in f:
                ref_src = Path(root) / f
                break
        if ref_src:
            break
    ref_dst = REPO / "assets" / "characters" / "suoming" / "reference" / "ref.wav"
    ref_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lp(ref_src), lp(ref_dst))
    sha = hashlib.sha256(ref_dst.read_bytes()).hexdigest()
    print(f"reference: {ref_dst.name} sha256={sha[:16]}... (expect c4322e32...)",
          "MATCH" if sha.startswith("c4322e32") else "MISMATCH!")

    # 4) anchor audio referenced by evaluation.anchor_texts
    anchor_src = DOTS / "data" / "work" / "standardized" / "fb" / \
        "fb8e3c080298ab2d0a404789e6973d68cf5af7bfadb8486851fae677f1aac1ca.wav"
    anchor_dst = audio_dst / anchor_src.name
    if anchor_src.is_file() and not anchor_dst.is_file():
        shutil.copy2(lp(anchor_src), lp(anchor_dst))
        print("anchor audio copied:", anchor_dst.name)


if __name__ == "__main__":
    sys.exit(main())
