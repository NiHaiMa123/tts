"""Qwen3-TTS backend worker.

Runs inside the dedicated qwen-tts env. Thin adapter over the official
``qwen_tts`` package (Qwen/Qwen3-TTS-12Hz-*). Config arrives as JSON in
``TTS_BACKEND_JSON``.

Codec roundtrip: if the installed package exposes the 12Hz speech
tokenizer with encode/decode we run a real roundtrip; otherwise we
return ``unsupported``.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _worker_base import WorkerServer  # noqa: E402


class Qwen3TTSWorker(WorkerServer):
    backend_id = "qwen3_tts"
    backend_version = "qwen-tts (pip)"

    def __init__(self):
        super().__init__()
        cfg = json.loads(os.environ.get("TTS_BACKEND_JSON", "{}"))
        self.model_cfg = cfg.get("model") or {}
        self.cache_cfg = cfg.get("cache") or {}
        self.runtime_cfg = cfg.get("runtime") or {}
        self.generation_cfg = cfg.get("generation") or {}
        self._model = None
        self._torch = None

    def _resolve_snapshot(self, model_id: str,
                          revision: str | None = None) -> str:
        """Resolve a repo id to its local snapshot dir (offline-safe).

        Falls back to the literal id when it is already a path or when the
        snapshot is not cached — from_pretrained then handles it.
        """
        if Path(model_id).exists():
            return model_id
        cache_root = (self.model_cfg.get("cache_dir")
                      or self.cache_cfg.get("root"))
        revision = revision or self.model_cfg.get("revision") or None
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                model_id,
                revision=revision,
                cache_dir=cache_root,
                local_files_only=True,
            )
            self.emit_log(f"resolved {model_id} -> {path}")
            return path
        except Exception as exc:
            self.emit_log(f"snapshot resolve failed for {model_id}: {exc}",
                          level="warning")
            return model_id

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        dtype_name = self.runtime_cfg.get("precision", "bfloat16")
        dtype = getattr(torch, dtype_name, torch.bfloat16)
        device_map = self.runtime_cfg.get("device_map", "cuda:0")
        attn = self.runtime_cfg.get("attn_implementation", "sdpa")
        model_id = self._resolve_snapshot(
            self.model_cfg.get("path_or_id",
                               "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
        kwargs: dict = {
            "device_map": device_map,
            "dtype": dtype,
            "attn_implementation": attn,
        }
        self.emit_log(
            f"loading Qwen3TTSModel.from_pretrained({model_id!r}, {kwargs})")
        self._model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        return self._model

    # -- protocol handlers -------------------------------------------------------
    def handle_health(self, params: dict) -> dict:
        vram = None
        device = "cpu"
        try:
            torch = self._torch
            if torch is None:
                import torch as _t
                torch = _t
            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
                vram = {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                }
        except Exception:
            pass
        version = self.backend_version
        try:
            import qwen_tts
            version = f"qwen-tts {getattr(qwen_tts, '__version__', '?')}"
        except Exception:
            pass
        return {
            "backend_id": self.backend_id,
            "backend_version": version,
            "model_id": self.model_cfg.get("path_or_id"),
            "revision": self.model_cfg.get("revision"),
            "device": device,
            "dtype": self.runtime_cfg.get("precision", "bfloat16"),
            "loaded": self._model is not None,
            "vram": vram,
            "capabilities": self.handle_capabilities(params),
        }

    def handle_capabilities(self, params: dict) -> dict:
        return {
            "zero_shot": True,
            "codec_roundtrip": True,   # attempted; may return unsupported
            "fine_tune": True,
            "lora": False,
            "voice_design": False,     # Base checkpoint: clone only
            "streaming": False,
        }

    def handle_generate(self, params: dict) -> dict:
        import numpy as np
        import soundfile as sf
        import torch

        model = self._load()
        options = params.get("options") or {}
        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        gen = dict(self.generation_cfg)
        gen.update(options)

        ref_audio = params.get("reference_audio")
        if not ref_audio:
            raise ValueError("qwen3_tts voice clone requires reference_audio")

        call_kwargs = {
            "text": str(params["text"]),
            "language": gen.get("language", "Chinese"),
            "ref_audio": ref_audio,
            "ref_text": params.get("reference_text"),
            "max_new_tokens": int(gen.get("max_new_tokens", 2048)),
            "do_sample": bool(gen.get("do_sample", True)),
            "top_k": gen.get("top_k", 50),
            "top_p": gen.get("top_p", 1.0),
            "temperature": gen.get("temperature", 0.9),
            "repetition_penalty": gen.get("repetition_penalty", 1.05),
            "subtalker_dosample": gen.get("subtalker_dosample", True),
            "subtalker_top_k": gen.get("subtalker_top_k", 50),
            "subtalker_top_p": gen.get("subtalker_top_p", 1.0),
            "subtalker_temperature": gen.get("subtalker_temperature", 0.9),
            "x_vector_only_mode": gen.get("x_vector_only_mode", False),
            "non_streaming_mode": gen.get("non_streaming_mode", True),
        }
        fn = getattr(model, "generate_voice_clone")
        sig = inspect.signature(fn)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values())
        if not has_varkw:
            call_kwargs = {k: v for k, v in call_kwargs.items()
                           if k in sig.parameters and v is not None}
        else:
            call_kwargs = {k: v for k, v in call_kwargs.items()
                           if v is not None}
        if seed is not None and "seed" in sig.parameters:
            call_kwargs["seed"] = int(seed)

        output_path = Path(params["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        wavs, sr = fn(**call_kwargs)
        wall = time.monotonic() - t0
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        wav_np = np.asarray(wav, dtype=np.float32).squeeze()
        sf.write(str(output_path), wav_np, int(sr))
        return {
            "sample_rate": int(sr),
            "duration": float(len(wav_np) / int(sr)),
            "output_path": str(output_path),
            "wall_seconds": wall,
            "metadata": {"seed": seed, "generation": call_kwargs},
        }

    def handle_codec_roundtrip(self, params: dict) -> dict:
        """12Hz tokenizer encode->decode when the package exposes it."""
        import numpy as np
        import soundfile as sf
        import torch

        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            return {
                "status": "unsupported",
                "reason": "installed qwen-tts does not expose the 12Hz "
                          "tokenizer encode/decode API",
            }

        encode = getattr(tokenizer, "encode", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(encode) or not callable(decode):
            return {"status": "unsupported",
                    "reason": "tokenizer object lacks encode/decode"}

        audio_path = Path(params["audio_path"])
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav_np = data.mean(axis=1).astype(np.float32)

        t0 = time.monotonic()
        encoded = encode(wav_np, sr=sr)
        wavs, out_sr = decode(encoded)
        wall = time.monotonic() - t0
        recon = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        recon_np = np.asarray(recon, dtype=np.float32).squeeze()
        out_sr = int(out_sr or getattr(tokenizer, "sample_rate", sr))
        output_path = Path(params["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), recon_np, out_sr)
        return {
            "status": "ok",
            "sample_rate": out_sr,
            "duration": float(len(recon_np) / out_sr),
            "output_path": str(output_path),
            "wall_seconds": wall,
            "metadata": {"codec": "Qwen3-TTS-Tokenizer-12Hz"},
        }

    def _load_tokenizer(self):
        """Load the official 12Hz tokenizer; None when unavailable."""
        tokenizer_id = (self.model_cfg.get("tokenizer")
                        or "Qwen/Qwen3-TTS-Tokenizer-12Hz")
        try:
            from qwen_tts import Qwen3TTSTokenizer
        except ImportError:
            return None
        try:
            resolved = self._resolve_snapshot(
                tokenizer_id,
                revision=self.model_cfg.get("tokenizer_revision"))
            self.emit_log(f"loading tokenizer {resolved}")
            return Qwen3TTSTokenizer.from_pretrained(resolved)
        except Exception as exc:
            self.emit_log(f"tokenizer load failed: {exc}", level="warning")
            return None

    def handle_shutdown(self, params: dict) -> dict:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {"stopped": True}


if __name__ == "__main__":
    raise SystemExit(Qwen3TTSWorker().serve())
