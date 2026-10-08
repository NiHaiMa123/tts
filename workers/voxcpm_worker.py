"""VoxCPM2 backend worker.

Runs inside the dedicated voxcpm env. Thin adapter over the official
``voxcpm`` package (openbmb/VoxCPM2). Config arrives as JSON in
``TTS_BACKEND_JSON``.

Codec roundtrip: VoxCPM2 uses an asymmetric AudioVAE (16kHz in -> 48kHz
out). If the installed version exposes encode/decode on the loaded model
we run a real roundtrip; otherwise we return ``unsupported`` — never a
fake equivalent test.
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


class VoxCPMWorker(WorkerServer):
    backend_id = "voxcpm2"
    backend_version = "voxcpm (pip)"

    def __init__(self):
        super().__init__()
        cfg = json.loads(os.environ.get("TTS_BACKEND_JSON", "{}"))
        self.model_cfg = cfg.get("model") or {}
        self.cache_cfg = cfg.get("cache") or {}
        self.runtime_cfg = cfg.get("runtime") or {}
        self.generation_cfg = cfg.get("generation") or {}
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from voxcpm import VoxCPM

        self._torch = torch
        kwargs: dict = {}
        sig = None
        try:
            sig = inspect.signature(VoxCPM.from_pretrained)
        except (TypeError, ValueError):
            pass
        cache_root = self.model_cfg.get("cache_dir") or self.cache_cfg.get("root")
        candidates = {
            "load_denoiser": bool(self.runtime_cfg.get("load_denoiser", False)),
            "cache_dir": cache_root,
            "local_files_only": bool(self.runtime_cfg.get("local_files_only", False)),
        }
        for key, value in candidates.items():
            if value is None or value is False and key != "load_denoiser":
                continue
            if sig is None or key in sig.parameters:
                kwargs[key] = value
        # Prefer fully-local loading: repo id + cache_dir + local_files_only
        # keeps VoxCPM's internal snapshot_download offline-safe.
        if cache_root and "local_files_only" in (
                sig.parameters if sig else {}):
            kwargs["local_files_only"] = True
        model_id = self.model_cfg.get("path_or_id", "openbmb/VoxCPM2")
        revision = self.model_cfg.get("revision")
        if revision and (sig is None or "revision" in sig.parameters):
            kwargs["revision"] = revision
        self.emit_log(f"loading VoxCPM.from_pretrained({model_id!r}, {kwargs})")
        self._model = VoxCPM.from_pretrained(model_id, **kwargs)
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
            import voxcpm
            version = f"voxcpm {getattr(voxcpm, '__version__', '?')}"
        except Exception:
            pass
        return {
            "backend_id": self.backend_id,
            "backend_version": version,
            "model_id": self.model_cfg.get("path_or_id"),
            "revision": self.model_cfg.get("revision"),
            "device": device,
            "dtype": "bfloat16",
            "loaded": self._model is not None,
            "vram": vram,
            "capabilities": self.handle_capabilities(params),
        }

    def handle_capabilities(self, params: dict) -> dict:
        return {
            "zero_shot": True,
            "codec_roundtrip": True,   # attempted; may return unsupported
            "fine_tune": False,
            "lora": True,
            "voice_design": True,
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

        call_kwargs = {
            "text": str(params["text"]),
            "prompt_wav_path": params.get("reference_audio") or None,
            "prompt_text": params.get("reference_text") or None,
            "cfg_value": gen.get("cfg_value", 2.0),
            "inference_timesteps": int(gen.get("inference_timesteps", 10)),
            "normalize": bool(gen.get("normalize", True)),
            "denoise": bool(gen.get("denoise", False)),
            "retry_badcase": bool(gen.get("retry_badcase", True)),
        }
        # Public generate() is (*args, **kwargs) -> ndarray; the real
        # signature lives on _generate().
        target = getattr(model, "_generate", model.generate)
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            call_kwargs = {k: v for k, v in call_kwargs.items()
                           if k in sig.parameters}
        if seed is not None and sig is not None and "seed" in sig.parameters:
            call_kwargs["seed"] = int(seed)

        output_path = Path(params["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # voxcpm logs "Badcase detected" to stderr when retry_badcase
        # re-rolls a bad generation; capture it so we can report the real
        # retry count instead of claiming first-pass success.
        import contextlib
        import io
        errbuf = io.StringIO()
        t0 = time.monotonic()
        with contextlib.redirect_stderr(errbuf):
            wav = model.generate(**call_kwargs)
        wall = time.monotonic() - t0
        retry_count = errbuf.getvalue().count("Badcase detected")

        sr = int(getattr(getattr(model, "tts_model", model), "sample_rate",
                         getattr(model, "sample_rate", 48000)))
        wav_np = np.asarray(wav, dtype=np.float32).squeeze()
        sf.write(str(output_path), wav_np, sr)
        return {
            "sample_rate": sr,
            "duration": float(len(wav_np) / sr),
            "output_path": str(output_path),
            "wall_seconds": wall,
            "metadata": {
                "seed": seed,
                "generation": call_kwargs,
                "retry_badcase_enabled": bool(call_kwargs.get(
                    "retry_badcase", True)),
                "retry_count": retry_count,
            },
        }

    def handle_codec_roundtrip(self, params: dict) -> dict:
        """Real AudioVAE encode->decode when the model exposes it."""
        import numpy as np
        import soundfile as sf
        import torch

        model = self._load()
        tts_model = getattr(model, "tts_model", model)
        vae = getattr(tts_model, "audio_vae", None)
        encode = getattr(vae, "encode", None) if vae is not None else None
        decode = getattr(vae, "decode", None) if vae is not None else None
        if vae is None or not callable(encode) or not callable(decode):
            return {
                "status": "unsupported",
                "reason": "installed voxcpm does not expose an audio "
                          "encode/decode API on the loaded model",
            }

        audio_path = Path(params["audio_path"])
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1)).float()
        in_sr = int(getattr(vae, "sample_rate", 16000))
        if sr != in_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, in_sr)
        device = next(tts_model.parameters()).device
        wav = wav.to(device).unsqueeze(0).unsqueeze(0)  # [B,1,T]

        t0 = time.monotonic()
        with torch.inference_mode():
            lat = encode(wav, in_sr)
            recon = decode(lat)
        wall = time.monotonic() - t0
        if isinstance(recon, dict):
            recon = recon.get("audio", next(iter(recon.values())))
        elif isinstance(recon, (tuple, list)):
            recon = recon[0]
        recon_np = np.asarray(
            recon.float().cpu().squeeze().numpy(), dtype=np.float32)

        out_sr = int(getattr(vae, "out_sample_rate",
                     getattr(vae, "sample_rate", 48000)))
        output_path = Path(params["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), recon_np, out_sr)
        return {
            "status": "ok",
            "sample_rate": out_sr,
            "duration": float(len(recon_np) / out_sr),
            "output_path": str(output_path),
            "wall_seconds": wall,
            "metadata": {"codec": "AudioVAE V2 (asymmetric 16k->48k)"},
        }

    def handle_codec_probe(self, params: dict) -> dict:
        """Diagnostic codec probe (Phase 5A).

        Emits intermediate artifacts so the host can attribute artifacts:
          - ``<stem>_input16k.wav``: the exact resampled signal fed to the
            encoder (the information ceiling for the roundtrip).
          - ``<stem>_roundtrip48.wav``: encode -> decode(default sr_cond).
          - ``<stem>_roundtrip_cond16000.wav`` (variant ``cond16000``):
            same latents decoded with ``sr_cond=16000`` to isolate the
            decoder's HF resynthesis contribution.

        Also returns the latent tensor shape, VAE dtype/device and the
        decoded output length so host-side analysis can check alignment.
        """
        import hashlib
        import numpy as np
        import soundfile as sf
        import torch

        model = self._load()
        tts_model = getattr(model, "tts_model", model)
        vae = getattr(tts_model, "audio_vae", None)
        encode = getattr(vae, "encode", None) if vae is not None else None
        decode = getattr(vae, "decode", None) if vae is not None else None
        if vae is None or not callable(encode) or not callable(decode):
            return {
                "status": "unsupported",
                "reason": "installed voxcpm does not expose an audio "
                          "encode/decode API on the loaded model",
            }

        audio_path = Path(params["audio_path"])
        out_dir = Path(params["output_dir"])
        stem = params.get("stem") or audio_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1)).float()
        in_sr = int(getattr(vae, "sample_rate", 16000))
        if sr != in_sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, in_sr)

        input16k_path = out_dir / f"{stem}_input{in_sr // 1000}k.wav"
        sf.write(str(input16k_path), wav.numpy(), in_sr)

        device = next(tts_model.parameters()).device
        wav_dev = wav.to(device).unsqueeze(0).unsqueeze(0)  # [B,1,T]

        t0 = time.monotonic()
        with torch.inference_mode():
            lat = encode(wav_dev, in_sr)
            variants = list(params.get("variants") or ["default"])
            artifacts = {}
            for variant in variants:
                if variant == "default":
                    recon = decode(lat)
                    name = f"{stem}_roundtrip48.wav"
                elif variant == "cond16000":
                    sr_cond = torch.tensor([16000], device=lat.device,
                                           dtype=torch.int32)
                    recon = decode(lat, sr_cond)
                    name = f"{stem}_roundtrip_cond16k.wav"
                else:
                    continue
                if isinstance(recon, dict):
                    recon = recon.get("audio", next(iter(recon.values())))
                elif isinstance(recon, (tuple, list)):
                    recon = recon[0]
                recon_np = np.asarray(
                    recon.float().cpu().squeeze().numpy(), dtype=np.float32)
                path = out_dir / name
                sf.write(str(path), recon_np, 48000)
                artifacts[variant] = {
                    "path": str(path),
                    "samples": int(recon_np.size),
                }
        wall = time.monotonic() - t0

        try:
            vae_dtype = str(next(vae.parameters()).dtype)
        except StopIteration:
            vae_dtype = "unknown"

        def _sha(p: Path) -> str:
            h = hashlib.sha256()
            h.update(p.read_bytes())
            return h.hexdigest()

        out_sr = int(getattr(vae, "out_sample_rate", 48000))
        return {
            "status": "ok",
            "encode_sample_rate": in_sr,
            "decode_sample_rate": out_sr,
            "vae_dtype": vae_dtype,
            "vae_device": str(device),
            "latent_shape": list(lat.shape),
            "latent_rate_hz": in_sr / int(getattr(vae, "chunk_size", 1)),
            "encode_input": {
                "path": str(input16k_path),
                "samples": int(wav.numel()),
                "sha256": _sha(input16k_path),
            },
            "artifacts": {
                k: {**v, "sha256": _sha(Path(v["path"]))}
                for k, v in artifacts.items()
            },
            "wall_seconds": wall,
            "metadata": {"codec": "AudioVAE V2 (asymmetric 16k->48k)",
                         "probe": True},
        }

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
    raise SystemExit(VoxCPMWorker().serve())
