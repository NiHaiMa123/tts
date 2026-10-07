"""Dots legacy backend worker.

Runs inside the legacy dotstts Python env. Thin adapter: loads
``DotsTtsRuntime`` (base or trainable-delta/LoRA variant) and serves the
JSONL worker protocol. Model path / adapter / sampling defaults arrive as
JSON in the ``TTS_BACKEND_JSON`` env var.

Codec roundtrip uses the model's own AudioVAE encoder/decoder — a real
waveform -> latents -> waveform path, never faked.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _worker_base import WorkerServer  # noqa: E402

_FULLWIDTH_EMPHASIS = re.compile(r"[！]+")
_HALFWIDTH_EMPHASIS = re.compile(r"[!]+")
_QUESTION_EMPHASIS = re.compile(r"(？)!+")


def _soften_emphasis(text: str) -> str:
    text = _QUESTION_EMPHASIS.sub(r"\1", text)
    text = _FULLWIDTH_EMPHASIS.sub("。", text)
    text = _HALFWIDTH_EMPHASIS.sub(".", text)
    return text


class DotsWorker(WorkerServer):
    backend_id = "dots_legacy"
    backend_version = "dots.tts 0.3.1 (legacy)"

    def __init__(self):
        super().__init__()
        cfg = json.loads(os.environ.get("TTS_BACKEND_JSON", "{}"))
        self.model_cfg = cfg.get("model") or {}
        self.runtime_cfg = cfg.get("runtime") or {}
        self.generation_cfg = cfg.get("generation") or {}
        self._runtimes: dict[str, object] = {}
        self._active_variant: str | None = None
        self._torch = None

    # -- model management ------------------------------------------------------
    def _torch_module(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _variant_from_options(self, options: dict) -> str:
        variant = (options or {}).get("variant", "default")
        if variant == "default":
            variant = "lora" if self.model_cfg.get("adapter") else "base"
        if variant not in ("base", "lora"):
            raise ValueError(f"unknown variant: {variant}")
        if variant == "lora" and not self.model_cfg.get("adapter"):
            raise ValueError("variant 'lora' requested but no adapter configured")
        return variant

    def _unload(self) -> None:
        self._runtimes.clear()
        self._active_variant = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _ensure_runtime(self, variant: str):
        if self._active_variant == variant and variant in self._runtimes:
            return self._runtimes[variant]
        # One model resident at a time: drop previous variant before loading.
        self._unload()
        from dots_tts.runtime import DotsTtsRuntime
        from dots_tts.utils.util import seed_everything

        seed_everything(int(self.generation_cfg.get("base_seed", 42)))
        path = self.model_cfg["path_or_id"]
        revision = self.model_cfg.get("revision") or None
        cache_dir = self.model_cfg.get("cache_dir") or None
        kwargs = dict(
            revision=revision,
            cache_dir=cache_dir,
            precision=self.runtime_cfg.get("precision", "bfloat16"),
            optimize=bool(self.runtime_cfg.get("optimize", False)),
            max_generate_length=int(self.runtime_cfg.get("max_generate_length", 500)),
            max_sequence_length=int(self.runtime_cfg.get("max_sequence_length", 2048)),
            vocoder_merge_steps=int(self.runtime_cfg.get("vocoder_merge_steps", 4)),
            warmup_on_optimize=bool(self.runtime_cfg.get("warmup_on_optimize", False)),
        )
        self.emit_log(f"loading dots runtime variant={variant} model={path}")
        if variant == "lora":
            kwargs["merge_lora"] = bool(self.runtime_cfg.get("merge_lora", False))
            runtime = DotsTtsRuntime.from_pretrained_with_trainable_delta(
                path, self.model_cfg["adapter"], **kwargs
            )
        else:
            runtime = DotsTtsRuntime.from_pretrained(path, **kwargs)
        self._runtimes[variant] = runtime
        self._active_variant = variant
        return runtime

    # -- protocol handlers -------------------------------------------------------
    def handle_health(self, params: dict) -> dict:
        torch = self._torch_module()
        vram = None
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            vram = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated(),
            }
        return {
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "model_id": self.model_cfg.get("path_or_id"),
            "revision": self.model_cfg.get("revision"),
            "device": device,
            "dtype": self.runtime_cfg.get("precision", "bfloat16"),
            "loaded": self._active_variant is not None,
            "active_variant": self._active_variant,
            "vram": vram,
            "capabilities": self.handle_capabilities(params),
        }

    def handle_capabilities(self, params: dict) -> dict:
        return {
            "zero_shot": True,
            "codec_roundtrip": True,
            "fine_tune": False,
            "lora": bool(self.model_cfg.get("adapter")),
            "voice_design": False,
            "streaming": False,
        }

    def handle_generate(self, params: dict) -> dict:
        import soundfile as sf
        from dots_tts.utils.util import seed_everything

        options = params.get("options") or {}
        variant = self._variant_from_options(options)
        runtime = self._ensure_runtime(variant)

        text = str(params["text"])
        if options.get("soften_emphasis",
                       self.generation_cfg.get("soften_emphasis", False)):
            text = _soften_emphasis(text)

        seed = params.get("seed")
        if seed is not None:
            seed_everything(int(seed))

        gen = dict(self.generation_cfg)
        gen.update({k: v for k, v in options.items() if k in (
            "language", "template_name", "ode_method", "num_steps",
            "guidance_scale", "speaker_scale", "normalize_text",
        )})
        output_path = Path(params["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        result = runtime.generate(
            text=text,
            prompt_audio_path=params.get("reference_audio") or None,
            prompt_text=params.get("reference_text") or None,
            language=gen.get("language"),
            template_name=gen.get("template_name"),
            ode_method=gen.get("ode_method"),
            num_steps=gen.get("num_steps"),
            guidance_scale=gen.get("guidance_scale"),
            speaker_scale=gen.get("speaker_scale", 1.5),
            normalize_text=bool(gen.get("normalize_text", False)),
        )
        audio = result["audio"].float().cpu().squeeze().numpy()
        sr = int(result["sample_rate"])
        sf.write(str(output_path), audio, sr)
        wall = time.monotonic() - t0
        return {
            "sample_rate": sr,
            "duration": float(len(audio) / sr),
            "output_path": str(output_path),
            "wall_seconds": wall,
            "metadata": {
                "variant": variant,
                "seed": seed,
                "generation": gen,
                "adapter": self.model_cfg.get("adapter") if variant == "lora" else None,
            },
        }

    def handle_codec_roundtrip(self, params: dict) -> dict:
        """Real AudioVAE encode -> decode of the input waveform."""
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        # Reuse whichever variant is loaded; roundtrip only needs the VAE,
        # which is shared. Load base variant if nothing is resident.
        runtime = self._ensure_runtime(
            self._active_variant or
            ("lora" if self.model_cfg.get("adapter") else "base")
        )
        model = runtime.model
        vae = model.vocoder

        audio_path = Path(params["audio_path"])
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1)).float()
        if sr != vae.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, vae.sample_rate)
        wav = wav.to(next(vae.parameters()).device).unsqueeze(0).unsqueeze(0)

        t0 = time.monotonic()
        with torch.inference_mode():
            # Official VAE reconstruction path: extract_latents ->
            # inference_from_latents (posterior sample -> decode).
            recon = vae.inference({"sample": wav})["sample"]
        wall = time.monotonic() - t0
        recon_np = recon.float().cpu().squeeze().numpy().astype(np.float32)

        output_path = Path(params["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), recon_np, int(vae.sample_rate))
        return {
            "status": "ok",
            "sample_rate": int(vae.sample_rate),
            "duration": float(len(recon_np) / vae.sample_rate),
            "output_path": str(output_path),
            "wall_seconds": wall,
            "metadata": {"codec": "AudioVAE(bigvgan-style)",
                          "hop_size": int(vae.hop_size)},
        }

    def handle_shutdown(self, params: dict) -> dict:
        self._unload()
        return {"stopped": True}


if __name__ == "__main__":
    raise SystemExit(DotsWorker().serve())
