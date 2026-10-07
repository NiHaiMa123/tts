"""character_tts — backend-agnostic character TTS platform.

The platform layer never imports model internals. Each backend runs in its
own Python environment as a JSONL stdin/stdout subprocess worker.
"""

__version__ = "0.1.0"
