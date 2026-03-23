# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Backend layer: file parsing, processing engine, and inference."""

from __future__ import annotations

from emosense.backend.file_parser import FileParser
from emosense.backend.processing_engine import InferenceResult, ProcessingEngine

__all__ = [
    "FileParser",
    "InferenceResult",
    "ProcessingEngine",
]
