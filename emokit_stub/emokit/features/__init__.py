"""EmoKit feature extraction pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np


class FeaturePipeline:
    """Stub feature extraction pipeline."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, dict):
            arrays = [v.reshape(v.shape[0], -1) if v.ndim > 1 else v for v in X.values()]
            return np.concatenate(arrays, axis=-1).astype(np.float32)
        if isinstance(X, np.ndarray):
            if X.ndim > 2:
                return X.reshape(X.shape[0], -1).astype(np.float32)
            return X.astype(np.float32)
        return np.asarray(X, dtype=np.float32)
