"""EmoKit model registry and base classes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all EmoKit emotion recognition models."""

    @abstractmethod
    def fit(self, X: Any, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


class _SimpleNetwork(nn.Module):
    """Minimal feed-forward network for stub models."""

    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.network(x)


class _StubModel(BaseModel):
    """Concrete stub model that works for all EmoSense model types."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self._name = name
        self._params = params
        self._n_classes = params.get("n_classes", 2)
        self._net: _SimpleNetwork | None = None
        self._in_features: int | None = None
        self._attention_weights: np.ndarray | None = None

    def fit(self, X: Any, y: np.ndarray) -> None:
        if isinstance(X, dict):
            flat = np.concatenate(
                [v.reshape(v.shape[0], -1) if v.ndim > 1 else v for v in X.values()],
                axis=-1,
            )
        else:
            flat = X.reshape(X.shape[0], -1) if X.ndim > 1 else X

        self._in_features = flat.shape[-1]
        self._net = _SimpleNetwork(self._in_features, self._n_classes)

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(flat), torch.LongTensor(y),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        n_epochs = self._params.get("n_epochs", 3)
        self._net.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

        n_modalities = len(X) if isinstance(X, dict) else 1
        if n_modalities > 1:
            w = np.random.dirichlet(np.ones(n_modalities))
            self._attention_weights = w.astype(np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._net is None:
            return np.ones((1, self._n_classes)) / self._n_classes

        self._net.eval()
        if isinstance(X, dict):
            flat = np.concatenate(
                [v.reshape(v.shape[0], -1) if v.ndim > 1 else v for v in X.values()],
                axis=-1,
            )
        elif isinstance(X, np.ndarray):
            flat = X.reshape(X.shape[0], -1) if X.ndim > 1 else X[np.newaxis, :]
        else:
            flat = X
        with torch.no_grad():
            logits = self._net(torch.FloatTensor(flat))
        return logits.numpy()

    def save(self, path: str) -> None:
        state = {
            "name": self._name,
            "params": self._params,
            "in_features": self._in_features,
            "n_classes": self._n_classes,
            "attention_weights": self._attention_weights,
        }
        if self._net is not None:
            state["state_dict"] = self._net.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self._in_features = state.get("in_features")
        self._n_classes = state.get("n_classes", 2)
        self._attention_weights = state.get("attention_weights")
        if self._in_features is not None and "state_dict" in state:
            self._net = _SimpleNetwork(self._in_features, self._n_classes)
            self._net.load_state_dict(state["state_dict"])
            self._net.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            return torch.zeros(x.size(0), self._n_classes)
        self._net.eval()
        with torch.no_grad():
            return self._net(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_attention_weights(self) -> np.ndarray | None:
        return self._attention_weights


_REGISTRY: dict[str, type] = {}


def build_model(name: str, params: dict[str, Any]) -> _StubModel:
    """Build a model by registry name. Returns a stub model."""
    return _StubModel(name, params)
