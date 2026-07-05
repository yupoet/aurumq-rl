"""Probe / MLP heads for Kronos embeddings (#7).

These are lightweight, CPU-only supervised heads that sit on top of a frozen
sequence-model embedding (e.g. Kronos) — or on any feature matrix. They share a
minimal ``fit(X, y) -> self`` / ``predict(X) -> (n,)`` interface (the ``Head``
protocol) so the P3 eval harness can swap them interchangeably.

Three heads are provided:

* ``LinearProbeHead``   — ridge regression probe (the classic "is the signal
  linearly decodable from the embedding?" diagnostic).
* ``LogisticProbeHead`` — L2 logistic-regression probe for binary targets,
  with ``predict_proba``.
* ``MLPHead``           — a small torch MLP (regression or classification) with
  a held-out validation split + early stopping, for capturing nonlinear
  structure the linear probe misses.

Every head standardises features with a ``StandardScaler`` that is **fit on the
training data only**; ``predict`` merely *transforms* with the frozen scaler, so
test-time data — however wildly scaled — never leaks back into the fitted
statistics.

Concat mode
-----------
"Concat mode" (embeddings ‖ factor panel, or multi-source embeddings side by
side) needs **no special plumbing here**: every head treats all input columns
identically, so the caller simply ``np.concatenate([...], axis=1)`` before
``fit``/``predict``. There is deliberately no per-block logic in this module —
keeping the heads source-agnostic is the point.

Notes
-----
* The MLP's internal train/validation split is a plain positional tail split.
  TODO(#5): once ``PurgedWalkForwardCV`` lands (issue #5), use it for the
  internal validation split so early-stopping selection is purge/embargo-aware
  rather than a naive contiguous tail.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

__all__ = [
    "Head",
    "LinearProbeHead",
    "LogisticProbeHead",
    "MLPHead",
]


@runtime_checkable
class Head(Protocol):
    """Minimal supervised-head interface shared by every probe/MLP head.

    A head is any object exposing ``fit(X, y) -> self`` and
    ``predict(X) -> ndarray`` of shape ``(n_samples,)``. Classification heads
    additionally expose ``predict_proba``; ``Head`` intentionally does not
    require it so a regression probe still satisfies the protocol.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Head: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class LinearProbeHead:
    """Ridge-regression linear probe with train-only standardisation.

    Parameters
    ----------
    alpha:
        L2 regularisation strength passed to :class:`sklearn.linear_model.Ridge`.

    Attributes
    ----------
    coef_:
        Fitted ridge coefficients in the *standardised* feature space, shape
        ``(n_features,)``.
    scaler_:
        The :class:`~sklearn.preprocessing.StandardScaler` fit on the training
        features only.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.scaler_: StandardScaler | None = None
        self._model: Ridge | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> LinearProbeHead:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self.scaler_ is None:
            raise RuntimeError("LinearProbeHead.predict called before fit().")
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict(Xs).ravel()

    @property
    def coef_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LinearProbeHead.coef_ accessed before fit().")
        return self._model.coef_.ravel()


class LogisticProbeHead:
    """L2 logistic-regression probe for binary targets, train-only scaling.

    Parameters
    ----------
    C:
        Inverse regularisation strength passed to
        :class:`sklearn.linear_model.LogisticRegression`.
    max_iter:
        Solver iteration cap (raised from the sklearn default so the probe
        converges on standardised embeddings without a warning).

    Attributes
    ----------
    scaler_:
        The :class:`~sklearn.preprocessing.StandardScaler` fit on training
        features only.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.scaler_: StandardScaler | None = None
        self._model: LogisticRegression | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> LogisticProbeHead:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel().astype(np.int64)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self._model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self._model.fit(Xs, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self.scaler_ is None:
            raise RuntimeError("LogisticProbeHead.predict called before fit().")
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict(Xs).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self.scaler_ is None:
            raise RuntimeError("LogisticProbeHead.predict_proba called before fit().")
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        return self._model.predict_proba(Xs)


class MLPHead:
    """Small torch MLP head (regression or classification) with early stopping.

    A frozen ``StandardScaler`` (fit on training features only) feeds a
    fully-connected network with ReLU activations and dropout. Training carves
    a contiguous validation tail off the (scaled) training set and early-stops
    on validation loss, restoring the best-val weights before returning.

    Determinism: given identical ``seed`` and inputs, two independent ``fit``
    calls produce bit-for-bit comparable predictions (all seeds are set and the
    validation split is a fixed positional tail). This holds on CPU torch.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dims:
        Width of each hidden layer.
    dropout:
        Dropout probability applied after each hidden activation.
    task:
        ``"regression"`` (default) or ``"classification"`` (binary).
    epochs, batch_size, lr:
        Standard optimisation controls (Adam).
    val_frac:
        Fraction of the training rows (contiguous tail) held out for early
        stopping. If 0 (or too small to yield a batch) early stopping is
        disabled and the final-epoch weights are kept.
    patience:
        Early-stopping patience in epochs (no val improvement).
    seed:
        Seed for all RNGs (torch + numpy) used during fit.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        task: str = "regression",
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-3,
        val_frac: float = 0.15,
        patience: int = 10,
        seed: int = 0,
    ) -> None:
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_frac = val_frac
        self.patience = patience
        self.seed = seed
        self.scaler_: StandardScaler | None = None
        self._model = None  # torch.nn.Module, lazily built in fit()

    # -- internals ---------------------------------------------------------

    def _build(self):
        import torch.nn as nn

        n_out = 2 if self.task == "classification" else 1
        layers: list[nn.Module] = []
        prev = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPHead:
        import torch

        # Full determinism: seed every RNG fit() touches, before any tensor op.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)

        if self.task == "classification":
            y_arr = np.asarray(y).ravel().astype(np.int64)
        else:
            y_arr = np.asarray(y, dtype=np.float32).ravel()

        n = len(Xs)
        n_val = int(n * self.val_frac)
        # Need at least one train and one val batch for a meaningful split.
        use_val = n_val >= 1 and (n - n_val) >= 1
        if use_val:
            X_tr, X_val = Xs[: n - n_val], Xs[n - n_val :]
            y_tr, y_val = y_arr[: n - n_val], y_arr[n - n_val :]
        else:
            X_tr, y_tr = Xs, y_arr
            X_val = y_val = None

        model = self._build()
        model.train()
        self._torch = torch  # cache module ref for predict()

        loss_fn = (
            torch.nn.CrossEntropyLoss() if self.task == "classification" else torch.nn.MSELoss()
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        Xt = torch.from_numpy(X_tr)
        if self.task == "classification":
            yt = torch.from_numpy(y_tr)
        else:
            yt = torch.from_numpy(y_tr).unsqueeze(1)

        if use_val:
            Xv = torch.from_numpy(X_val)
            yv = (
                torch.from_numpy(y_val)
                if self.task == "classification"
                else torch.from_numpy(y_val).unsqueeze(1)
            )

        n_tr = len(Xt)
        # Deterministic per-epoch shuffle driven by a seeded generator.
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for _epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n_tr, generator=gen)
            for start in range(0, n_tr, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                out = model(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()

            if use_val:
                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(Xv), yv).item())
                if val_loss < best_val - 1e-9:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        return self

    # -- inference ---------------------------------------------------------

    def _forward(self, X: np.ndarray):
        if self._model is None or self.scaler_ is None:
            raise RuntimeError("MLPHead used before fit().")
        torch = self._torch
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            return self._model(torch.from_numpy(Xs))

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self._forward(X)
        if self.task == "classification":
            return out.argmax(dim=1).numpy().astype(np.int64)
        return out.squeeze(1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for task='classification'.")
        out = self._forward(X)
        return self._torch.softmax(out, dim=1).numpy()
