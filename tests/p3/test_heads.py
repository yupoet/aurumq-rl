"""Tests for aurumq_rl.p3.heads — probe / MLP heads for Kronos embeddings (#7).

Synthetic-only: no real Kronos embeddings, no GPU, no data files. All
randomness is seeded so the suite is deterministic (torch + sklearn are
CPU-only in this environment).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

from aurumq_rl.p3.heads import (
    Head,
    LinearProbeHead,
    LogisticProbeHead,
    MLPHead,
)

# =============================================================================
# Synthetic data builders
# =============================================================================


def _linear_signal(n: int = 2000, d: int = 12, noise: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = X @ w + noise * rng.standard_normal(n)
    return X, y, w


def _binary_signal(n: int = 2000, d: int = 10, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return X, y


def _nonlinear_signal(n: int = 3000, d: int = 8, noise: float = 0.05, seed: int = 2):
    """y depends on X through sin() + a cross term — a linear probe should
    struggle to capture this while a small MLP can.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = np.sin(2.0 * X[:, 0]) + X[:, 1] * X[:, 2] - 0.5 * X[:, 3] ** 2
    y = y + noise * rng.standard_normal(n)
    return X, y


def _train_test_split(n: int, test_frac: float = 0.25):
    n_test = int(n * test_frac)
    return np.arange(0, n - n_test), np.arange(n - n_test, n)


# =============================================================================
# 1. Linear probe recovers a linear signal
# =============================================================================


class TestLinearProbeHead:
    def test_oos_ic_beats_shuffled_control(self):
        X, y, _w = _linear_signal(seed=0)
        train_idx, test_idx = _train_test_split(len(X))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        head = LinearProbeHead(alpha=1.0)
        head.fit(X_train, y_train)
        pred = head.predict(X_test)
        ic_real, _ = spearmanr(pred, y_test)

        rng = np.random.default_rng(123)
        y_shuffled = rng.permutation(y_train)
        shuffled_head = LinearProbeHead(alpha=1.0)
        shuffled_head.fit(X_train, y_shuffled)
        pred_shuffled = shuffled_head.predict(X_test)
        ic_shuffled, _ = spearmanr(pred_shuffled, y_test)

        assert ic_real > 0.3, f"expected clearly positive OOS IC, got {ic_real:.3f}"
        assert ic_real > ic_shuffled + 0.2, (
            f"real-label IC ({ic_real:.3f}) should clearly beat the "
            f"shuffled-label control ({ic_shuffled:.3f})"
        )

    def test_recovers_weight_direction(self):
        X, y, w = _linear_signal(n=4000, d=12, noise=0.2, seed=0)
        head = LinearProbeHead(alpha=1.0)
        head.fit(X, y)
        cos_sim = np.dot(head.coef_, w) / (np.linalg.norm(head.coef_) * np.linalg.norm(w))
        assert cos_sim > 0.9, (
            f"expected fitted coef direction close to true w, cos_sim={cos_sim:.3f}"
        )

    def test_sample_weight_accepted(self):
        X, y, _w = _linear_signal(n=500, seed=3)
        weights = np.ones(len(X))
        head = LinearProbeHead()
        head.fit(X, y, sample_weight=weights)
        preds = head.predict(X)
        assert preds.shape == (len(X),)


# =============================================================================
# 2. Logistic probe on a binary signal
# =============================================================================


class TestLogisticProbeHead:
    def test_auc_and_accuracy_beat_chance(self):
        X, y = _binary_signal(seed=1)
        train_idx, test_idx = _train_test_split(len(X))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        head = LogisticProbeHead(C=1.0)
        head.fit(X_train, y_train)
        proba = head.predict_proba(X_test)[:, 1]
        pred = head.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, pred)
        assert auc > 0.65, f"expected AUC clearly > 0.5, got {auc:.3f}"
        assert acc > 0.6, f"expected accuracy clearly > 0.5, got {acc:.3f}"

    def test_predict_proba_shape_and_range(self):
        X, y = _binary_signal(n=300, seed=4)
        head = LogisticProbeHead()
        head.fit(X, y)
        proba = head.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()


# =============================================================================
# 3. MLP captures nonlinearity that the linear probe misses
# =============================================================================


class TestMLPHeadNonlinear:
    def test_mlp_beats_linear_probe_on_nonlinear_signal(self):
        X, y = _nonlinear_signal(n=3000, d=8, seed=2)
        train_idx, test_idx = _train_test_split(len(X))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        linear = LinearProbeHead(alpha=1.0)
        linear.fit(X_train, y_train)
        ic_linear, _ = spearmanr(linear.predict(X_test), y_test)

        mlp = MLPHead(
            input_dim=X.shape[1],
            hidden_dims=(64, 32),
            dropout=0.1,
            epochs=120,
            batch_size=128,
            lr=5e-3,
            val_frac=0.15,
            patience=15,
            seed=42,
        )
        mlp.fit(X_train, y_train)
        ic_mlp, _ = spearmanr(mlp.predict(X_test), y_test)

        assert ic_mlp > 0.3, f"expected MLP to capture the nonlinear signal, got IC={ic_mlp:.3f}"
        assert ic_mlp > ic_linear, (
            f"MLP IC ({ic_mlp:.3f}) should beat the linear probe IC ({ic_linear:.3f}) "
            "on a nonlinear target"
        )

    def test_mlp_classification_mode(self):
        X, y = _binary_signal(n=1500, d=6, seed=5)
        train_idx, test_idx = _train_test_split(len(X))
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        mlp = MLPHead(
            input_dim=X.shape[1],
            hidden_dims=(32, 16),
            task="classification",
            epochs=60,
            batch_size=128,
            lr=5e-3,
            val_frac=0.15,
            patience=10,
            seed=42,
        )
        mlp.fit(X_train, y_train)
        proba = mlp.predict_proba(X_test)
        pred = mlp.predict(X_test)

        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        auc = roc_auc_score(y_test, proba[:, 1])
        assert auc > 0.6, f"expected classification MLP AUC > 0.5, got {auc:.3f}"
        assert set(np.unique(pred)) <= {0, 1}


# =============================================================================
# 4. No train/test scaler leakage + determinism
# =============================================================================


class TestNoLeakageAndDeterminism:
    @pytest.mark.parametrize("head_cls", [LinearProbeHead, LogisticProbeHead])
    def test_scaler_fit_on_train_only(self, head_cls):
        rng = np.random.default_rng(7)
        X_train = rng.standard_normal((500, 6))
        y_train = (
            (rng.standard_normal(500) > 0).astype(np.int64)
            if head_cls is LogisticProbeHead
            else rng.standard_normal(500)
        )

        head = head_cls()
        head.fit(X_train, y_train)
        mean_before = head.scaler_.mean_.copy()
        scale_before = head.scaler_.scale_.copy()

        # A test set on a wildly different scale/location must NOT perturb the
        # fitted scaler when we merely call predict() on it.
        X_test_shifted = rng.standard_normal((200, 6)) * 1000.0 + 5000.0
        _ = head.predict(X_test_shifted)

        np.testing.assert_array_equal(head.scaler_.mean_, mean_before)
        np.testing.assert_array_equal(head.scaler_.scale_, scale_before)

    def test_linear_probe_deterministic_refit(self):
        X, y, _w = _linear_signal(n=400, seed=9)
        head_a = LinearProbeHead(alpha=2.0)
        head_a.fit(X, y)
        head_b = LinearProbeHead(alpha=2.0)
        head_b.fit(X, y)
        np.testing.assert_allclose(head_a.predict(X), head_b.predict(X), rtol=1e-10)

    def test_mlp_deterministic_refit_same_seed(self):
        X, y = _nonlinear_signal(n=400, d=5, seed=11)
        kwargs = dict(
            input_dim=X.shape[1],
            hidden_dims=(16, 8),
            epochs=20,
            batch_size=64,
            val_frac=0.2,
            patience=5,
            seed=99,
        )
        head_a = MLPHead(**kwargs)
        head_a.fit(X, y)
        head_b = MLPHead(**kwargs)
        head_b.fit(X, y)
        np.testing.assert_allclose(head_a.predict(X), head_b.predict(X), rtol=1e-5, atol=1e-6)


# =============================================================================
# 5. Shared fit/predict interface
# =============================================================================


class TestSharedInterface:
    def test_linear_probe_satisfies_head_protocol(self):
        head = LinearProbeHead()
        assert isinstance(head, Head)

    def test_logistic_probe_satisfies_head_protocol(self):
        head = LogisticProbeHead()
        assert isinstance(head, Head)

    def test_mlp_satisfies_head_protocol(self):
        head = MLPHead(input_dim=4, hidden_dims=(8,), epochs=1)
        assert isinstance(head, Head)

    def test_predict_output_shapes(self):
        X, y, _w = _linear_signal(n=200, d=5, seed=0)
        Xb, yb = _binary_signal(n=200, d=5, seed=1)

        linear = LinearProbeHead().fit(X, y)
        logistic = LogisticProbeHead().fit(Xb, yb)
        mlp = MLPHead(input_dim=5, hidden_dims=(8,), epochs=3, batch_size=64).fit(X, y)

        assert linear.predict(X).shape == (200,)
        assert logistic.predict(Xb).shape == (200,)
        assert mlp.predict(X).shape == (200,)

    def test_concat_mode_is_just_column_concatenation(self):
        """Optional 'concat' mode (embeddings || factor panel) needs no special
        plumbing: any head treats all input columns identically, so the caller
        concatenating features before fit/predict is sufficient (documented in
        the heads.py module docstring).
        """
        rng = np.random.default_rng(21)
        emb = rng.standard_normal((300, 6))
        factors = rng.standard_normal((300, 3))
        w_emb = rng.standard_normal(6)
        w_fac = rng.standard_normal(3)
        y = emb @ w_emb + factors @ w_fac + 0.1 * rng.standard_normal(300)

        X_concat = np.concatenate([emb, factors], axis=1)
        head = LinearProbeHead().fit(X_concat, y)
        assert head.predict(X_concat).shape == (300,)
        assert head.coef_.shape == (9,)
