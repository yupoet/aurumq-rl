"""Tests for the MASTER-lite model-layer fixes in scripts/p3/master_train.py.

Pure-numpy helpers (masked market vector, per-date label z-score) run anywhere;
model-forward tests need torch and skip on CPU-only boxes without it (they run
on the 4070 as part of the bring-up).
"""

from __future__ import annotations

import numpy as np
import pytest
from p3.master_train import masked_market_vector, zscore_labels_per_date

# =============================================================================
# masked_market_vector
# =============================================================================


def test_market_vector_ignores_absent_zero_rows():
    # 2 present stocks with mean 2.0; 2 absent stocks sitting at 0 after the
    # NaN->0 fill would dilute the naive mean to 1.0
    x_anchor = np.array([[1.0], [3.0], [0.0], [0.0]], dtype=np.float32)
    present = np.array([True, True, False, False])
    naive = x_anchor.mean(axis=0)
    masked = masked_market_vector(x_anchor, present)
    np.testing.assert_allclose(naive, [1.0])
    np.testing.assert_allclose(masked, [2.0])


def test_market_vector_all_absent_falls_back_to_full_mean():
    x_anchor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    present = np.array([False, False])
    np.testing.assert_allclose(masked_market_vector(x_anchor, present), [2.0, 3.0])


def test_market_vector_dtype_and_shape():
    x_anchor = np.ones((5, 3), dtype=np.float32)
    out = masked_market_vector(x_anchor, np.ones(5, dtype=bool))
    assert out.shape == (3,) and out.dtype == np.float32


# =============================================================================
# zscore_labels_per_date
# =============================================================================


def test_label_zscore_normalizes_per_date_over_present():
    y = np.array([[1.0, 2.0, 3.0, 0.0], [10.0, 20.0, 30.0, 0.0]], dtype=np.float32)
    present = np.array([[True, True, True, False]] * 2)
    z = zscore_labels_per_date(y, present)
    for d in range(2):
        vals = z[d][present[d]]
        np.testing.assert_allclose(vals.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(vals.std(), 1.0, atol=1e-5)


def test_label_zscore_leaves_absent_cells_untouched():
    y = np.array([[1.0, 2.0, 7.5]], dtype=np.float32)
    present = np.array([[True, True, False]])
    z = zscore_labels_per_date(y, present)
    assert z[0, 2] == pytest.approx(7.5)


def test_label_zscore_constant_date_does_not_explode():
    y = np.full((1, 4), 0.02, dtype=np.float32)
    present = np.ones((1, 4), dtype=bool)
    z = zscore_labels_per_date(y, present)
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(z[0], 0.0, atol=1e-5)


def test_label_zscore_all_absent_date_is_inert():
    y = np.array([[0.5, -0.5]], dtype=np.float32)
    present = np.zeros((1, 2), dtype=bool)
    z = zscore_labels_per_date(y, present)
    np.testing.assert_allclose(z[0], [0.5, -0.5])


# =============================================================================
# MasterLite forward + padding mask (torch required)
# =============================================================================


def _tiny_model_and_input():
    torch = pytest.importorskip("torch")
    from p3.master_train import build_model

    torch.manual_seed(0)
    n, length, f, d = 6, 3, 4, 8
    model = build_model(n_factors=f, d_model=d, n_heads=2, dropout=0.0)
    model.eval()
    x = torch.randn(n, length, f)
    market = torch.randn(f)
    return torch, model, x, market


def test_forward_without_mask_returns_finite_scores():
    torch, model, x, market = _tiny_model_and_input()
    with torch.no_grad():
        scores = model(x, market)
    assert scores.shape == (x.shape[0],)
    assert torch.isfinite(scores).all()


def test_pad_mask_isolates_present_rows_from_absent_rows():
    torch, model, x, market = _tiny_model_and_input()
    pad = torch.zeros(x.shape[0], dtype=torch.bool)
    pad[-1] = True  # last stock is absent
    x_mut = x.clone()
    x_mut[-1] += 5.0  # perturb ONLY the absent stock

    with torch.no_grad():
        base = model(x, market, pad_mask=pad)
        masked_mut = model(x_mut, market, pad_mask=pad)
        unmasked = model(x, market)
        unmasked_mut = model(x_mut, market)

    # with the mask, mutating the absent row cannot leak into present rows
    torch.testing.assert_close(base[:-1], masked_mut[:-1])
    # sanity: without the mask the same mutation DOES leak (mask is load-bearing)
    assert not torch.allclose(unmasked[:-1], unmasked_mut[:-1])


def test_all_absent_pad_mask_falls_back_to_no_mask():
    torch, model, x, market = _tiny_model_and_input()
    pad = torch.ones(x.shape[0], dtype=torch.bool)
    with torch.no_grad():
        scores = model(x, market, pad_mask=pad)
    assert torch.isfinite(scores).all()
