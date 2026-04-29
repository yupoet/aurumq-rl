"""GTJA Alpha 191 (国泰君安 191) factor library.

Each factor is implemented in a ``batch_NNN_MMM`` submodule, and
**self-registers** at import time by calling
``register_gtja191(FactorEntry(...))`` at module bottom. Importing
this package triggers all batch modules to be imported, which
populates ``aurumq_rl.factors.registry.GTJA191_REGISTRY`` for
downstream consumers.

Quality flags
-------------
* ``0`` — formula matches the Guotai Junan paper, no known errata.
* ``1`` — errata-conservative: paper formula has known issues
  (``wpwp/Alpha-101-GTJA-191`` errata list). Best-effort impl.
* ``2`` — stub: paper formula is ambiguous (e.g. ``gtja_143`` recursive
  SELF reference). Output is all-null.

Benchmark factors (gtja_149 / gtja_181 / gtja_182) currently use
cross-section mean of close as a CSI300 proxy, matching Daic115.
Production wiring to the real CSI300 OHLC is a Phase D task.
"""

from aurumq_rl.factors.registry import GTJA191_REGISTRY as REGISTRY

# G191A batches (gtja_001 .. gtja_100)
# G191B batches (gtja_101 .. gtja_191)
from . import (  # noqa: F401  (G191A)  # noqa: F401  (G191B)
    batch_001_020,
    batch_021_040,
    batch_041_060,
    batch_061_080,
    batch_081_100,
    batch_101_120,
    batch_121_140,
    batch_141_160,
    batch_161_180,
    batch_181_191,
)

__all__ = ["REGISTRY"]
