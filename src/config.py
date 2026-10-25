"""Shared configuration constants for the application.

Keep the meaning of the cluster-count constants explicit:

* ``DEFAULT_K`` is the initial slider value used when the user starts an
  experiment.
* ``OPTIMAL_K`` is the evaluated optimum selected by the automatic option.

The two values are intentionally different.
"""

from __future__ import annotations

K_MIN = 2
K_MAX = 8
K_RANGE = range(K_MIN, K_MAX + 1)

DEFAULT_K = 4
OPTIMAL_K = 5
