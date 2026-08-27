"""The steady-aim lineage's search, pinned to the arm's read window.

`tune_quiet` is used unchanged -- same families, same objective, same optuna
sampler, same fit caching, same checkpointing. This driver only fixes the two
things the head-to-head fixes for every arm:

* **the read profile** is pinned to `deep` (12,000 characters), because arm B is
  defined as the steady-aim model at its shipped read window. Letting the search
  wander between `fast`, `std` and `deep` would make the arm's read window an
  outcome of the search rather than a property of the arm, and arm C -- the
  fusion at 12,000 -- would no longer be a controlled comparison against it.
  The side effect is that all 1,000 trials land on one profile instead of
  spreading over three, which is more search per profile, not less.

* **the project and the experiment**, so trials, `best.json` and the MLflow runs
  land in this run's directory rather than the lineage's original one.

The lineage's known blind spot is inherited deliberately and recorded in the
spec: no `fast` or `std` cascade has ever been evaluated in it, and pinning to
`deep` does not change that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-head-to-head-v1")
# Must be set before `tune_quiet` is imported: it reads QUIET_PROJECT at module
# scope to place TUNING, and an import-order mistake here would silently write a
# thousand trials into the wrong lineage's directory.
os.environ["QUIET_PROJECT"] = str(PROJECT)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402

from training import tune_quiet  # noqa: E402

#: Arm B is the steady-aim model at its shipped 12,000-character window.
tune_quiet.PROFILES = ("deep",)

_set_experiment = mlflow.set_experiment
mlflow.set_experiment = lambda *_a, **_k: _set_experiment("pii-head-to-head-v1")


def main() -> int:
    if tune_quiet.PROFILES != ("deep",):
        raise SystemExit("profile pin was lost; arm B would not be at 12,000 chars")
    return tune_quiet.main()


if __name__ == "__main__":
    raise SystemExit(main())
