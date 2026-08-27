"""Turn arm B's winning cascade trial back into a servable artifact.

`quiet_materialize` unchanged: it refits the winner deterministically from the
same fit split and the same seeds, re-derives the operating points exactly as the
trial did, and refuses to emit a model whose calibration numbers drift more than
0.02 from the trial's own record. That check is the point of the module -- a
refit that does not reproduce its own trial is a bug, and it fails rather than
shipping a "better" model that is simply not the one the search selected.

This driver only redirects it at this run's project, the same way `h2h_steady`
redirects the search.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path("/home/lence/workspace/pii_master/projects/pii-head-to-head-v1")
os.environ["QUIET_PROJECT"] = str(PROJECT)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import quiet_materialize  # noqa: E402


def main() -> int:
    if quiet_materialize.PROJECT != PROJECT:
        raise SystemExit(f"materialiser pointed at {quiet_materialize.PROJECT}, not this run")
    return quiet_materialize.main()


if __name__ == "__main__":
    raise SystemExit(main())
