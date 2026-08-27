"""Persist approval, loop state, run.json, and open the parent MLflow run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_pipeline import (  # noqa: E402
    ApprovalRecord,
    LoopRoute,
    LoopState,
    require_approval,
    set_cpus,
    tracking,
)
from runjson import RUN_JSON, init_run, record_command, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN = {
    "operation": "full-loop",
    "project": "projects/pii-stage2",
    "modules": ["eval", "tune", "datagen"],
    "budget": {
        "max_trials": 30,
        "max_wallclock_s": 28800,
        "target": {"macro_f2": 0.92, "severity_recall_min": 0.90},
    },
    "editable_surfaces": [
        "src/pii_master/",
        "training/",
        "eval/",
    ],
    "target": "projects/pii-stage2",
    "tracking": "file-store ./mlruns",
    "deploy": "none yet",
}


def main() -> int:
    set_cpus(1)
    init_run()
    approval = ApprovalRecord.create(PLAN, approved_by="user")
    path = approval.save(ROOT / "approvals" / "full-loop.json")
    require_approval(approval, expected_plan=PLAN)
    state = LoopState(
        project="pii-stage2",
        stage="measure",
        round_no=0,
        route=LoopRoute.CONTINUE,
        approval_gates_hit=1,
    )
    with tracking.pipeline_run("pii-stage2", experiment="pii-stage2") as parent:
        state.mlflow_parent_run_id = parent.info.run_id
        state.save(ROOT / "loop_state.json")
        data = json.loads(RUN_JSON.read_text())
        data.setdefault("artifacts", {})["mlflow_parent_run_id"] = parent.info.run_id
        save(data)
        record_command(
            "projects/pii-stage2/scripts/init_run.py",
            output=json.dumps({
                "approval": str(path),
                "plan_sha256": approval.plan_sha256,
                "mlflow_parent": parent.info.run_id,
            }),
            context="setup",
            cwd=str(ROOT.parent.parent),
        )
        print(json.dumps({
            "approval": str(path),
            "plan_sha256": approval.plan_sha256,
            "mlflow_parent_run_id": parent.info.run_id,
            "loop_state": str(ROOT / "loop_state.json"),
        }, indent=2))
        # Parent must stay open across later processes — persist the id;
        # this process ends the run. Child measure scripts log sibling runs
        # under the same experiment rather than nesting after the parent closes.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
