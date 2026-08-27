from __future__ import annotations

import json
from pathlib import Path

from model_pipeline import ApprovalRecord, LoopRoute, LoopState, require_approval


PROJECT = Path(__file__).resolve().parent
plan = json.loads((PROJECT / "approval_plan_1000.json").read_text(encoding="utf-8"))
approval_path = PROJECT / "approvals" / "full-loop-1000.json"
approval = ApprovalRecord.load(approval_path)
require_approval(approval, expected_plan=plan)
state_path = PROJECT / "loop_state.json"
state = LoopState.resume(state_path)
if state is None:
    raise RuntimeError(f"missing loop state: {state_path}")
if state.route == LoopRoute.AWAITING_APPROVAL:
    state.advance(
        stage="measure",
        route=LoopRoute.CONTINUE,
        approval_record=str(approval_path),
        plan_sha256=approval.plan_sha256,
    ).save(state_path)
print(
    json.dumps(
        {
            "approval": str(approval_path),
            "plan_sha256": approval.plan_sha256,
            "route": state.route.value,
            "stage": state.stage,
            "verified": True,
        },
        sort_keys=True,
    )
)
