#!/usr/bin/env python3
from argparse import ArgumentParser
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from output_dataset_reclassification import (  # noqa: E402
    ReclassificationBlockedError,
    TransactionError,
    apply_reclassification,
    build_reclassification_plan,
    finalize_reclassification,
    verify_reclassification,
)


def _verification_payload(mode, result):
    return {
        "mode": mode,
        "ok": result.ok,
        "run_counts": result.run_counts,
        "checkpoint_count": result.checkpoint_count,
        "retained_audit_files": result.retained_audit_files,
        "errors": list(result.errors),
    }


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()

    try:
        if args.apply:
            plan = build_reclassification_plan(output_root)
            result = apply_reclassification(
                plan, project_root=PROJECT_ROOT
            )
            payload = _verification_payload("apply", result)
            payload["total_runs"] = len(plan.actions)
            payload["dataset_counts"] = plan.dataset_counts
        elif args.verify:
            report_path = output_root / "migration_report.json"
            finalized = False
            if report_path.is_file():
                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                finalized = (
                    report.get("reclassification_state") == "finalized"
                )
            result = verify_reclassification(
                output_root, finalized=finalized
            )
            payload = _verification_payload("verify", result)
        elif args.finalize:
            result = finalize_reclassification(output_root)
            payload = _verification_payload("finalize", result)
        else:
            plan = build_reclassification_plan(output_root)
            payload = {
                "mode": "dry-run",
                "total_runs": len(plan.actions),
                "dataset_counts": plan.dataset_counts,
                "unresolved": len(plan.unresolved),
                "collisions": len(plan.collisions),
            }
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("ok", True) else 1
    except (
        OSError,
        ValueError,
        ReclassificationBlockedError,
        TransactionError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
