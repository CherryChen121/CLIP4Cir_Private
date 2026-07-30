#!/usr/bin/env python3

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from model_output_migration import (  # noqa: E402
    MigrationBlockedError,
    SourceChangedError,
    apply_migration,
    build_migration_plan,
    finalize_source,
    find_legacy_writer_pids,
    migration_report_payload,
    scan_legacy_outputs,
    verify_migration,
)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _verification_payload(verification) -> dict:
    return {
        "ok": verification.ok,
        "checked_files": verification.checked_files,
        "checked_bytes": verification.checked_bytes,
        "errors": list(verification.errors),
    }


def _parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Audit, migrate, verify, and finalize legacy CLIP4Cir model outputs"
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Legacy model output directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Organized model output root",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--apply",
        action="store_true",
        help="Apply the audited migration and strict cleanup",
    )
    modes.add_argument(
        "--verify",
        action="store_true",
        help="Verify the existing migration manifest",
    )
    modes.add_argument(
        "--finalize",
        action="store_true",
        help="Verify and remove only the empty legacy source tree",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()

    if args.verify or args.finalize:
        verification = verify_migration(
            output_root / "migration_manifest.csv"
        )
        print(
            json.dumps(
                _verification_payload(verification),
                sort_keys=True,
                indent=2,
            )
        )
        if not verification.ok:
            return 1
        if args.finalize:
            finalize_source(source, verification)
        return 0

    writer_pids = find_legacy_writer_pids(PROJECT_ROOT)
    scan = scan_legacy_outputs(
        source,
        output_root,
        now=datetime.now(timezone.utc),
        pid_is_alive=_pid_is_alive,
        legacy_writer_pids=writer_pids,
    )
    plan = build_migration_plan(scan)
    if args.apply:
        plan = apply_migration(plan)
    print(
        json.dumps(
            migration_report_payload(plan),
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MigrationBlockedError, SourceChangedError, OSError) as error:
        print(f"migration blocked: {error}", file=sys.stderr)
        raise SystemExit(1)
