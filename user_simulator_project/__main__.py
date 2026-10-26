"""Independent stage-1 entry: ``python -m user_simulator_project ...``.

Modes (exactly one):

- ``--dry-run``: print the safe plan; no LLM callable, no file writes;
- ``--synthetic``: accepted synthetic fixture + deterministic fake callable,
  exported as ``fake_on_synthetic``;
- ``--prepared-dir PATH``: an explicit local P0 prepared directory loaded
  through every existing verification gate + deterministic fake callable,
  exported as ``fake_on_real_context``.

Everything is offline: no network client, no API key, no environment reads.
Stdout carries only safe numbers, fixed enums and digests: never reviews,
labels, paths or private error text. Exports go to a new directory under the
repository's ignored ``results/`` or ``Dataset/`` trees and refuse to
overwrite anything.
"""

from __future__ import annotations

import argparse
import sys

import amazon_p0
from temporal_context import TemporalManifestError

from .export import (
    DEFAULT_MAX_TASKS,
    FIXED_SEED,
    MAX_TASKS_CAP,
    ExportError,
    ExportResult,
    export_prepared,
    export_synthetic,
)

_DRY_RUN_LINES = (
    "stage1 dry run (no LLM callable, no file writes)",
    "mode --synthetic: accepted fixture provider + deterministic fake callable "
    "-> fake_on_synthetic artifact",
    "mode --prepared-dir: local verified prepared directory through every P0 "
    "gate (artifact size/row gates before parsing, policy, selection, prior, "
    "provider, split counts, target-copy exclusion) + deterministic fake "
    "callable -> fake_on_real_context artifact",
    "gates: output must be a new directory under this repository's results/ "
    "or Dataset/ (anchored to the repository root, not the process cwd); "
    "existing paths are refused; the artifact publishes via an owned .part "
    "file and an atomic hard-link step (fail-closed atomic_publish_unavailable "
    "when hard links are unsupported); a failed run removes only its own "
    "artifacts plus the directory when empty, preserving unknown/late files",
    f"limits: default max-tasks {DEFAULT_MAX_TASKS}, cap {MAX_TASKS_CAP}; seed "
    f"{FIXED_SEED} fixed; serial execution (max_workers=1)",
    "contract: simulation_run.v1 strict; resource gates before JSON parse; "
    "no groundtruth/labels/raw prompts/keys/paths in artifacts",
    "honesty: fake callable only; the artifact cannot measure real LLM "
    "quality, accuracy or memory effects",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="user_simulator_project",
        description=(
            "Offline stage-1 simulation export: run the P1 workflow with a "
            "deterministic fake callable and write one SimulationRunV1 "
            "artifact (no network, no API key)."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the safe plan; do not call any LLM or write files.",
    )
    mode.add_argument(
        "--synthetic",
        action="store_true",
        help="Export the accepted synthetic fixture with a fake callable.",
    )
    mode.add_argument(
        "--prepared-dir",
        metavar="PATH",
        help="Local verified P0 prepared directory (read-only) to export.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        help="New output directory under the repository's results/ or "
        "Dataset/ (anchored to the repo root; required for export modes).",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=DEFAULT_MAX_TASKS,
        help=f"Finite task prefix, 1..{MAX_TASKS_CAP} (default {DEFAULT_MAX_TASKS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help=f"Fixed seed {FIXED_SEED} (any other value is rejected).",
    )
    return parser


def _print_result(result: ExportResult) -> None:
    counts = result.status_counts
    print(
        "stage1 export ok: "
        f"mode={result.mode} source_kind={result.source_kind} "
        f"tasks={result.record_count} "
        f"success={counts.get('success', 0)} "
        f"degraded={counts.get('degraded', 0)} "
        f"refused={counts.get('refused', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"schema={result.schema_version} "
        f"artifact_sha256={result.artifact_sha256} "
        "fake_only=true real_llm_effect=not_testable accuracy_claim=none"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        for line in _DRY_RUN_LINES:
            print(line)
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required for export modes")
    try:
        if args.synthetic:
            result = export_synthetic(
                args.output_dir, max_tasks=args.max_tasks, seed=args.seed
            )
        else:
            result = export_prepared(
                args.prepared_dir,
                args.output_dir,
                max_tasks=args.max_tasks,
                seed=args.seed,
            )
    except ExportError as exc:
        print(f"stage1 export failed: {exc.code}", file=sys.stderr)
        return 2
    except amazon_p0.AmazonP0Error as exc:
        print(f"stage1 export failed: {exc.code}", file=sys.stderr)
        return 2
    except TemporalManifestError:
        print("stage1 export failed: runtime_manifest_rejected", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print("stage1 export failed: prepared_dir_missing", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - safe category only, no traceback
        print("stage1 export failed: internal_error", file=sys.stderr)
        return 2
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
