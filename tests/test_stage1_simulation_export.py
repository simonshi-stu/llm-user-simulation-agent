"""Stage-1 tests: independent simulation export -> SimulationRunV1 artifact.

Covers the export entry itself (dry-run, synthetic mode, output rules, safe
errors), the repository-anchored output gate (never guessed from the process
working directory), the part/write/publish semantics (no overwrite, no
half-written final file), the exported artifact (strict contract round-trip,
pseudonym pairing labels, resource-bounded diagnostics, forbidden
groundtruth/prompt/key fields) and the authorized cross-repo check that the
evaluation project validates the exported artifact through its standalone CLI.
Cross-repo tests skip when the sibling repository is absent; production code
never imports it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import amazon_p0
from user_simulator_project import export as export_module
from user_simulator_project import simulation_run_v1 as contract
from user_simulator_project.__main__ import main

EXPECTED_SCHEMA_HASH = "70256fa957a3103e2892f3e3a22e67b23b2a85c1a9ac66c05b18a7418f0da85c"
B_REPO = Path(__file__).resolve().parents[2] / "amazon-instruments-evaluation-agent"
EVIDENCE_REF_PATTERN = re.compile(r"ev_[0-9a-f]{16}")
USER_GROUP_PATTERN = re.compile(r"ug_[0-9a-f]{16}")
FORBIDDEN_KEYS = {
    "actual_stars",
    "groundtruth",
    "target_text",
    "raw_prompt",
    "prompt_text",
    "api_key",
    "file_path",
    "path",
    "target_review",
    "user_id",
    "item_id",
}


def _walk_keys(value) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(key)
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _test_repo(tmp_path: Path, monkeypatch) -> Path:
    """Explicit safe test root (simulates the module's real repository root)."""
    root = tmp_path / "testrepo"
    monkeypatch.setattr(export_module, "_REPO_ROOT", root)
    return root


def _export(tmp_path: Path, monkeypatch, name: str = "run-a", **kwargs):
    root = _test_repo(tmp_path, monkeypatch)
    output = root / "results" / name
    result = export_module.export_synthetic(output, **kwargs)
    return output, result


def _artifact_text(tmp_path: Path, monkeypatch, name: str = "publish-source") -> str:
    output, _ = _export(tmp_path, monkeypatch, name=name, max_tasks=1)
    return (output / export_module.ARTIFACT_FILENAME).read_text(encoding="utf-8")


def _clean_env(cache_root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in ("API", "KEY", "TOKEN", "SECRET"))
    }
    env["PYTHONPYCACHEPREFIX"] = str(cache_root)
    return env


def test_schema_hash_pinned_and_mirror_matches_canonical() -> None:
    assert contract.schema_hash() == EXPECTED_SCHEMA_HASH
    if not B_REPO.is_dir():
        pytest.skip("sibling evaluation repository not present")
    code = (
        "from amazon_eval_agent.contracts import simulation_run_v1 as c; "
        "print(c.schema_hash())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=B_REPO, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == EXPECTED_SCHEMA_HASH


def test_dry_run_prints_safe_plan_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "no LLM callable, no file writes" in captured.out
    assert "fake callable only" in captured.out
    assert "anchored to the repository root" in captured.out
    assert list(tmp_path.iterdir()) == []


def test_output_root_is_module_anchored_not_cwd(tmp_path, monkeypatch) -> None:
    # The production gate must not trust a directory named ``results`` that
    # only looks ignored because it sits under the current working directory.
    assert export_module._REPO_ROOT == Path(export_module.__file__).resolve().parents[1]
    assert export_module._ignored_output_root(
        export_module._REPO_ROOT / "results" / "x"
    )
    assert not export_module._ignored_output_root(tmp_path / "results" / "x")

    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    monkeypatch.chdir(fake_repo)
    output = fake_repo / "results" / "run"
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(output)
    assert exc.value.code == "output_dir_not_ignored"
    assert not output.exists()


def test_synthetic_export_writes_contract_valid_artifact(tmp_path, monkeypatch) -> None:
    output, result = _export(tmp_path, monkeypatch, max_tasks=2)
    artifact = output / export_module.ARTIFACT_FILENAME
    data = artifact.read_bytes()
    validation = contract.validate_payload(data)
    assert validation.valid, validation.errors
    assert result.artifact_sha256 == hashlib.sha256(data).hexdigest()
    assert result.schema_hash == EXPECTED_SCHEMA_HASH
    assert result.status_counts == {
        "success": 2,
        "degraded": 0,
        "refused": 0,
        "failed": 0,
    }
    assert {path.name for path in output.iterdir()} == {
        export_module.ARTIFACT_FILENAME
    }

    document = json.loads(data.decode("utf-8"))
    assert document["schema_version"] == "simulation_run.v1"
    assert document["source"]["kind"] == "fake_on_synthetic"
    assert document["source"]["producer"] == "llm-user-simulation-agent"
    assert document["source"]["model_source"] == "offline_injected_fake_callable"
    assert document["simulated"] is True
    assert document["record_count"] == 2
    keys = set(_walk_keys(document))
    assert not (keys & FORBIDDEN_KEYS)
    text = data.decode("utf-8")
    assert "user-A" not in text
    assert "TARGET_SOURCE" not in text
    assert "SYNTHETIC_GROUNDTRUTH" not in text

    records = document["records"]
    for record in records:
        assert record["status"] == "success"
        assert type(record["stars"]) is int and 1 <= record["stars"] <= 5
        assert isinstance(record["review"], str) and record["review"].strip()
        assert record["prediction_origin"] == "llm"
        assert all(
            EVIDENCE_REF_PATTERN.fullmatch(ref) for ref in record["evidence_refs"]
        )
        assert set(record["diagnostics"]) == {
            "llm_attempts",
            "repair",
            "prompt_bytes",
            "memory",
        }
        assert set(record["diagnostics"]["memory"]) == {
            "candidates",
            "time_eligible",
            "selected",
            "prompt_included",
            "filtered_injection",
            "write",
        }
        assert 0 <= record["diagnostics"]["prompt_bytes"] <= 65536
    # The same simulated user keeps one pseudonym pairing label across tasks.
    assert records[0]["user_group"] == records[1]["user_group"]
    assert USER_GROUP_PATTERN.fullmatch(records[0]["user_group"])


def test_export_refuses_existing_output_dir(tmp_path, monkeypatch) -> None:
    root = _test_repo(tmp_path, monkeypatch)
    output = root / "results" / "taken"
    output.mkdir(parents=True)
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(output)
    assert exc.value.code == "output_dir_exists"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_export_rejects_non_ignored_output_dir(tmp_path, monkeypatch) -> None:
    _test_repo(tmp_path, monkeypatch)
    output = tmp_path / "elsewhere" / "run"
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(output)
    assert exc.value.code == "output_dir_not_ignored"
    assert not output.exists()


def test_fixed_seed_and_task_bounds(tmp_path, monkeypatch) -> None:
    _test_repo(tmp_path, monkeypatch)
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(tmp_path / "results" / "s", seed=43)
    assert exc.value.code == "unsupported_seed"
    for bad in (0, export_module.MAX_TASKS_CAP + 1):
        with pytest.raises(export_module.ExportError) as exc:
            export_module.export_synthetic(
                tmp_path / "results" / f"t{bad}", max_tasks=bad
            )
        assert exc.value.code == "invalid_max_tasks"
    assert not (tmp_path / "results").exists()


def test_prepared_export_reuses_verified_loader_and_cleans_own_dir(
    tmp_path, monkeypatch
) -> None:
    root = _test_repo(tmp_path, monkeypatch)
    calls: list[str] = []

    def spy(prepared_dir):
        calls.append(str(prepared_dir))
        raise amazon_p0.AmazonP0Error("prepared_dir_missing")

    monkeypatch.setattr(export_module, "load_verified_prepared_dataset", spy)
    output = root / "results" / "prep"
    with pytest.raises(amazon_p0.AmazonP0Error):
        export_module.export_prepared(tmp_path / "nowhere", output, max_tasks=1)
    assert calls == [str(tmp_path / "nowhere")]
    assert not output.exists()


def test_failed_run_cleans_its_new_output_dir(tmp_path, monkeypatch) -> None:
    root = _test_repo(tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic internal boom")

    monkeypatch.setattr(export_module, "run_reliable_suite", boom)
    output = root / "results" / "boom"
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(output)
    assert exc.value.code == "export_failed"
    assert not output.exists()


def test_export_write_failure_cleans_whole_new_dir(tmp_path, monkeypatch) -> None:
    root = _test_repo(tmp_path, monkeypatch)

    def no_fsync(*args, **kwargs):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(export_module.os, "fsync", no_fsync)
    output = root / "results" / "write-fail"
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(output, max_tasks=1)
    assert exc.value.code == "artifact_write_failed"
    assert not output.exists()


def test_publish_refuses_late_existing_final(tmp_path, monkeypatch) -> None:
    text = _artifact_text(tmp_path, monkeypatch)
    target = tmp_path / "testrepo" / "late"
    target.mkdir(parents=True)
    final = target / export_module.ARTIFACT_FILENAME
    final.write_text("late marker", encoding="utf-8")
    with pytest.raises(export_module.ExportError) as exc:
        export_module._publish_artifact(target, text)
    assert exc.value.code == "output_file_exists"
    assert final.read_text(encoding="utf-8") == "late marker"
    assert not (target / (export_module.ARTIFACT_FILENAME + ".part")).exists()


def test_publish_cleans_part_when_fsync_fails(tmp_path, monkeypatch) -> None:
    text = _artifact_text(tmp_path, monkeypatch)

    def no_fsync(*args, **kwargs):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(export_module.os, "fsync", no_fsync)
    target = tmp_path / "testrepo" / "fsync-fail"
    target.mkdir(parents=True)
    with pytest.raises(export_module.ExportError) as exc:
        export_module._publish_artifact(target, text)
    assert exc.value.code == "artifact_write_failed"
    assert not (target / export_module.ARTIFACT_FILENAME).exists()
    assert not (target / (export_module.ARTIFACT_FILENAME + ".part")).exists()


def test_publish_fails_closed_when_hardlink_unavailable(tmp_path, monkeypatch) -> None:
    text = _artifact_text(tmp_path, monkeypatch)

    def no_link(*args, **kwargs):
        raise OSError("hardlink unsupported")

    monkeypatch.setattr(export_module.os, "link", no_link)
    target = tmp_path / "testrepo" / "no-hardlink"
    target.mkdir(parents=True)
    foreign = target / "foreign_marker.txt"
    foreign.write_text("keep", encoding="utf-8")
    with pytest.raises(export_module.ExportError) as exc:
        export_module._publish_artifact(target, text)
    assert exc.value.code == "atomic_publish_unavailable"
    assert not (target / export_module.ARTIFACT_FILENAME).exists()
    assert not (target / (export_module.ARTIFACT_FILENAME + ".part")).exists()
    assert foreign.read_text(encoding="utf-8") == "keep"


def test_publish_refuses_preexisting_part_without_touching_it(
    tmp_path, monkeypatch
) -> None:
    text = _artifact_text(tmp_path, monkeypatch)
    target = tmp_path / "testrepo" / "preexisting-part"
    target.mkdir(parents=True)
    part = target / (export_module.ARTIFACT_FILENAME + ".part")
    part.write_text("SYNTHETIC_PREEXISTING_PART", encoding="utf-8")
    with pytest.raises(export_module.ExportError) as exc:
        export_module._publish_artifact(target, text)
    assert exc.value.code == "part_file_exists"
    assert part.read_text(encoding="utf-8") == "SYNTHETIC_PREEXISTING_PART"
    assert not (target / export_module.ARTIFACT_FILENAME).exists()


def test_whole_export_preserves_late_foreign_final(tmp_path, monkeypatch) -> None:
    root = _test_repo(tmp_path, monkeypatch)
    output = root / "results" / "late-final"
    original = export_module.run_reliable_suite

    def racing(*args, **kwargs):
        (output / export_module.ARTIFACT_FILENAME).write_text(
            "SYNTHETIC_LATE_FOREIGN_FINAL", encoding="utf-8"
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(export_module, "run_reliable_suite", racing)
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(output, max_tasks=1)
    assert exc.value.code == "output_file_exists"
    final = output / export_module.ARTIFACT_FILENAME
    assert final.read_text(encoding="utf-8") == "SYNTHETIC_LATE_FOREIGN_FINAL"
    assert output.is_dir()  # preserved; never a recursive discard
    assert not (output / (export_module.ARTIFACT_FILENAME + ".part")).exists()


def test_whole_export_preserves_late_foreign_part(tmp_path, monkeypatch) -> None:
    root = _test_repo(tmp_path, monkeypatch)
    output = root / "results" / "late-part"
    original = export_module.run_reliable_suite

    def racing(*args, **kwargs):
        (output / (export_module.ARTIFACT_FILENAME + ".part")).write_text(
            "SYNTHETIC_PREEXISTING_PART", encoding="utf-8"
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(export_module, "run_reliable_suite", racing)
    with pytest.raises(export_module.ExportError) as exc:
        export_module.export_synthetic(output, max_tasks=1)
    assert exc.value.code == "part_file_exists"
    part = output / (export_module.ARTIFACT_FILENAME + ".part")
    assert part.read_text(encoding="utf-8") == "SYNTHETIC_PREEXISTING_PART"
    assert output.is_dir()
    assert not (output / export_module.ARTIFACT_FILENAME).exists()


def test_cli_failures_are_safe_codes(tmp_path, monkeypatch, capsys) -> None:
    _test_repo(tmp_path, monkeypatch)
    code = main(
        [
            "--synthetic",
            "--output-dir",
            str(tmp_path / "results" / "x"),
            "--seed",
            "43",
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert "unsupported_seed" in captured.err
    assert "Traceback" not in captured.err


def test_anonymize_user_group_is_deterministic_pseudonym() -> None:
    first = export_module.anonymize_user_group("user-A")
    assert first == export_module.anonymize_user_group("user-A")
    assert first != export_module.anonymize_user_group("user-B")
    assert USER_GROUP_PATTERN.fullmatch(first)
    assert "user-A" not in first


def test_build_document_rejects_status_inconsistent_records(
    tmp_path, monkeypatch
) -> None:
    output, _ = _export(tmp_path, monkeypatch, name="run-c", max_tasks=1)
    document = json.loads(
        (output / export_module.ARTIFACT_FILENAME).read_text(encoding="utf-8")
    )
    bad_record = document["records"][0]
    bad_record["stars"] = True
    with pytest.raises(contract.ContractError) as exc:
        contract.build_document(
            source_kind="fake_on_synthetic",
            dataset_fingerprint_sha256="a" * 64,
            manifest_fingerprint_sha256="b" * 64,
            config_fingerprint_sha256="c" * 64,
            records=[bad_record],
        )
    assert "invalid_stars" in exc.value.codes


def test_export_then_evaluation_project_validates_cross_repo(
    tmp_path, monkeypatch
) -> None:
    if not B_REPO.is_dir():
        pytest.skip("sibling evaluation repository not present")
    output, _ = _export(tmp_path, monkeypatch, name="run-b", max_tasks=3)
    artifact = output / export_module.ARTIFACT_FILENAME
    source = json.loads(artifact.read_text(encoding="utf-8"))["source"]
    env = _clean_env(tmp_path / "cache")
    env["PYTHONPATH"] = str(B_REPO)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "amazon_eval_agent",
            "validate",
            "--input",
            str(artifact),
            "--json",
        ],
        cwd=B_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["artifact_validated"] is True
    assert report["record_count"] == 3
    assert report["status_counts"]["success"] == 3
    assert report["all_predictions_succeeded"] is True
    assert report["real_measurement_status"] == "not_verified_or_not_run"
    assert report["fingerprint_verification"] == "declared_only_not_independently_verified"
    assert report["accuracy_claim"] == "none"
    assert report["scoring_available"] is False
    assert report["internal_consistency_note"]
    # The reader echoes exactly the declared source fingerprints.
    assert (
        report["fingerprints"]["dataset_fingerprint_sha256"]
        == source["dataset_fingerprint_sha256"]
    )
    assert (
        report["fingerprints"]["manifest_fingerprint_sha256"]
        == source["manifest_fingerprint_sha256"]
    )
    assert (
        report["fingerprints"]["config_fingerprint_sha256"]
        == source["config_fingerprint_sha256"]
    )
