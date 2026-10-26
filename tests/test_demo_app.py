"""P3 demo app tests: loopback stdlib client only, no external network, fake-only.

Run (Windows, independent basetemp avoids interference with other local runs):

    python -m pytest tests/test_demo_app.py -q ^
        --basetemp=.pytest_tmp\\demo

These tests never call a real model, never download data and never leave a
long-running server: every server is started on an ephemeral loopback port and
shut down in the fixture; the CLI ``--check`` path is exercised as a subprocess
with a timeout and must exit by itself.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from demo_app import (
    DemoAppError,
    MAX_BODY_BYTES,
    MAX_CONNECTIONS_HARD_CAP,
    WindowRateLimiter,
    execute_demo,
    main,
    make_server,
    run_contract_check,
)
from demo_fixtures import (
    DEMO_MODE_IDS,
    DEMO_SCENARIOS,
    GT_MARKER,
    SCENARIO_BY_ID,
    DemoFixtureError,
    build_demo_provider,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web_demo"


# ---------------------------------------------------------------------------
# Loopback HTTP helpers (stdlib http.client only).
# ---------------------------------------------------------------------------


def _port(server) -> int:
    return int(server.server_address[1])


def _request(server, method: str, path: str, *, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", _port(server), timeout=30)
    try:
        merged = {}
        if headers:
            merged.update(headers)
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        data = response.read()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, data
    finally:
        connection.close()


def _raw_request(server, method: str, path: str, headers, body: bytes = b""):
    connection = http.client.HTTPConnection("127.0.0.1", _port(server), timeout=30)
    try:
        connection.putrequest(method, path)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        if body:
            connection.send(body)
        response = connection.getresponse()
        data = response.read()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, data
    finally:
        connection.close()


def _meta(server):
    status, _headers, data = _request(server, "GET", "/api/meta")
    assert status == 200
    payload = json.loads(data.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _csrf(server) -> str:
    token = _meta(server).get("csrf_token")
    assert isinstance(token, str) and len(token) >= 16
    return token


def _post_run(server, payload, token, *, origin=True, content_type="application/json", extra=None, raw=None):
    headers = {"Content-Type": content_type, "X-Demo-CSRF": token}
    if origin:
        headers["Origin"] = f"http://127.0.0.1:{_port(server)}"
    if extra:
        headers.update(extra)
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    return _request(server, "POST", "/api/run", body=body, headers=headers)


def _run_ok(server, scenario_id: str, mode_id: str = "all_modes"):
    status, _headers, data = _post_run(server, {"scenario": scenario_id, "mode": mode_id}, _csrf(server))
    assert status == 200, data[:400]
    return json.loads(data.decode("utf-8"))


def _body_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _forbidden_markers() -> tuple[str, ...]:
    return (GT_MARKER, "GROUNDTRUTH", "Traceback", "=== ROLE", "C:\\Users\\", "/Users/", "sk-", "api_key")


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner_root(tmp_path_factory):
    return tmp_path_factory.mktemp("p3-demo-runner")


@pytest.fixture(scope="module")
def demo_server(runner_root):
    server = make_server(
        "127.0.0.1", 0,
        runner=lambda scenario, mode: execute_demo(scenario, mode, temp_root=runner_root),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Static page / meta contracts.
# ---------------------------------------------------------------------------


def test_meta_index_static_and_security_headers(demo_server):
    status, headers, data = _request(demo_server, "GET", "/")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert b"SIMULATED" in data
    assert "content-security-policy" in headers
    assert "default-src 'none'" in headers["content-security-policy"]
    assert headers.get("x-content-type-options") == "nosniff"
    assert "access-control-allow-origin" not in headers

    for path, expected_type in (("/app.js", "text/javascript"), ("/styles.css", "text/css")):
        status, headers, data = _request(demo_server, "GET", path)
        assert status == 200, path
        assert headers["content-type"].startswith(expected_type)
        assert "content-security-policy" in headers

    meta = _meta(demo_server)
    assert meta["simulated"] is True
    assert meta["app"] == "p3-local-portfolio-demo"
    assert meta["all_modes_id"] == "all_modes"
    assert meta["all_modes_id"] in {entry["id"] for entry in meta["modes"]}
    assert meta["limits"]["max_connections"] >= 1
    assert {entry["id"] for entry in meta["scenarios"]} == {scenario.scenario_id for scenario in DEMO_SCENARIOS}
    assert {entry["id"] for entry in meta["modes"]} >= set(DEMO_MODE_IDS)
    assert meta["p2_panel"]["status_zh"] == "评测集成待验收"
    assert meta["p2_panel"]["real_llm_zh"] == "真实 LLM 实测未运行"
    assert meta["p2_panel"]["accuracy_zh"] == "未实测"

    status, headers, data = _request(demo_server, "GET", "/favicon.svg")
    assert status == 200
    assert headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in data
    assert b"<script" not in data
    assert b"href" not in data


def test_frontend_static_contracts_no_innerhtml_no_external_urls():
    app_js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    styles_css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    for source in (app_js, index_html, styles_css):
        assert "https://" not in source
        assert "http://" not in source
    assert "innerHTML" not in app_js
    assert "textContent" in app_js
    assert "/app.js" in index_html
    assert "/styles.css" in index_html
    assert "SIMULATED" in index_html
    assert "url(" not in styles_css


def test_static_allowlist_blocks_traversal(demo_server):
    for path in (
        "/static/../demo_app.py",
        "/%2e%2e/demo_app.py",
        "/etc/passwd",
        "/web_demo/app.js",
        "/../demo_app.py",
        "/app.js/../demo_app.py",
    ):
        status, _headers, data = _request(demo_server, "GET", path)
        assert status == 404, path
        assert b"Traceback" not in data


def test_host_header_protection(demo_server):
    port = _port(demo_server)
    status, _headers, _data = _request(demo_server, "GET", "/api/meta", headers={"Host": "evil.example"})
    assert status == 403
    status, _headers, _data = _request(
        demo_server, "GET", "/api/meta", headers={"Host": f"127.0.0.1:{port + 1}"},
    )
    assert status == 403
    status, _headers, _data = _request(demo_server, "GET", "/api/meta", headers={"Host": f"localhost:{port}"})
    assert status == 200
    status, _headers, _data = _request(demo_server, "GET", "/api/meta", headers={"Host": "127.0.0.1"})
    assert status == 200


# ---------------------------------------------------------------------------
# POST boundary: origin / CSRF / content type / body limits / malformed JSON.
# ---------------------------------------------------------------------------


def test_post_requires_json_content_type(demo_server):
    status, _headers, data = _post_run(
        demo_server, {"scenario": "cross_date_success", "mode": "No_Memory"}, _csrf(demo_server),
        content_type="text/plain",
    )
    assert status == 415
    assert json.loads(data.decode("utf-8"))["error"] == "unsupported_media_type"


def test_post_origin_and_csrf(demo_server):
    token = _csrf(demo_server)
    good_body = json.dumps({"scenario": "cross_date_success", "mode": "No_Memory"}).encode("utf-8")

    status, _headers, data = _post_run(
        demo_server, None, token, origin=False,
        extra={"Origin": "http://evil.example"}, raw=good_body,
    )
    assert status == 403
    assert json.loads(data.decode("utf-8"))["error"] == "forbidden_origin"

    status, _headers, data = _post_run(
        demo_server, None, "wrong-token-000000000000", origin=True, raw=good_body,
    )
    assert status == 403
    assert json.loads(data.decode("utf-8"))["error"] == "invalid_csrf"

    status, _headers, data = _raw_request(
        demo_server, "POST", "/api/run",
        {"Content-Type": "application/json"}, good_body,
    )
    assert status == 403
    assert json.loads(data.decode("utf-8"))["error"] == "invalid_csrf"

    status, _headers, data = _post_run(
        demo_server, None, token, origin=False, raw=good_body,
    )
    assert status == 200, data[:300]

    status, _headers, data = _post_run(
        demo_server, None, token, origin=True, raw=good_body,
    )
    assert status == 200, data[:300]


def test_body_limits_and_malformed_json(demo_server):
    token = _csrf(demo_server)
    port = _port(demo_server)
    base_headers = {"Content-Type": "application/json", "X-Demo-CSRF": token, "Origin": f"http://127.0.0.1:{port}"}

    status, _headers, data = _raw_request(
        demo_server, "POST", "/api/run",
        {**base_headers, "Content-Length": str(MAX_BODY_BYTES + 1)},
    )
    assert status == 413
    assert json.loads(data.decode("utf-8"))["error"] == "body_too_large"

    for raw_length in ("-5", "abc", "1.5", "+5", ""):
        status, _headers, data = _raw_request(
            demo_server, "POST", "/api/run",
            {**base_headers, "Content-Length": raw_length},
        )
        assert status == 400, (raw_length, data[:200])
        assert json.loads(data.decode("utf-8"))["error"] in {"invalid_content_length", "bad_request"}

    status, _headers, data = _raw_request(
        demo_server, "POST", "/api/run", {**base_headers, "Transfer-Encoding": "chunked"},
    )
    assert status == 400
    assert json.loads(data.decode("utf-8"))["error"] == "unsupported_transfer_encoding"

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.putrequest("POST", "/api/run")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("X-Demo-CSRF", token)
        connection.putheader("Origin", f"http://127.0.0.1:{port}")
        connection.putheader("Content-Length", "3")
        connection.putheader("Content-Length", "3")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read().decode("utf-8"))["error"] == "invalid_content_length"
    finally:
        connection.close()

    malformed_cases = [
        (b"{not json", "invalid_json"),
        (b'{"scenario":"cross_date_success","scenario":"x","mode":"No_Memory"}', "duplicate_json_field"),
        (json.dumps({"scenario": "cross_date_success", "mode": "No_Memory", "user_id": "u"}).encode(), "unknown_field"),
        (json.dumps({"scenario": "cross_date_success"}).encode(), "missing_field"),
        (b"[]", "not_json_object"),
        (json.dumps({"scenario": "evil<script>alert(1)</script>", "mode": "No_Memory"}).encode(), "invalid_scenario_value"),
        (json.dumps({"scenario": "cross_date_success", "mode": "<img src=x>"}).encode(), "invalid_mode_value"),
        (json.dumps({"scenario": True, "mode": "No_Memory"}).encode(), "invalid_scenario_type"),
    ]
    for raw, expected_code in malformed_cases:
        status, _headers, data = _post_run(demo_server, None, token, raw=raw)
        assert status == 400, (expected_code, data[:200])
        payload = json.loads(data.decode("utf-8"))
        assert payload["error"] == expected_code
        assert b"evil<script>" not in data
        assert b"<img" not in data

    nested = {"scenario": "cross_date_success", "mode": "No_Memory", "extra": {}}
    node = nested["extra"]
    for _ in range(12):
        node["next"] = {}
        node = node["next"]
    status, _headers, data = _post_run(demo_server, None, token, raw=json.dumps(nested).encode())
    assert status == 400
    assert json.loads(data.decode("utf-8"))["error"] == "json_too_deep"

    # A malformed request must not poison later normal requests.
    body = _run_ok(demo_server, "cross_date_success", "No_Memory")
    assert body["simulated"] is True


def test_method_and_route_allowlist(demo_server):
    token = _csrf(demo_server)
    status, headers, data = _request(demo_server, "PUT", "/api/run")
    assert status == 405
    assert "GET" in headers.get("allow", "")
    status, _headers, _data = _request(demo_server, "GET", "/api/run")
    assert status == 405
    status, _headers, _data = _post_run(demo_server, {"scenario": "cross_date_success", "mode": "No_Memory"}, token)
    assert status == 200
    status, _headers, data = _raw_request(
        demo_server, "POST", "/api/meta",
        {"Content-Type": "application/json", "X-Demo-CSRF": token, "Content-Length": "0"},
    )
    assert status == 405
    status, _headers, _data = _raw_request(
        demo_server, "POST", "/api/nope",
        {"Content-Type": "application/json", "X-Demo-CSRF": token, "Content-Length": "0"},
    )
    assert status == 404
    status, _headers, _data = _request(demo_server, "GET", "/api/nope")
    assert status == 404


# ---------------------------------------------------------------------------
# Run projections and scenario contracts.
# ---------------------------------------------------------------------------


def test_run_all_modes_projection_and_mode_toggles(demo_server):
    response = _run_ok(demo_server, "cross_date_success", "all_modes")
    assert response["simulated"] is True
    assert response["synthetic_session_id"].startswith("demosession-")
    mode_results = response["mode_results"]
    assert len(mode_results) == 4
    assert {entry["mode"]["id"] for entry in mode_results} == set(DEMO_MODE_IDS)
    toggles = {entry["mode"]["id"]: entry["mode"]["toggles"] for entry in mode_results}
    assert toggles["No_Memory"] == {"generated_memory": False, "trusted_stats": False}
    assert toggles["Generated_Review_Memory"] == {"generated_memory": True, "trusted_stats": False}
    assert toggles["Trusted_History_Stats"] == {"generated_memory": False, "trusted_stats": True}
    assert toggles["Combined"] == {"generated_memory": True, "trusted_stats": True}
    scopes = {entry["context_scope_sha256"] for entry in mode_results}
    assert len(scopes) == 1
    assert response["comparison"]["same_provider_contexts"] is True
    for entry in mode_results:
        assert entry["contract_passed"] is True
        assert all(entry["mode_checks"].values())
        assert entry["simulated"] is True
        assert re.fullmatch(r"[0-9a-f]{64}", entry["run_hash"])
        assert len(entry["tasks"]) == 3

    combined = next(entry for entry in mode_results if entry["mode"]["id"] == "Combined")
    no_memory = next(entry for entry in mode_results if entry["mode"]["id"] == "No_Memory")
    trusted = next(entry for entry in mode_results if entry["mode"]["id"] == "Trusted_History_Stats")
    first, second, third = combined["tasks"]
    assert first["memory"]["candidates"] == 0
    assert second["memory"]["candidates"] == 1 and second["memory"]["time_eligible"] == 1
    assert second["memory"]["selected"] == 1 and second["memory"]["prompt_included"] == 1
    assert second["memory"]["selected_entries"][0]["prompt_included"] is True
    assert second["memory"]["write"] == "written"
    assert third["memory"]["candidates"] == 2 and third["memory"]["time_eligible"] == 1
    assert no_memory["tasks"][1]["memory"]["candidates"] == 0
    assert no_memory["tasks"][1]["memory"]["write"] == "not_attempted"
    assert no_memory["tasks"][0]["render_summary"]["stats_rendered"] is False
    assert trusted["tasks"][0]["render_summary"]["stats_rendered"] is True
    assert trusted["tasks"][0]["render_summary"]["stats_included_by_mode"] is True
    assert no_memory["tasks"][0]["render_summary"]["stats_included_by_mode"] is False

    for entry in mode_results:
        for task in entry["tasks"]:
            assert set(task["actual"]["evidence_refs"]) <= set(task["rendered_evidence_ids"])
            assert task["actual"]["status"] == "success"
            assert task["expected"]["matches"] is True
            assert task["actual"]["review"].startswith("SYNTHETIC")

    text = json.dumps(response, ensure_ascii=False)
    for marker in _forbidden_markers():
        assert marker not in text
    assert response["p2_panel"]["accuracy_zh"] == "未实测"
    compare_rows = response["comparison"]["rows"]
    assert len(compare_rows) == 4
    assert all(row["contract_passed"] is True for row in compare_rows)


def test_failure_refusal_degradation_and_repair_contracts(demo_server):
    repaired = _run_ok(demo_server, "repair_once", "Combined")
    entry = repaired["mode_results"][0]
    task = entry["tasks"][0]
    assert task["actual"]["status"] == "success"
    assert task["actual"]["reason"] == "validated_model_output_repaired"
    assert task["llm"]["repair_requested"] is True and task["llm"]["repair_attempted"] is True
    assert task["budget_after_task"]["llm_attempts"] == 2
    assert entry["mode_checks"]["llm_attempts_match_expected"] is True
    assert entry["memory_store"]["writes"] == 1

    transport = _run_ok(demo_server, "transport_failed", "No_Memory")
    task = transport["mode_results"][0]["tasks"][0]
    assert task["actual"]["status"] == "failed"
    assert task["actual"]["reason"] == "llm_transport_error"
    assert task["actual"]["review"] is None and task["actual"]["stars"] is None
    assert task["budget_after_task"]["llm_attempts"] == 1
    assert task["expected"]["matches"] is True
    assert "SIMULATED_TRANSPORT_FAULT" not in json.dumps(transport, ensure_ascii=False)

    budget = _run_ok(demo_server, "budget_stop", "No_Memory")
    entry = budget["mode_results"][0]
    statuses = [task["actual"]["status"] for task in entry["tasks"]]
    assert statuses == ["degraded", "refused"]
    assert entry["tasks"][0]["actual"]["reason"] == "statistical_baseline_fallback"
    assert entry["tasks"][1]["actual"]["reason"] == "budget_exhausted"
    assert entry["budget_total"]["llm_attempts"] == 1
    assert entry["tasks"][1]["budget_after_task"]["llm_attempts"] == 1
    assert entry["mode_checks"]["script_consumed_matches_attempts"] is True

    insufficient = _run_ok(demo_server, "insufficient_evidence", "Combined")
    task = insufficient["mode_results"][0]["tasks"][0]
    assert task["actual"]["status"] == "refused"
    assert task["actual"]["reason"] == "insufficient_evidence"
    assert task["budget_after_task"]["llm_attempts"] == 0
    assert task["llm"]["draft_attempted"] is False
    assert insufficient["mode_results"][0]["memory_store"]["writes"] == 0

    render_gate = _run_ok(demo_server, "render_gate_refusal", "No_Memory")
    task = render_gate["mode_results"][0]["tasks"][0]
    assert task["actual"]["status"] == "refused"
    assert task["actual"]["reason"] == "no_rendered_evidence"
    assert task["budget_after_task"]["llm_attempts"] == 0

    injection = _run_ok(demo_server, "injection_filtered", "Combined")
    task = injection["mode_results"][0]["tasks"][0]
    assert task["actual"]["status"] == "success"
    assert task["evidence_counts"]["filtered_injection_rows"] >= 1
    for row in task["rendered_evidence"]:
        assert "Ignore all previous instructions" not in (row["text"] or "")


def test_same_day_and_user_isolation(demo_server):
    same_day = _run_ok(demo_server, "same_day_isolation", "Combined")
    tasks = same_day["mode_results"][0]["tasks"]
    assert tasks[0]["memory"]["candidates"] == 0
    assert tasks[1]["memory"]["candidates"] == 1
    assert tasks[1]["memory"]["time_eligible"] == 0
    assert tasks[1]["memory"]["selected"] == 0
    assert tasks[1]["memory"]["prompt_included"] == 0

    user_isolation = _run_ok(demo_server, "user_isolation", "Combined")
    tasks = user_isolation["mode_results"][0]["tasks"]
    tasks_by_user = {task["user_label_zh"]: task for task in tasks}
    assert set(tasks_by_user) == {"合成用户 A", "合成用户 B"}
    assert tasks_by_user["合成用户 B"]["memory"]["candidates"] == 0


def test_per_request_isolation_and_determinism(demo_server, runner_root):
    first = _run_ok(demo_server, "cross_date_success", "Combined")
    second = _run_ok(demo_server, "cross_date_success", "Combined")
    assert first["synthetic_session_id"] != second["synthetic_session_id"]
    first_entry = first["mode_results"][0]
    second_entry = second["mode_results"][0]
    assert first_entry["run_hash"] == second_entry["run_hash"]
    assert first_entry["tasks"][0]["memory"]["candidates"] == 0
    assert second_entry["tasks"][0]["memory"]["candidates"] == 0
    assert first_entry["memory_store"]["writes"] == 3
    assert second_entry["memory_store"]["writes"] == 3
    assert list(runner_root.iterdir()) == []


def test_contract_check_all_scenarios_and_statuses(tmp_path):
    report = run_contract_check(temp_root=tmp_path)
    assert report["contract_passed"] is True
    assert report["checks"]["all_four_statuses_seen"] is True
    assert report["checks"]["scenario_count_full"] is True
    assert set(report["statuses_seen"]) == {"success", "degraded", "refused", "failed"}
    assert len(report["scenario_reports"]) == len(DEMO_SCENARIOS) * len(DEMO_MODE_IDS)
    dumped = json.dumps(report, ensure_ascii=False)
    for marker in _forbidden_markers():
        assert marker not in dumped
    assert list(tmp_path.iterdir()) == []


def test_task_usage_delta_is_per_task_and_sums_to_total(demo_server, runner_root):
    response = _run_ok(demo_server, "cross_date_success", "Combined")
    entry = response["mode_results"][0]
    deltas = [task["task_usage_delta"]["llm_attempts"] for task in entry["tasks"]]
    cumulative = [task["budget_after_task"]["llm_attempts"] for task in entry["tasks"]]
    assert deltas == [1, 1, 1]
    assert cumulative == [1, 2, 3]
    assert entry["mode_checks"]["task_usage_delta_sums_to_total"] is True
    for task in entry["tasks"]:
        for key, value in task["task_usage_delta"].items():
            assert isinstance(value, int) and value >= 0, key
        assert task["task_usage_note_zh"]
    for key in entry["tasks"][0]["task_usage_delta"]:
        assert sum(task["task_usage_delta"][key] for task in entry["tasks"]) == entry["budget_total"][key]

    repaired = _run_ok(demo_server, "repair_once", "Combined")
    delta = repaired["mode_results"][0]["tasks"][0]["task_usage_delta"]
    assert delta["llm_attempts"] == 2
    # P1's ledger counts transport-level completions: the invalid draft call
    # and the repaired call both returned text, so successes == 2 here.
    assert delta["llm_successes"] == 2

    budget = _run_ok(demo_server, "budget_stop", "No_Memory")
    entry = budget["mode_results"][0]
    assert [task["task_usage_delta"]["llm_attempts"] for task in entry["tasks"]] == [1, 0]
    assert [task["budget_after_task"]["llm_attempts"] for task in entry["tasks"]] == [1, 1]


# ---------------------------------------------------------------------------
# Server hardening: internal errors, rate limit, inflight / connection bound.
# ---------------------------------------------------------------------------


def _start_temp_server(**kwargs):
    server = make_server("127.0.0.1", 0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_internal_error_is_safe_and_recovery_works():
    calls = {"count": 0}

    def flaky_runner(scenario, mode):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("SECRET_MARKER C:\\Users\\private /etc/passwd traceback")
        return {"simulated": True, "ok": True}

    server, thread = _start_temp_server(runner=flaky_runner)
    try:
        token = _csrf(server)
        status, _headers, data = _post_run(server, {"scenario": "cross_date_success", "mode": "No_Memory"}, token)
        assert status == 500
        body = _body_text(data)
        assert json.loads(body)["error"] == "internal_error"
        assert "SECRET_MARKER" not in body
        assert "C:\\Users" not in body
        assert "/etc/passwd" not in body
        assert "Traceback" not in body
        status, _headers, data = _post_run(server, {"scenario": "cross_date_success", "mode": "No_Memory"}, token)
        assert status == 200
        assert json.loads(data.decode("utf-8"))["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_rate_limit_is_bounded_and_safe():
    calls = {"count": 0}

    def counting_runner(scenario, mode):
        calls["count"] += 1
        return {"simulated": True}

    server, thread = _start_temp_server(runner=counting_runner, rate_limit=(2, 60.0))
    try:
        token = _csrf(server)
        payload = {"scenario": "cross_date_success", "mode": "No_Memory"}
        assert _post_run(server, payload, token)[0] == 200
        assert _post_run(server, payload, token)[0] == 200
        status, _headers, data = _post_run(server, payload, token)
        assert status == 429
        assert json.loads(data.decode("utf-8"))["error"] == "rate_limited"
        assert calls["count"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_inflight_limit_returns_safe_busy_and_recovers():
    release = threading.Event()
    entered = threading.Event()

    def blocking_runner(scenario, mode):
        entered.set()
        release.wait(10)
        return {"simulated": True}

    server, thread = _start_temp_server(runner=blocking_runner, max_inflight=1, inflight_wait=0.05)
    try:
        token = _csrf(server)
        payload = {"scenario": "cross_date_success", "mode": "No_Memory"}
        results = {}

        def first_request():
            results["first"] = _post_run(server, payload, token)

        worker = threading.Thread(target=first_request, daemon=True)
        worker.start()
        assert entered.wait(10)
        status, _headers, data = _post_run(server, payload, token)
        assert status == 503
        assert json.loads(data.decode("utf-8"))["error"] == "server_busy"
        release.set()
        worker.join(timeout=10)
        assert results["first"][0] == 200
        release.set()
        assert _post_run(server, payload, token)[0] == 200
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_connection_cap_rejects_before_thread_and_recovers():
    server, thread = _start_temp_server(
        runner=lambda scenario, mode: {"simulated": True},
        max_connections=2, connection_wait=0.05,
    )
    held: list[socket.socket] = []
    try:
        for _ in range(2):
            held.append(socket.create_connection(("127.0.0.1", _port(server)), timeout=10))
        time.sleep(0.3)  # let both permits be taken and their handler threads start
        third = socket.create_connection(("127.0.0.1", _port(server)), timeout=10)
        third.settimeout(10)
        try:
            data = third.recv(4096)
        except socket.timeout:
            data = b""
        third.close()
        assert data == b"" or b"503" in data
        assert b"server_busy" in data or data == b""

        for connection in held:
            try:
                connection.close()
            except OSError:
                pass
        held = []
        deadline = time.time() + 10
        recovered = False
        while time.time() < deadline:
            try:
                status, _headers, _data = _request(server, "GET", "/api/meta")
                if status == 200:
                    recovered = True
                    break
            except OSError:
                pass
            time.sleep(0.1)
        assert recovered
    finally:
        for connection in held:
            try:
                connection.close()
            except OSError:
                pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_handle_error_logs_safe_category_without_traceback(caplog, capsys):
    server, thread = _start_temp_server(runner=lambda scenario, mode: {"simulated": True})
    try:
        with caplog.at_level(logging.WARNING, logger="p3_demo"):
            try:
                raise RuntimeError("PRIVATE_MARKER C:\\Users\\secret /etc/passwd")
            except RuntimeError:
                server.handle_error(None, ("127.0.0.1", 12345))
        captured = capsys.readouterr()
        assert "connection_error" in caplog.text
        assert "PRIVATE_MARKER" not in caplog.text
        assert "Traceback" not in caplog.text
        assert "C:\\Users" not in caplog.text
        assert "PRIVATE_MARKER" not in captured.err
        assert "Traceback" not in captured.err
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_rate_limit_and_connection_config_hard_caps():
    with pytest.raises(DemoAppError) as excinfo:
        WindowRateLimiter(601, 60.0)
    assert excinfo.value.code == "invalid_rate_limit"
    with pytest.raises(DemoAppError):
        WindowRateLimiter(0, 60.0)
    with pytest.raises(DemoAppError):
        WindowRateLimiter(10, 3601.0)
    with pytest.raises(DemoAppError) as excinfo:
        make_server("127.0.0.1", 0, max_connections=0)
    assert excinfo.value.code == "invalid_connection_limit"
    with pytest.raises(DemoAppError):
        make_server("127.0.0.1", 0, max_connections=MAX_CONNECTIONS_HARD_CAP + 1)
    with pytest.raises(DemoAppError):
        make_server("127.0.0.1", 0, connection_wait=-1)
    with pytest.raises(DemoAppError):
        make_server("127.0.0.1", 0, rate_limit=(601, 60.0))


# ---------------------------------------------------------------------------
# Bind validation, CLI --check, port-in-use, temp ownership.
# ---------------------------------------------------------------------------


def test_external_bind_and_invalid_ports_rejected(tmp_path):
    with pytest.raises(DemoAppError) as excinfo:
        make_server("0.0.0.0", 0)
    assert excinfo.value.code == "host_not_allowed"
    with pytest.raises(DemoAppError) as excinfo:
        make_server("127.0.0.1", -1)
    assert excinfo.value.code == "invalid_port"
    assert main(["--host", "0.0.0.0", "--port", "0"]) == 2
    assert main(["--host", "127.0.0.1", "--port", "70000"]) == 2


def test_cli_check_exits_by_itself_and_cleans_temp(tmp_path):
    env = dict(os.environ)
    env["TEMP"] = str(tmp_path)
    env["TMP"] = str(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "demo_app.py"), "--check"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    assert "demo check ok" in completed.stdout
    leftovers = [entry.name for entry in tmp_path.iterdir() if entry.name.startswith(("p3-demo-check-", "p3-demo-fixture-"))]
    assert leftovers == []


def test_cli_check_busy_port_safe_category(tmp_path):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.0.1", 0))
        busy_port = holder.getsockname()[1]
        env = dict(os.environ)
        env["TEMP"] = str(tmp_path)
        env["TMP"] = str(tmp_path)
        env["TMPDIR"] = str(tmp_path)
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "demo_app.py"), "--check", "--port", str(busy_port)],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
        )
        assert completed.returncode == 2
        assert "port_in_use_or_unavailable" in completed.stderr
    finally:
        holder.close()


def test_fixture_temp_ownership_only_cleans_own_directory(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="p3-demo-fixture-", dir=str(tmp_path)) as temp:
        fixture_dir = Path(temp) / "fixture"
        fixture = build_demo_provider(fixture_dir)
        assert fixture.root == fixture_dir
        assert len(fixture.tasks) == 6
        assert fixture_dir.exists()
    assert sentinel.exists()
    assert [entry.name for entry in tmp_path.iterdir()] == ["keep.txt"]

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(DemoFixtureError) as excinfo:
        build_demo_provider(nonempty)
    assert excinfo.value.code == "fixture_root_not_empty"
    assert (nonempty / "existing.txt").read_text(encoding="utf-8") == "do not delete"


def test_scenario_catalog_expected_contracts_stay_fixed():
    assert len(DEMO_SCENARIOS) == 9
    ids = [scenario.scenario_id for scenario in DEMO_SCENARIOS]
    assert len(set(ids)) == len(ids)
    expected = {scenario.scenario_id: [entry.status for entry in scenario.expected] for scenario in DEMO_SCENARIOS}
    assert expected["cross_date_success"] == ["success", "success", "success"]
    assert expected["repair_once"] == ["success"]
    assert expected["transport_failed"] == ["failed"]
    assert expected["budget_stop"] == ["degraded", "refused"]
    assert expected["insufficient_evidence"] == ["refused"]
    assert SCENARIO_BY_ID["budget_stop"].expected_llm_attempts == 1
