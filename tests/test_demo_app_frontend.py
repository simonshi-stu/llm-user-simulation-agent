"""Front-end contract tests for the local synthetic demo.

The tests run the real ``web_demo/app.js`` inside a small in-memory DOM stub
and exercise the server meta contract plus crafted sanitized run payloads.
No browser framework is required. When ``node`` is unavailable the behavioural
test skips; static HTML/JS/CSS contracts still run.
"""

from __future__ import annotations

import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading

import pytest

from demo_app import execute_demo, make_server

NODE = shutil.which("node")
REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web_demo"


@pytest.fixture(scope="module")
def runner_root(tmp_path_factory):
    return tmp_path_factory.mktemp("p3-frontend-runner")


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


def _meta(server):
    connection = http.client.HTTPConnection("127.0.0.1", int(server.server_address[1]), timeout=30)
    try:
        connection.request("GET", "/api/meta")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        return payload
    finally:
        connection.close()


def _task(ordinal: int, *, memory=None, delta=None, cumulative=None, review=None) -> dict:
    evidence_id = "ev_" + ("0" if ordinal == 1 else "1") * 16
    return {
        "task_ordinal": ordinal,
        "target_day": "2024-01-05",
        "user_label_zh": "合成用户 A",
        "item_label_zh": "合成商品 X",
        "actual": {
            "status": "success",
            "status_label_zh": "成功",
            "reason": "validated_model_output",
            "origin": "llm",
            "stars": 4,
            "review": review or "SYNTHETIC 合成演示输出（fake-only）",
            "evidence_refs": [evidence_id],
            "simulated": True,
        },
        "expected": {
            "status": "success",
            "reason": "validated_model_output",
            "label_zh": "成功",
            "matches": True,
            "note_zh": "预期合同：严格校验通过的成功结果。",
        },
        "memory": memory or {
            "candidates": 0, "time_eligible": 0, "selected": 0, "prompt_included": 0,
            "filtered_injection": 0, "write": "written", "write_reason": None,
            "selected_entries": [], "prompt_included_sequences": [],
        },
        "budget_after_task": cumulative or {
            "llm_attempts": 1, "llm_successes": 1, "llm_failures": 0, "tool_calls": 4,
            "prompt_bytes_charged": 100, "output_token_budget_charged": 700,
        },
        "task_usage_delta": delta or {
            "llm_attempts": 1, "llm_successes": 1, "llm_failures": 0, "tool_calls": 4,
            "prompt_bytes_charged": 100, "output_token_budget_charged": 700,
        },
        "task_usage_note_zh": "相邻累计快照差值",
        "llm": {
            "draft_attempted": True, "repair_requested": False, "repair_attempted": False,
            "repair_reserve_failure": None, "validation_reason_first": None,
            "validation_reason_after_repair": None,
        },
        "render_summary": {
            "prompt_sha256_short": "a" * 16, "prompt_bytes": 100,
            "history_rendered": 2, "reference_rendered": 1, "history_truncated": 0,
            "reference_truncated": 0, "memory_selected": 0, "memory_prompt_included": [],
            "memory_filtered_injection": 0, "memory_skipped_by_budget": 0,
            "stats_rendered": False, "stats_included_by_mode": False, "prior_rendered": True,
        },
        "evidence_counts": {
            "candidate_rows": 3, "tool_returned_rows": 3, "registered_rows": 3,
            "filtered_empty_rows": 0, "filtered_injection_rows": 0,
            "skipped_by_tool_cap_rows": 0, "tool_text_truncated_rows": 0,
            "tool_payload_bytes": 300, "cited_rows": 1,
        },
        "rendered_evidence": [{
            "evidence_id": evidence_id, "kind": "user_history",
            "event_time": "2024-01-01T00:00:00+00:00", "stars": 5.0,
            "text": "SECRET_EVIDENCE_MARKER 合成历史（仅用于 UI 测试）",
            "text_capped_for_ui": False,
        }],
        "rendered_evidence_ids": [evidence_id],
        "cited_registry": [{
            "evidence_id": evidence_id, "kind": "user_history",
            "event_time": "2024-01-01T00:00:00+00:00", "stars": 5.0,
            "text_sha256_short": "f" * 16, "text_chars_total": 30, "text_truncated": False,
        }],
    }


def _memory(candidates, eligible, selected, included):
    return {
        "candidates": candidates, "time_eligible": eligible, "selected": selected,
        "prompt_included": included, "filtered_injection": 0, "write": "written",
        "write_reason": None, "selected_entries": [], "prompt_included_sequences": [],
    }


def _crafted_response(meta) -> dict:
    budget_total = {
        "llm_attempts": 3, "llm_successes": 3, "llm_failures": 0, "tool_calls": 12,
        "prompt_bytes_charged": 300, "output_token_budget_charged": 2100,
        "prompt_tokens_estimate": 150, "actual_token_usage": "unknown",
        "estimated_cost_usd": "unknown",
    }
    spec = {
        "max_llm_calls_per_task": 2, "max_llm_calls_run": 40,
        "max_tool_calls_per_task": 8, "max_tool_calls_run": 200,
        "max_prompt_bytes_per_call": 16000, "max_prompt_bytes_run": 240000,
        "max_output_tokens_per_call": 700, "max_output_tokens_run": 14000,
        "deadline_seconds": None,
    }
    no_memory = {
        "mode": {"id": "No_Memory", "label_zh": "模式 A · No_Memory",
                 "toggles": {"generated_memory": False, "trusted_stats": False},
                 "contract_source": "temporal_context.TEMPORAL_ABLATION_CONTRACTS"},
        "simulated": True, "context_scope_sha256": "c" * 64, "run_hash": "a" * 64,
        "run_id_sha256_16": "b" * 16, "model_source": "offline_injected_fake_callable",
        "status_counts": {"success": 3, "degraded": 0, "refused": 0, "failed": 0},
        "contract_passed": True, "mode_checks": {"x": True},
        "tasks": [
            _task(1, memory=_memory(0, 0, 0, 0)),
            _task(2, memory=_memory(0, 0, 0, 0), cumulative={"llm_attempts": 2}),
            _task(3, memory=_memory(0, 0, 0, 0), cumulative={"llm_attempts": 3}),
        ],
        "budget_total": dict(budget_total), "budget_spec": spec, "memory_store": None,
    }
    combined = json.loads(json.dumps(no_memory))
    combined["mode"] = {
        "id": "Combined", "label_zh": "模式 D · Combined",
        "toggles": {"generated_memory": True, "trusted_stats": True},
        "contract_source": "temporal_context.TEMPORAL_ABLATION_CONTRACTS",
    }
    combined["run_hash"] = "d" * 64
    combined["tasks"] = [
        _task(1, memory=_memory(0, 0, 0, 0), review="SECRET_REVIEW_MARKER 合成评论"),
        _task(2, memory=_memory(1, 1, 1, 1), cumulative={"llm_attempts": 2}),
        _task(3, memory=_memory(1, 1, 1, 1), cumulative={"llm_attempts": 3}),
    ]
    combined["memory_store"] = {
        "entries": 3, "users": 1, "write_attempts": 3, "writes": 3,
        "rejected_by_reason": {},
    }
    return {
        "app": "p3-local-portfolio-demo",
        "schema_version": meta["schema_version"],
        "simulated": True,
        "synthetic_session_id": "demosession-abc123",
        "scenario": meta["scenarios"][0],
        "modes_run": ["No_Memory", "Combined"],
        "all_modes_id": meta["all_modes_id"],
        "context_scope_sha256": "c" * 64,
        "mode_results": [no_memory, combined],
        "comparison": {
            "same_provider_contexts": True,
            "note_zh": "同一上下文",
            "interpretation_zh": "差异只来自两个模式开关",
            "rows": [
                {
                    "mode_id": "No_Memory", "label_zh": "模式 A · No_Memory",
                    "generated_memory": False, "trusted_stats": False,
                    "status_counts": {"success": 3, "degraded": 0, "refused": 0, "failed": 0},
                    "llm_attempts": 3, "memory_prompt_included_total": 0,
                    "stats_rendered_tasks": 0, "contract_passed": True,
                },
                {
                    "mode_id": "Combined", "label_zh": "模式 D · Combined",
                    "generated_memory": True, "trusted_stats": True,
                    "status_counts": {"success": 3, "degraded": 0, "refused": 0, "failed": 0},
                    "llm_attempts": 3, "memory_prompt_included_total": 2,
                    "stats_rendered_tasks": 3, "contract_passed": True,
                },
            ],
        },
        "p2_panel": meta["p2_panel"],
        "safety_zh": meta.get("safety_zh", []),
        "honesty": [],
        "privacy_zh": "仅合成数据。",
    }


JS_HARNESS = r'''
"use strict";
const fs = require("fs");
const vm = require("vm");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const allNodes = [];
const registry = {};
const validIds = new Set(input.html_ids);
const blobs = [];

function makeElement(tag) {
  const node = {
    tagName: String(tag).toUpperCase(),
    id: "",
    children: [],
    attributes: {},
    listeners: {},
    textContent: "",
    className: "",
    disabled: false,
    hidden: false,
    checked: false,
    detached: false,
    parentNode: null,
    href: "",
    download: "",
    appendChild(child) {
      this.children.push(child);
      child.parentNode = this;
      markTree(child, false);
      return child;
    },
    removeChild(child) {
      const index = this.children.indexOf(child);
      if (index >= 0) this.children.splice(index, 1);
      child.parentNode = null;
      markTree(child, true);
      return child;
    },
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    },
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
    },
    removeAttribute(name) {
      delete this.attributes[name];
    },
    addEventListener(type, fn) {
      (this.listeners[type] = this.listeners[type] || []).push(fn);
    },
    focus() {
      documentStub.activeElement = this;
    },
    click() {
      const handlers = this.listeners.click || [];
      for (const handler of handlers) handler({ target: this });
    },
    remove() {
      if (this.parentNode) this.parentNode.removeChild(this);
    },
    querySelector() {
      return null;
    },
  };
  Object.defineProperty(node, "firstChild", {
    get() {
      return this.children.length > 0 ? this.children[0] : null;
    },
  });
  Object.defineProperty(node, "value", {
    get() {
      if (this._value !== undefined) return this._value;
      return this.attributes.value !== undefined ? this.attributes.value : "";
    },
    set(newValue) {
      this._value = String(newValue);
    },
  });
  return node;
}

function markTree(node, detached) {
  node.detached = detached;
  for (const child of node.children) markTree(child, detached);
}

const documentStub = {
  activeElement: null,
  body: makeElement("body"),
  lastAnchor: null,
  createElement(tag) {
    const element = makeElement(tag);
    allNodes.push(element);
    return element;
  },
  getElementById(id) {
    if (!validIds.has(id)) {
      throw new Error("app.js references an id missing from index.html: " + id);
    }
    if (!registry[id]) {
      const element = makeElement("div");
      element.id = id;
      registry[id] = element;
      allNodes.push(element);
    }
    return registry[id];
  },
  querySelector(selector) {
    if (selector === 'input[name="demo-mode"]:checked') {
      for (const node of allNodes) {
        if (node.tagName === "INPUT" && node.attributes.name === "demo-mode" && node.checked && !node.detached) {
          return node;
        }
      }
    }
    return null;
  },
  addEventListener() {},
};

const windowStub = {
  document: documentStub,
  setTimeout() {
    return 0;
  },
};

function Blob(parts, options) {
  this.parts = parts;
  this.type = options && options.type;
  blobs.push(this);
}

let fetchMode = "queue";
const fetchQueue = [];
async function fetchStub(path, options) {
  if (fetchMode === "reject") throw new Error("stub fetch failure");
  const item = fetchQueue.shift();
  if (!item) throw new Error("unexpected fetch: " + path);
  if (item.error) throw new Error(item.error);
  return {
    ok: item.ok !== false,
    status: item.status || 200,
    json: async function () { return item.json; },
  };
}

const sandbox = {
  document: documentStub,
  window: windowStub,
  URL: {
    createObjectURL() { return "blob:stub"; },
    revokeObjectURL() {},
  },
  Blob: Blob,
  fetch: fetchStub,
  console: console,
  __P3_TEST__: true,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(input.app_path, "utf8"), sandbox, { filename: "app.js" });

const hooks = sandbox.__P3_TEST_HOOKS__;
if (!hooks) throw new Error("test hooks missing");
const byId = hooks.byId;

function flatten(node) {
  let out = typeof node.textContent === "string" ? node.textContent : "";
  for (const child of node.children) out += " " + flatten(child);
  return out;
}

function findById(root, id) {
  if (root.id === id || root.attributes.id === id) return root;
  for (const child of root.children) {
    const found = findById(child, id);
    if (found) return found;
  }
  return null;
}

const checks = {};
function check(name, condition) {
  checks[name] = condition === true;
}

const meta = input.meta;
const response = input.response;

async function main() {
/* 1) every byId reference has a real HTML id */
check("all_byid_ids_resolve", input.byid_refs.every(function (id) { return !!documentStub.getElementById(id); }));

/* 2) meta success, default selection and display order */
const selection = hooks.applyMeta(meta);
check("default_all_modes", selection.modeId === "all_modes" && meta.all_modes_id === "all_modes");
check("default_scenario", selection.scenarioId !== "" && selection.runnable === true);
check("checked_is_default", hooks.selectedMode() === "all_modes");
check("all_modes_first", hooks.orderModesForDisplay(meta.modes, meta.all_modes_id)[0].id === "all_modes");
byId("run-button").disabled = true;
hooks.state.csrfToken = "x".repeat(32);
check("run_enabled", hooks.updateRunAvailability() === true && byId("run-button").disabled === false);

/* 3) defensive meta failures */
const metaNoAll = JSON.parse(JSON.stringify(meta));
delete metaNoAll.all_modes_id;
const fallback = hooks.applyMeta(metaNoAll);
check("fallback_first_mode", fallback.modeId === metaNoAll.modes[0].id && fallback.runnable === true);
check("fallback_checked", hooks.selectedMode() === metaNoAll.modes[0].id);
const emptySelection = hooks.applyMeta({ scenarios: [], modes: [] });
byId("run-button").disabled = false;
check("empty_meta_not_runnable", emptySelection.runnable === false);
check("empty_meta_disables_run", hooks.updateRunAvailability() === false && byId("run-button").disabled === true);
hooks.state.csrfToken = null;
check("missing_token_disables_run", hooks.updateRunAvailability() === false);
hooks.applyMeta(meta);
hooks.state.csrfToken = "x".repeat(32);

/* 4) successful meta init via fetch */
fetchQueue.push({ json: meta });
await hooks.init();
check("meta_init_success", hooks.state.csrfToken === meta.csrf_token &&
  byId("server-status").textContent.indexOf("已连接") >= 0 &&
  byId("run-button").disabled === false);

/* 5) full response render: overview -> visualization -> analysis -> summary */
hooks.state.lastResponse = response;
hooks.renderResponse(response);
check("overview_rendered", flatten(byId("run-summary")).indexOf("任务总数") >= 0 &&
  byId("overview-lead").textContent.indexOf("合成任务") >= 0);
check("state_success_count", byId("state-success-count").textContent === "3");
check("state_grid_values", byId("state-degraded-count").textContent === "0" &&
  byId("state-refused-count").textContent === "0" &&
  byId("state-failed-count").textContent === "0");
check("timeline_rendered", byId("timeline").children.length === 3);
check("detail_options", byId("detail-mode-select").children.length === 2);
check("compare_rendered", flatten(byId("compare-table")).indexOf("与预期一致") >= 0);
check("evidence_rendered", flatten(byId("task-panel")).indexOf("SECRET_EVIDENCE_MARKER") >= 0);
const noMemoryStats = flatten(byId("stats-panel"));
check("stats_untested", noMemoryStats.indexOf("未实测") >= 0);
check("stats_no_memory_disabled", noMemoryStats.indexOf("未启用") >= 0);
check("analysis_facts", flatten(byId("analysis-facts")).indexOf("四态计数") >= 0);
check("analysis_limits", flatten(byId("analysis-limits")).indexOf("不能说明模型准确性") >= 0);
check("summary_accuracy_untested", byId("summary-accuracy").textContent === "未实测");
check("summary_engineering", byId("summary-engineering").children.length >= 5);
check("summary_project_b", byId("summary-b-status").textContent.indexOf("真实指标未测") >= 0);
check("download_enabled", byId("download-button").disabled === false);

/* 6) mode change re-renders run-level stats, not just task details */
byId("detail-mode-select").value = "Combined";
hooks.onDetailModeChange();
const combinedStats = flatten(byId("stats-panel"));
check("stats_rerendered", combinedStats !== noMemoryStats);
check("combined_memory_counts", combinedStats.indexOf("候选合计：2") >= 0 &&
  combinedStats.indexOf("进入 prompt 合计：2") >= 0 &&
  combinedStats.indexOf("记忆库写入：3") >= 0);
check("state_strip_mode_note", byId("state-strip-note").textContent.indexOf("Combined") >= 0);

/* 7) task tabs, focus retention and keyboard navigation */
hooks.state.activeTaskOrdinal = 1;
hooks.state.activeTaskTab = "evidence";
hooks.state.focusAfterRender = null;
hooks.renderTaskArea();
const tabs = byId("task-tabs");
check("task_tab_count", tabs.children.length === 3);
check("task_roving_tabindex", tabs.children[0].getAttribute("tabindex") === "0" &&
  tabs.children[1].getAttribute("tabindex") === "-1");
tabs.children[1].listeners.click[0]();
check("task_click_sets_ordinal", hooks.state.activeTaskOrdinal === 2);
check("task_focus_retained", documentStub.activeElement === tabs.children[1]);
tabs.listeners.keydown[0]({ key: "ArrowRight", target: tabs.children[1], preventDefault() {} });
check("task_arrow_moves", hooks.state.activeTaskOrdinal === 3);
check("task_arrow_focus", documentStub.activeElement === tabs.children[2]);
tabs.listeners.keydown[0]({ key: "Home", target: tabs.children[2], preventDefault() {} });
check("task_home_moves", hooks.state.activeTaskOrdinal === 1);
check("task_home_focus", documentStub.activeElement === tabs.children[0]);

/* 8) memory funnel and budget subtab render from data */
const memorySubtab = findById(byId("task-panel"), "subtab-memory");
memorySubtab.listeners.click[0]();
const memoryText = flatten(byId("task-panel"));
check("memory_funnel_rendered", memoryText.indexOf("候选") >= 0 &&
  memoryText.indexOf("时间合格") >= 0 && memoryText.indexOf("进入 prompt") >= 0 &&
  memoryText.indexOf("写入") >= 0);
check("memory_focus_retained", documentStub.activeElement === findById(byId("task-panel"), "subtab-memory"));
const budgetSubtab = findById(byId("task-panel"), "subtab-budget");
budgetSubtab.listeners.click[0]();
const budgetText = flatten(byId("task-panel"));
check("budget_labels", budgetText.indexOf("本任务真实用量") >= 0 &&
  budgetText.indexOf("截至本任务累计") >= 0 &&
  budgetText.indexOf("预算占用") >= 0);
const budgetPanel = findById(byId("task-panel"), "subtab-panel");
check("subtab_aria", !!budgetPanel && budgetPanel.getAttribute("aria-labelledby") === "subtab-budget");

/* 9) run success and safe error path */
fetchQueue.push({ json: response });
const runOk = await hooks.runDemo();
check("run_success", runOk === true && !!hooks.state.lastResponse &&
  byId("status-region").textContent.indexOf("运行完成") >= 0);
fetchQueue.push({ error: "模拟网络失败" });
const runFailed = await hooks.runDemo();
check("run_error", runFailed === false && byId("error-region").hidden === false &&
  byId("error-region").textContent.indexOf("运行失败") >= 0 &&
  flatten(byId("error-region")).indexOf("模拟网络失败") >= 0);

/* 10) meta fetch failure and incomplete meta are friendly, not fake-ready */
fetchQueue.push({ error: "meta 服务不可达" });
await hooks.init();
check("meta_fetch_failure", byId("server-status").textContent.indexOf("未连接") >= 0 &&
  byId("error-region").hidden === false);
fetchQueue.push({ json: { scenarios: [], modes: [] } });
await hooks.init();
check("meta_incomplete", hooks.state.csrfToken === null &&
  byId("run-button").disabled === true &&
  byId("error-region").hidden === false);

/* 11) downloadable report is sanitized and marks accuracy untested */
hooks.state.lastResponse = response;
const report = hooks.buildReport(response);
const reportText = JSON.stringify(report);
check("report_accuracy", report.accuracy_zh === "未实测" && report.summary.accuracy_zh === "未实测");
check("report_no_review_text", reportText.indexOf("SECRET_REVIEW_MARKER") < 0);
check("report_no_evidence_text", reportText.indexOf("SECRET_EVIDENCE_MARKER") < 0);
check("report_no_csrf_or_path", reportText.indexOf("csrf") < 0 && reportText.indexOf("C:\\\\Users") < 0);
check("report_has_aggregates", reportText.indexOf("p3-local-portfolio-demo") >= 0 &&
  reportText.indexOf("demosession-abc123") >= 0 && reportText.indexOf("未实测") >= 0);
const downloaded = hooks.downloadReport();
const blobText = blobs.length > 0 ? String(blobs[blobs.length - 1].parts[0]) : "";
check("download_called", downloaded === true && blobText.indexOf("SECRET_REVIEW_MARKER") < 0 &&
  blobText.indexOf("SECRET_EVIDENCE_MARKER") < 0);

process.stdout.write(JSON.stringify({ checks: checks }));
}

main().catch(function (err) {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
'''


def _run_harness(demo_server, tmp_path) -> dict:
    if NODE is None:
        pytest.skip("node is not available; static contracts still run")
    meta = _meta(demo_server)
    response = _crafted_response(meta)
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    html_ids = sorted(set(re.findall(r'\bid="([^"]+)"', html)))
    byid_refs = sorted(set(re.findall(r'byId\("([^"]+)"\)', js)))
    payload = {
        "app_path": str(WEB_DIR / "app.js"),
        "html_ids": html_ids,
        "byid_refs": byid_refs,
        "meta": meta,
        "response": response,
    }
    harness = tmp_path / "p3_frontend_harness.js"
    harness.write_text(JS_HARNESS, encoding="utf-8")
    completed = subprocess.run(
        [NODE, str(harness)],
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[-4000:]
    return json.loads(completed.stdout.decode("utf-8"))


def test_node_contract_render_run_and_report(demo_server, tmp_path):
    result = _run_harness(demo_server, tmp_path)
    checks = result["checks"]
    failed = sorted(key for key, value in checks.items() if value is not True)
    assert failed == [], "failed front-end checks: " + ", ".join(failed)


def _css_block(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"missing CSS selector: {selector}"
    return match.group(1)


def test_static_ids_and_structure():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    refs = set(re.findall(r'byId\("([^"]+)"\)', js))
    assert refs - ids == set()
    for required in (
        "overview-card", "visualization", "analysis", "summary",
        "run-summary", "state-strip", "timeline", "task-tabs", "task-panel",
        "compare-table", "stats-panel", "analysis-facts", "analysis-limits",
        "summary-accuracy", "summary-engineering", "summary-improvements",
        "summary-b-status", "download-button", "run-button",
    ):
        assert f'id="{required}"' in html, required
    order = [html.index(f'id="{name}"') for name in ("overview-card", "visualization", "analysis", "summary")]
    assert order == sorted(order)
    assert "未实测" in html
    assert "独立评测项目 B" in html
    assert "合成 Demo 已上线" in html
    assert "真实指标未测" in html
    assert "RMSE" in html


def test_static_security_and_report_boundaries():
    app_js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    styles_css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    for source in (app_js, index_html, styles_css):
        assert "https://" not in source
        assert "http://" not in source
    for source in (app_js, index_html):
        assert "innerHTML" not in source
    assert "textContent" in app_js
    assert "document.write" not in app_js
    assert "eval(" not in app_js
    assert "/app.js" in index_html
    assert "/styles.css" in index_html
    assert "SIMULATED" in index_html
    assert "url(" not in styles_css
    assert "/api/meta" in app_js and "/api/run" in app_js


def test_mobile_and_desktop_css_layout_contract_no_overflow_mask():
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    inline_block = _css_block(css, ".inline-control")
    assert "flex-wrap: wrap" in inline_block
    assert "min-width: 0" in inline_block
    assert "max-width: 100%" in inline_block

    inline_select_block = _css_block(css, ".inline-control select")
    assert "max-width: 100%" in inline_select_block
    assert "min-width: 0" in inline_select_block

    assert "@media (max-width: 560px)" in css
    mobile_block = css.split("@media (max-width: 560px)", 1)[1]
    assert "flex-direction: column" in mobile_block
    assert "@media (max-width: 880px)" in css
    assert "overflow-x: hidden" not in css

    table_wrap_block = _css_block(css, ".table-wrap")
    assert "overflow-x: auto" in table_wrap_block
    mono_block = _css_block(css, ".mono")
    assert "overflow-wrap: anywhere" in mono_block
    assert ".state-grid" in css and ".analysis-grid" in css and ".summary-grid" in css
    assert 'rel="icon" href="/favicon.svg"' in html
    assert "width=device-width" in html
    assert "max-width: 1440px" in css
