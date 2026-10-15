"""P3 front-end contract tests: Node DOM-stub behavior + CSS layout contracts.

No browser dependency is introduced. The JS behavior (default selection from
the meta contract, stats re-render on detail-mode change, task/subtab focus
retention and basic tab keyboard behavior, post-run Combined/budget-subtab DOM
structure) runs in Node inside a minimal in-memory DOM stub; these tests skip
when ``node`` is unavailable. The mobile (375px) layout fix is pinned by CSS
contract assertions. Real browser rendering at 375x812 and the post-run
screens are re-verified by the main session in a real browser.
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
WEB_DIR = Path(__file__).resolve().parents[1] / "web_demo"


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------


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


def _task(ordinal: int, *, memory=None, delta=None, cumulative=None) -> dict:
    evidence_id = "ev_" + "0" * 16
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
            "review": "SYNTHETIC 合成演示输出（fake-only）",
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
        "task_usage_note_zh": "差值口径",
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
            "text": "合成历史", "text_capped_for_ui": False,
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
        _task(1, memory=_memory(0, 0, 0, 0)),
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
            "same_provider_contexts": True, "note_zh": "同一上下文",
            "interpretation_zh": "合同差异", "rows": [],
        },
        "p2_panel": meta["p2_panel"],
        "safety_zh": [],
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

function markTree(node, detached) {
  node.detached = detached;
  for (const child of node.children) markTree(child, detached);
}

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

const documentStub = {
  activeElement: null,
  body: makeElement("body"),
  createElement(tag) {
    const element = makeElement(tag);
    allNodes.push(element);
    return element;
  },
  getElementById(id) {
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
      return null;
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

const sandbox = {
  document: documentStub,
  window: windowStub,
  URL: { createObjectURL() { return "blob:stub"; }, revokeObjectURL() {} },
  Blob: function Blob() {},
  fetch: function fetch() {
    throw new Error("fetch must not be called in the DOM stub test");
  },
  console: console,
  __P3_TEST__: true,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
const source = fs.readFileSync(input.app_path, "utf8");
vm.runInContext(source, sandbox, { filename: "app.js" });
const hooks = sandbox.__P3_TEST_HOOKS__;
if (!hooks) {
  console.error("test hooks missing");
  process.exit(3);
}
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
const meta = input.meta;
const response = input.response;

/* 1) default selection from the real server meta contract */
const selection = hooks.applyMeta(meta);
checks.default_all_modes = selection.modeId === "all_modes" && meta.all_modes_id === "all_modes";
checks.default_scenario = selection.scenarioId !== "" && selection.runnable === true;
checks.checked_is_default = hooks.selectedMode() === "all_modes";
checks.sorted_all_modes_first = hooks.orderModesForDisplay(meta.modes, meta.all_modes_id)[0].id === "all_modes";
byId("run-button").disabled = true;
hooks.state.csrfToken = "x".repeat(32);
checks.run_enabled_by_default = hooks.updateRunAvailability() === true && byId("run-button").disabled === false;

/* 2) defensive default when all_modes_id is missing */
const metaNoAll = JSON.parse(JSON.stringify(meta));
delete metaNoAll.all_modes_id;
const fallback = hooks.applyMeta(metaNoAll);
checks.fallback_first_mode = fallback.runnable === true && fallback.modeId === metaNoAll.modes[0].id;
checks.fallback_checked = hooks.selectedMode() === metaNoAll.modes[0].id;

/* 3) empty meta disables run instead of pretending ready */
const emptySelection = hooks.applyMeta({ scenarios: [], modes: [] });
byId("run-button").disabled = false;
checks.empty_not_runnable = emptySelection.runnable === false;
checks.empty_disables_run = hooks.updateRunAvailability() === false && byId("run-button").disabled === true;

/* restore the real meta */
hooks.applyMeta(meta);
hooks.state.csrfToken = "x".repeat(32);

/* 4) stats re-render on detail-mode change (No_Memory vs Combined) */
hooks.state.lastResponse = response;
hooks.state.detailModeId = "No_Memory";
hooks.renderStats();
const noMemoryStats = flatten(byId("stats-panel"));
byId("detail-mode-select").value = "Combined";
hooks.onDetailModeChange();
const combinedStats = flatten(byId("stats-panel"));
checks.stats_rerendered_on_mode_change = noMemoryStats !== combinedStats;
checks.nomem_disabled_label = noMemoryStats.indexOf("未启用") >= 0 && noMemoryStats.indexOf("记忆库写入：3") < 0;
checks.combined_counts = combinedStats.indexOf("候选合计：2") >= 0 &&
  combinedStats.indexOf("进入 prompt 合计：2") >= 0 &&
  combinedStats.indexOf("记忆库写入：3") >= 0;

/* 5) full post-run render + Combined + budget subtab DOM structure */
hooks.state.lastResponse = response;
hooks.renderResponse(response);
byId("detail-mode-select").value = "Combined";
hooks.onDetailModeChange();
const budgetTab = findById(byId("task-panel"), "subtab-budget");
checks.budget_subtab_present = !!budgetTab;
if (budgetTab) budgetTab.listeners.click[0]();
checks.budget_subtab_active = hooks.state.activeTaskTab === "budget";
const subtabPanel = findById(byId("task-panel"), "subtab-panel");
checks.subtab_panel_labelled = !!subtabPanel && subtabPanel.getAttribute("aria-labelledby") === "subtab-budget";
const taskPanelText = flatten(byId("task-panel"));
checks.budget_labels = taskPanelText.indexOf("本任务真实用量") >= 0 && taskPanelText.indexOf("截至本任务累计") >= 0;

/* 6) task tab focus retention + arrow keys + roving tabindex */
hooks.state.activeTaskOrdinal = 1;
hooks.state.activeTaskTab = "evidence";
hooks.state.focusAfterRender = null;
hooks.renderTaskArea();
const tabs = byId("task-tabs");
checks.task_tab_count = tabs.children.length === response.mode_results[1].tasks.length;
checks.roving_tabindex = tabs.children[0].getAttribute("tabindex") === "0" &&
  tabs.children[1].getAttribute("tabindex") === "-1";
checks.aria_controls = tabs.children[0].getAttribute("aria-controls") === "task-panel";
tabs.children[1].listeners.click[0]();
checks.click_sets_task = hooks.state.activeTaskOrdinal === 2;
checks.focus_retained_after_click = documentStub.activeElement === tabs.children[1];
tabs.listeners.keydown[0]({ key: "ArrowRight", target: tabs.children[1], preventDefault() {} });
checks.arrow_sets_task = hooks.state.activeTaskOrdinal === 3;
checks.focus_moved_after_arrow = documentStub.activeElement === tabs.children[2];
tabs.listeners.keydown[0]({ key: "Home", target: tabs.children[2], preventDefault() {} });
checks.home_sets_first = hooks.state.activeTaskOrdinal === 1;
checks.focus_home = documentStub.activeElement === tabs.children[0];
const memorySubtab = findById(byId("task-panel"), "subtab-memory");
memorySubtab.listeners.click[0]();
checks.subtab_switched = hooks.state.activeTaskTab === "memory";
checks.focus_retained_after_subtab = documentStub.activeElement === findById(byId("task-panel"), "subtab-memory");

process.stdout.write(JSON.stringify({ checks: checks }));
'''


def _run_harness(demo_server, tmp_path) -> dict:
    if NODE is None:
        pytest.skip("node is not available; real-browser re-verification is done by the main session")
    meta = _meta(demo_server)
    response = _crafted_response(meta)
    payload = {
        "app_path": str(WEB_DIR / "app.js"),
        "meta": meta,
        "response": response,
    }
    harness = tmp_path / "p3_js_harness.js"
    harness.write_text(JS_HARNESS, encoding="utf-8")
    completed = subprocess.run(
        [NODE, str(harness)],
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        capture_output=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[-2000:]
    return json.loads(completed.stdout.decode("utf-8"))


def test_node_default_selection_mode_change_and_focus(demo_server, tmp_path):
    result = _run_harness(demo_server, tmp_path)
    checks = result["checks"]
    failed = sorted(key for key, value in checks.items() if value is not True)
    assert failed == [], "failed front-end checks: " + ", ".join(failed)


def _css_block(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match, f"missing CSS selector: {selector}"
    return match.group(1)


def test_mobile_css_layout_contract_no_overflow_mask():
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

    assert 'rel="icon" href="/favicon.svg"' in html
    assert "width=device-width" in html
