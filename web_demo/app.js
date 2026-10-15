/* P3 local portfolio demo front-end.
 *
 * Safety/UX rules enforced here:
 * - No external requests: only same-origin /api/meta and /api/run.
 * - DOM is built exclusively with createElement + textContent; raw HTML
 *   string injection is never used, so any server-provided string (including
 *   synthetic review text) is rendered as inert text.
 * - The sanitized report is generated in the browser from already-received
 *   data; the server never writes files.
 * - Run-level ("run"/mode-level) stats do not depend on the selected task;
 *   they re-render when the detail mode changes, never on task switching.
 */
"use strict";

const state = {
  meta: null,
  csrfToken: null,
  lastResponse: null,
  detailModeId: null,
  activeTaskOrdinal: 1,
  activeTaskTab: "evidence",
  focusAfterRender: null,
  statusLabels: {},
};

const KIND_LABELS_ZH = {
  user_history: "真实历史（provider）",
  item_reference: "商品参考（provider）",
};

const WRITE_LABELS_ZH = {
  written: "已写入（成功且仅成功）",
  skipped: "跳过（非成功结果）",
  rejected: "被记忆闸门拒绝",
  not_attempted: "未启用（该模式无生成记忆）",
};

const TAB_IDS = ["evidence", "memory", "budget"];
const TAB_LABELS = { evidence: "证据（实际渲染）", memory: "记忆漏斗", budget: "预算与协议" };

/* ---------- tiny DOM helpers (textContent only) ---------- */

function el(tag, opts, children) {
  const node = document.createElement(tag);
  const options = opts || {};
  if (options.className) node.className = options.className;
  if (options.text !== undefined && options.text !== null) node.textContent = String(options.text);
  if (options.attrs) {
    Object.keys(options.attrs).forEach(function (key) {
      const value = options.attrs[key];
      if (value !== null && value !== undefined) node.setAttribute(key, String(value));
    });
  }
  if (options.on) {
    Object.keys(options.on).forEach(function (key) {
      node.addEventListener(key, options.on[key]);
    });
  }
  (children || []).forEach(function (child) {
    if (child) node.appendChild(child);
  });
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function byId(id) {
  return document.getElementById(id);
}

function text(value, fallback) {
  if (typeof value === "string" && value.length > 0) return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return typeof fallback === "string" ? fallback : "—";
}

function fmtNum(value) {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "—";
}

function fmtStars(value) {
  return typeof value === "number" && Number.isFinite(value) ? value + " 星" : "—";
}

function kvBox(term, value) {
  const dd = el("dd", { text: value });
  return el("div", {}, [el("dt", { text: term }), dd]);
}

function statusBadge(status, label) {
  const key = ["success", "degraded", "refused", "failed"].includes(status) ? status : "neutral";
  return el("span", { className: "badge badge-" + key, text: text(label, status) });
}

function callout(kind, message) {
  return el("p", { className: "callout callout-" + kind, text: message });
}

/* ---------- pure selection helpers (exposed for the Node contract test) ---------- */

function orderModesForDisplay(modes, allModesId) {
  const list = Array.isArray(modes) ? modes.slice() : [];
  if (typeof allModesId !== "string" || allModesId.length === 0) return list;
  return list.sort(function (a, b) {
    if (a.id === allModesId) return -1;
    if (b.id === allModesId) return 1;
    return 0;
  });
}

function defaultModeId(modes, allModesId) {
  const list = Array.isArray(modes) ? modes : [];
  const hasAll = typeof allModesId === "string" && allModesId.length > 0 &&
    list.some(function (mode) { return mode && mode.id === allModesId; });
  if (hasAll) return allModesId;
  for (let index = 0; index < list.length; index += 1) {
    if (list[index] && typeof list[index].id === "string" && list[index].id.length > 0) {
      return list[index].id;
    }
  }
  return "";
}

function resolveInitialSelection(modes, scenarios, allModesId) {
  const scenarioList = Array.isArray(scenarios) ? scenarios : [];
  const modeId = defaultModeId(modes, allModesId);
  let scenarioId = "";
  for (let index = 0; index < scenarioList.length; index += 1) {
    if (scenarioList[index] && typeof scenarioList[index].id === "string" && scenarioList[index].id.length > 0) {
      scenarioId = scenarioList[index].id;
      break;
    }
  }
  return { modeId: modeId, scenarioId: scenarioId, runnable: modeId !== "" && scenarioId !== "" };
}

function bindTablistKeyboard(container, getButtons, onActivateIndex) {
  // Bind once per container and refresh the getter/activate in place, so
  // re-rendering the tab buttons never stacks stale listeners.
  container.__p3Tablist = { getButtons: getButtons, activate: onActivateIndex };
  if (container.__p3TablistBound) return;
  container.__p3TablistBound = true;
  container.addEventListener("keydown", function (event) {
    const config = container.__p3Tablist;
    if (!config) return;
    const key = event.key;
    if (key !== "ArrowLeft" && key !== "ArrowRight" && key !== "Home" && key !== "End") return;
    const buttons = config.getButtons();
    if (buttons.length === 0) return;
    const index = buttons.indexOf(event.target);
    if (index < 0) return;
    let next = index;
    if (key === "ArrowLeft") next = (index - 1 + buttons.length) % buttons.length;
    else if (key === "ArrowRight") next = (index + 1) % buttons.length;
    else if (key === "Home") next = 0;
    else if (key === "End") next = buttons.length - 1;
    if (typeof event.preventDefault === "function") event.preventDefault();
    config.activate(next);
  });
}

/* ---------- meta / init ---------- */

async function fetchJSON(path, options) {
  const response = await fetch(path, Object.assign({ cache: "no-store" }, options || {}));
  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    data = null;
  }
  if (!response.ok) {
    const code = data && typeof data.error === "string" ? data.error : "http_" + response.status;
    throw new Error("安全类别：" + code);
  }
  return data;
}

function setConnection(message) {
  const badge = byId("server-status");
  if (badge) badge.textContent = message;
}

function setStatus(message) {
  const region = byId("status-region");
  if (region) region.textContent = message;
}

function showError(message) {
  const region = byId("error-region");
  if (!region) return;
  region.textContent = message;
  region.hidden = false;
}

function hideError() {
  const region = byId("error-region");
  if (!region) return;
  region.hidden = true;
  region.textContent = "";
}

function renderScenarioDescription() {
  const select = byId("scenario-select");
  const meta = state.meta;
  const description = byId("scenario-desc");
  if (!meta || !description) return;
  const scenario = (meta.scenarios || []).find(function (entry) {
    return entry.id === select.value;
  });
  if (!scenario) {
    description.textContent = "请选择一个场景。";
    return;
  }
  description.textContent =
    scenario.description_zh + "（任务数：" + fmtNum(scenario.task_count) + "；本地合成场景）";
}

function renderMeta(meta) {
  state.statusLabels = meta.status_labels_zh || {};
  const allModesId = typeof meta.all_modes_id === "string" ? meta.all_modes_id : "";
  const scenarios = Array.isArray(meta.scenarios) ? meta.scenarios : [];

  const scenarioSelect = byId("scenario-select");
  clear(scenarioSelect);
  scenarios.forEach(function (scenario) {
    scenarioSelect.appendChild(el("option", { text: scenario.label_zh, attrs: { value: scenario.id } }));
  });
  const scenarioStillValid = scenarios.some(function (scenario) { return scenario.id === scenarioSelect.value; });
  if (!scenarioStillValid && scenarios.length > 0) {
    scenarioSelect.value = scenarios[0].id;
  }
  scenarioSelect.disabled = false;
  if (!scenarioSelect.__p3Bound) {
    scenarioSelect.__p3Bound = true;
    scenarioSelect.addEventListener("change", function () {
      renderScenarioDescription();
    });
  }
  renderScenarioDescription();

  const modeList = byId("mode-list");
  clear(modeList);
  const defaultId = defaultModeId(meta.modes, allModesId);
  orderModesForDisplay(meta.modes, allModesId).forEach(function (mode, index) {
    const input = el("input", {
      attrs: {
        type: "radio",
        name: "demo-mode",
        id: "mode-option-" + index,
        value: mode.id,
        "aria-label": mode.label_zh,
      },
    });
    if (mode.id === defaultId) input.checked = true;
    const label = el("label", { className: "mode-option", attrs: { for: "mode-option-" + index } }, [input]);
    label.appendChild(el("span", { text: mode.label_zh }));
    modeList.appendChild(label);
  });
  byId("mode-fieldset").disabled = false;

  const p2 = meta.p2_panel || {};
  byId("p2-status").textContent = text(p2.status_zh, "评测集成待验收");
  byId("p2-real-llm").textContent = text(p2.real_llm_zh, "真实 LLM 实测未运行");
  byId("p2-note").textContent = text(p2.note_zh, byId("p2-note").textContent);

  const safetyList = byId("safety-list");
  clear(safetyList);
  (meta.safety_zh || []).forEach(function (note) {
    safetyList.appendChild(el("li", { text: note }));
  });
}

function applyMeta(meta) {
  state.meta = meta && typeof meta === "object" ? meta : {};
  renderMeta(state.meta);
  return resolveInitialSelection(state.meta.modes, state.meta.scenarios, state.meta.all_modes_id);
}

function updateRunAvailability() {
  const button = byId("run-button");
  if (!button) return false;
  const scenarioValue = byId("scenario-select") ? byId("scenario-select").value : "";
  const ready = !!(state.meta && state.csrfToken && knownMode(selectedMode()) && knownScenario(scenarioValue));
  button.disabled = !ready;
  return ready;
}

function knownScenario(id) {
  return !!(state.meta && (state.meta.scenarios || []).some(function (entry) { return entry.id === id; }));
}

function knownMode(id) {
  return !!(state.meta && (state.meta.modes || []).some(function (entry) { return entry.id === id; }));
}

function selectedMode() {
  const checked = document.querySelector('input[name="demo-mode"]:checked');
  return checked ? checked.value : "";
}

/* ---------- run ---------- */

async function runDemo() {
  if (!state.meta || !state.csrfToken) {
    showError("演示服务未就绪：请先启动 demo_app.py 并刷新页面。");
    return;
  }
  const scenarioId = byId("scenario-select").value;
  const modeId = selectedMode();
  if (!knownScenario(scenarioId) || !knownMode(modeId)) {
    showError("选择无效：只支持页面列出的固定场景与模式（枚举校验由前端与服务器各做一次）。");
    return;
  }
  const runButton = byId("run-button");
  runButton.disabled = true;
  runButton.textContent = "运行中…";
  const results = byId("results");
  results.setAttribute("aria-busy", "true");
  hideError();
  setStatus("正在运行合成演示（本地 fake callable，无真实模型调用）…");
  try {
    const response = await fetchJSON("/api/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Demo-CSRF": state.csrfToken,
      },
      body: JSON.stringify({ scenario: scenarioId, mode: modeId }),
    });
    state.lastResponse = response;
    state.detailModeId = null;
    state.activeTaskOrdinal = 1;
    state.activeTaskTab = "evidence";
    state.focusAfterRender = null;
    renderResponse(response);
    setStatus("运行完成：场景「" + text(response.scenario && response.scenario.label_zh) + "」。失败/拒绝/降级场景按预期合同展示，不计为成功。");
  } catch (err) {
    showError("运行失败：" + err.message + "。可修正选择后重试。");
    setStatus("运行失败，可重试。");
  } finally {
    updateRunAvailability();
    runButton.textContent = "运行合成演示";
    results.removeAttribute("aria-busy");
  }
}

function modeEntries(response) {
  return Array.isArray(response.mode_results) ? response.mode_results : [];
}

function activeModeEntry() {
  const entries = modeEntries(state.lastResponse || {});
  const found = entries.find(function (entry) {
    return entry.mode && entry.mode.id === state.detailModeId;
  });
  return found || entries[0] || null;
}

function renderResponse(response) {
  byId("empty-state").hidden = true;
  byId("results-body").hidden = false;

  renderRunSummary(response);

  const modeSelect = byId("detail-mode-select");
  clear(modeSelect);
  modeEntries(response).forEach(function (entry) {
    modeSelect.appendChild(el("option", {
      text: text(entry.mode && entry.mode.label_zh, entry.mode && entry.mode.id),
      attrs: { value: entry.mode && entry.mode.id },
    }));
  });
  if (modeEntries(response).length > 0) {
    modeSelect.value = modeEntries(response)[0].mode.id;
    state.detailModeId = modeEntries(response)[0].mode.id;
  }

  renderTimeline();
  renderTaskArea();
  renderCompare(response);
  renderStats();
  renderFooterRunInfo(response);

  byId("download-button").disabled = false;
}

function renderRunSummary(response) {
  const summary = byId("run-summary");
  clear(summary);
  const scenario = response.scenario || {};
  summary.appendChild(kvBox("场景", text(scenario.label_zh, scenario.id)));
  summary.appendChild(kvBox("场景说明", text(scenario.description_zh, "—")));
  summary.appendChild(kvBox("本次运行模式数", fmtNum((response.modes_run || []).length)));
  summary.appendChild(kvBox("合成会话 ID（临时、非真实 ID）", text(response.synthetic_session_id)));
  summary.appendChild(kvBox("simulated", "true（非真实用户/非真实消费）"));
  summary.appendChild(kvBox("上下文指纹（四模式共用）", (text(response.context_scope_sha256)).slice(0, 16) + "…"));
  summary.appendChild(kvBox("模型来源", "offline_injected_fake_callable（本地 fake）"));
  summary.appendChild(kvBox("准确率", "未实测（无真实模型调用）"));
  byId("privacy-note").textContent = text(response.privacy_zh, "仅合成数据。");
}

function renderTimeline() {
  const entry = activeModeEntry();
  const list = byId("timeline");
  clear(list);
  if (!entry) return;
  (entry.tasks || []).forEach(function (task) {
    const items = [];
    items.push(el("span", { className: "day", text: text(task.target_day) }));
    items.push(el("span", { text: text(task.user_label_zh) + " → " + text(task.item_label_zh) }));
    items.push(statusBadge(task.actual && task.actual.status, task.actual && task.actual.status_label_zh));
    if (task.llm && task.llm.repair_attempted) {
      items.push(el("span", { className: "repair-mark", text: "⟳ 发生过一次 repair" }));
    }
    if (task.memory && typeof task.memory.prompt_included === "number" && task.memory.prompt_included > 0) {
      items.push(el("span", { className: "hint", text: "生成记忆进入 prompt：" + task.memory.prompt_included }));
    }
    if (task.expected && task.expected.matches === false) {
      items.push(el("span", { text: "合同不匹配!" }));
    }
    list.appendChild(el("li", {}, items));
  });
}

/* ---------- task detail (tabs keep/restore focus; run-level stats stay run-level) ---------- */

function onDetailModeChange() {
  state.detailModeId = byId("detail-mode-select").value;
  state.activeTaskOrdinal = 1;
  state.focusAfterRender = null;
  renderTimeline();
  renderTaskArea();
  renderStats();
  renderFooterRunInfo(state.lastResponse || {});
}

function activateTask(ordinal) {
  state.activeTaskOrdinal = ordinal;
  state.focusAfterRender = { kind: "task", ordinal: ordinal };
  renderTaskArea();
}

function activateSubtab(tabId) {
  state.activeTaskTab = tabId;
  state.focusAfterRender = { kind: "subtab", id: tabId };
  renderTaskArea();
}

function renderTaskArea() {
  const entry = activeModeEntry();
  const tabs = byId("task-tabs");
  clear(tabs);
  const panel = byId("task-panel");
  clear(panel);
  if (!entry || !Array.isArray(entry.tasks) || entry.tasks.length === 0) {
    panel.appendChild(callout("info", "该模式下没有任务记录。"));
    state.focusAfterRender = null;
    return;
  }
  const buttonsByOrdinal = {};
  const buttons = [];
  entry.tasks.forEach(function (task) {
    const selected = task.task_ordinal === state.activeTaskOrdinal;
    const button = el("button", {
      className: "task-tab",
      text: "任务 " + task.task_ordinal + " · " + text(task.target_day),
      attrs: {
        type: "button",
        role: "tab",
        id: "task-tab-" + task.task_ordinal,
        "aria-selected": selected ? "true" : "false",
        "aria-controls": "task-panel",
        tabindex: selected ? "0" : "-1",
        "aria-label": "查看任务 " + task.task_ordinal + " 详情（" + text(task.user_label_zh) + "）",
      },
      on: {
        click: function () {
          activateTask(task.task_ordinal);
        },
      },
    });
    buttons.push(button);
    buttonsByOrdinal[task.task_ordinal] = button;
    tabs.appendChild(button);
  });
  bindTablistKeyboard(tabs, function () { return buttons; }, function (index) {
    const target = entry.tasks[index];
    if (target) activateTask(target.task_ordinal);
  });
  const task = entry.tasks.find(function (item) {
    return item.task_ordinal === state.activeTaskOrdinal;
  }) || entry.tasks[0];
  renderTaskPanel(panel, task, entry);
  if (state.focusAfterRender && state.focusAfterRender.kind === "task") {
    const ordinal = state.focusAfterRender.ordinal;
    const targetButton = buttonsByOrdinal[ordinal];
    state.focusAfterRender = null;
    if (targetButton && typeof targetButton.focus === "function") targetButton.focus();
  }
}

function renderTaskPanel(panel, task, entry) {
  panel.setAttribute("aria-labelledby", "task-tab-" + task.task_ordinal);
  const actual = task.actual || {};
  const expected = task.expected || {};

  if (expected.matches === false) {
    panel.appendChild(callout("danger", "合同不匹配：实际 " + text(actual.status) + "/" + text(actual.reason) +
      "，预期 " + text(expected.status) + "/" + text(expected.reason) + "。这是检查失败，请报告。"));
  } else if (expected.status && expected.status !== "success") {
    panel.appendChild(callout("warn", text(expected.note_zh, "该任务为预期非成功合同。") +
      "（实际：" + text(actual.status_label_zh, actual.status) + "）"));
  } else {
    panel.appendChild(callout("ok", "与预期合同一致：严格校验通过的成功结果。"));
  }

  const usageDelta = task.task_usage_delta || {};
  const cumulative = task.budget_after_task || {};
  const grid = el("div", { className: "task-grid" });
  const addBox = function (title, value) {
    grid.appendChild(el("div", { className: "stat-box" }, [
      el("h4", { text: title }),
      el("div", { className: "value", text: value }),
    ]));
  };
  addBox("实际状态", text(actual.status_label_zh, actual.status));
  addBox("预测评分", fmtStars(actual.stars));
  addBox("预测来源", actual.origin === "llm" ? "llm（fake）" : text(actual.origin));
  addBox("原因代码", text(actual.reason));
  addBox("记忆写入", text(WRITE_LABELS_ZH[task.memory && task.memory.write], task.memory && task.memory.write));
  addBox("LLM 尝试（本任务真实用量）", fmtNum(usageDelta.llm_attempts));
  addBox("LLM 尝试（截至本任务累计）", fmtNum(cumulative.llm_attempts));
  addBox("引用证据行数", fmtNum((actual.evidence_refs || []).length));
  addBox("注册证据行数", fmtNum(task.evidence_counts && task.evidence_counts.registered_rows));
  panel.appendChild(grid);

  if (typeof actual.review === "string" && actual.review.length > 0) {
    panel.appendChild(el("p", { className: "hint", text: "生成评论（本地 fake 模型输出，仅合成演示，不是真实评论）：" }));
    panel.appendChild(el("div", { className: "review-box", text: actual.review }));
  } else {
    panel.appendChild(el("p", { className: "hint", text: "该任务没有生成评论（非成功结果不产生评论内容）。" }));
  }

  const subtabs = el("div", {
    className: "subtabs",
    attrs: { id: "task-subtabs", role: "tablist", "aria-label": "任务详情子页" },
  });
  const subtabButtons = {};
  const subtabList = [];
  TAB_IDS.forEach(function (tabId) {
    const selected = state.activeTaskTab === tabId;
    const button = el("button", {
      className: "subtab",
      text: TAB_LABELS[tabId],
      attrs: {
        type: "button",
        role: "tab",
        id: "subtab-" + tabId,
        "aria-selected": selected ? "true" : "false",
        "aria-controls": "subtab-panel",
        tabindex: selected ? "0" : "-1",
      },
      on: {
        click: function () {
          activateSubtab(tabId);
        },
      },
    });
    subtabButtons[tabId] = button;
    subtabList.push(button);
    subtabs.appendChild(button);
  });
  bindTablistKeyboard(subtabs, function () { return subtabList; }, function (index) {
    if (TAB_IDS[index]) activateSubtab(TAB_IDS[index]);
  });
  panel.appendChild(subtabs);

  const tabPanel = el("div", {
    attrs: {
      id: "subtab-panel",
      role: "tabpanel",
      "aria-labelledby": "subtab-" + state.activeTaskTab,
    },
  });
  if (state.activeTaskTab === "evidence") renderEvidenceTab(tabPanel, task);
  else if (state.activeTaskTab === "memory") renderMemoryTab(tabPanel, task);
  else renderBudgetTab(tabPanel, task, entry);
  panel.appendChild(tabPanel);

  if (state.focusAfterRender && state.focusAfterRender.kind === "subtab") {
    const target = subtabButtons[state.focusAfterRender.id];
    state.focusAfterRender = null;
    if (target && typeof target.focus === "function") target.focus();
  }
}

function renderEvidenceTab(panel, task) {
  const counts = task.evidence_counts || {};
  panel.appendChild(el("p", {
    className: "hint",
    text: "证据漏斗（真实计数）：候选 " + fmtNum(counts.candidate_rows) +
      " → 工具返回 " + fmtNum(counts.tool_returned_rows) +
      " → 注册 " + fmtNum(counts.registered_rows) +
      " → 引用 " + fmtNum(counts.cited_rows) +
      "；过滤空行 " + fmtNum(counts.filtered_empty_rows) +
      "，过滤注入行 " + fmtNum(counts.filtered_injection_rows) +
      "，工具上限跳过 " + fmtNum(counts.skipped_by_tool_cap_rows) + "。",
  }));
  panel.appendChild(el("h4", { text: "实际渲染进 prompt 的证据（从记录的 prompt 提取，非事后编造）" }));
  const rendered = Array.isArray(task.rendered_evidence) ? task.rendered_evidence : [];
  if (rendered.length === 0) {
    panel.appendChild(callout("info", "没有证据行进入 prompt（可能被渲染预算裁剪、过滤，或该任务在调用前被拒绝）。"));
  } else {
    const list = el("ul", { className: "evidence-list" });
    rendered.forEach(function (row) {
      list.appendChild(el("li", {}, [
        el("div", { text: text(row.text) }),
        el("div", {
          className: "meta mono",
          text: text(row.evidence_id) + " · " + text(KIND_LABELS_ZH[row.kind], row.kind) +
            " · " + text(row.event_time) + " · " + fmtStars(row.stars) +
            (row.text_capped_for_ui ? " · UI 截断展示" : ""),
        }),
      ]));
    });
    panel.appendChild(list);
  }

  panel.appendChild(el("h4", { text: "被引用的注册表（cited registry；按实际渲染集合校验）" }));
  const cited = Array.isArray(task.cited_registry) ? task.cited_registry : [];
  if (cited.length === 0) {
    panel.appendChild(callout("info", "该结果没有引用任何证据行（非成功结果或被拒绝时通常为空）。"));
  } else {
    const list = el("ul", { className: "evidence-list" });
    cited.forEach(function (row) {
      list.appendChild(el("li", {}, [
        el("div", { className: "mono", text: text(row.evidence_id) }),
        el("div", {
          className: "meta",
          text: text(KIND_LABELS_ZH[row.kind], row.kind) + " · " + text(row.event_time) +
            " · " + fmtStars(row.stars) + " · sha256 " + text(row.text_sha256_short) +
            " · 原文字符数 " + fmtNum(row.text_chars_total) +
            (row.text_truncated === true ? " · 曾截断" : ""),
        }),
      ]));
    });
    panel.appendChild(list);
  }
}

function renderMemoryTab(panel, task) {
  const memory = task.memory || {};
  const funnel = el("div", { className: "funnel" });
  const steps = [
    ["候选（本用户）", memory.candidates],
    ["时间合格（严格更早）", memory.time_eligible],
    ["被选中", memory.selected],
    ["进入 prompt", memory.prompt_included],
  ];
  steps.forEach(function (step, index) {
    if (index > 0) funnel.appendChild(el("span", { className: "arrow", text: "→" }));
    funnel.appendChild(el("span", { className: "step" }, [
      el("span", { className: "num", text: fmtNum(step[1]) }),
      el("span", { text: step[0] }),
    ]));
  });
  funnel.appendChild(el("span", { className: "arrow", text: "→" }));
  funnel.appendChild(el("span", { className: "step" }, [
    el("span", { className: "num", text: memory.write === "written" ? "1" : "0" }),
    el("span", { text: "写入" }),
  ]));
  panel.appendChild(funnel);
  panel.appendChild(el("p", {
    className: "hint",
    text: "写入状态：" + text(WRITE_LABELS_ZH[memory.write], memory.write) +
      (memory.write_reason ? "（原因：" + text(memory.write_reason) + "）" : "") +
      "。generated memory 仅作表达风格参考，绝不是可信评分统计；此处是召回次数，不是提升率。",
  }));

  const entries = Array.isArray(memory.selected_entries) ? memory.selected_entries : [];
  if (entries.length === 0) {
    panel.appendChild(callout("info", "本任务没有召回任何生成记忆（候选为 0、同日/未来被排除，或该模式未启用生成记忆）。"));
    return;
  }
  const list = el("ul", { className: "evidence-list" });
  entries.forEach(function (entry) {
    list.appendChild(el("li", {}, [
      el("div", { text: text(entry.text) }),
      el("div", {
        className: "meta",
        text: "序列 " + fmtNum(entry.sequence) + " · " + fmtStars(entry.stars) +
          " · " + text(entry.event_time) +
          " · " + (entry.prompt_included ? "已进入 prompt" : "未进入 prompt（被预算/过滤裁剪）"),
      }),
    ]));
  });
  panel.appendChild(list);
}

function renderBudgetTab(panel, task, entry) {
  const delta = task.task_usage_delta || {};
  const cumulative = task.budget_after_task || {};
  const spec = (entry && entry.budget_spec) || {};
  const grid = el("div", { className: "task-grid" });
  const addBox = function (title, rows) {
    const box = el("div", { className: "stat-box" }, [el("h4", { text: title })]);
    const list = el("ul", {});
    rows.forEach(function (row) {
      list.appendChild(el("li", { text: row }));
    });
    box.appendChild(list);
    grid.appendChild(box);
  };
  addBox("本任务真实用量（相邻累计快照差值）", [
    "LLM 尝试：" + fmtNum(delta.llm_attempts),
    "LLM 成功：" + fmtNum(delta.llm_successes),
    "LLM 失败：" + fmtNum(delta.llm_failures),
    "工具读取：" + fmtNum(delta.tool_calls),
    "prompt 字节预算占用（预留）：" + fmtNum(delta.prompt_bytes_charged),
    "输出 token 预算预留（请求上限）：" + fmtNum(delta.output_token_budget_charged),
  ]);
  addBox("截至本任务累计（run 累计快照）", [
    "LLM 尝试：" + fmtNum(cumulative.llm_attempts),
    "LLM 成功：" + fmtNum(cumulative.llm_successes),
    "LLM 失败：" + fmtNum(cumulative.llm_failures),
    "工具读取：" + fmtNum(cumulative.tool_calls),
    "prompt 字节累计：" + fmtNum(cumulative.prompt_bytes_charged),
    "输出 token 预算累计：" + fmtNum(cumulative.output_token_budget_charged),
  ]);
  const llm = task.llm || {};
  addBox("修复阶段（严格）", [
    "草稿已尝试：" + (llm.draft_attempted ? "是" : "否"),
    "请求修复：" + (llm.repair_requested ? "是" : "否"),
    "修复已尝试：" + (llm.repair_attempted ? "是" : "否"),
    "首次校验原因：" + text(llm.validation_reason_first, "通过/未到达"),
    "修复后校验原因：" + text(llm.validation_reason_after_repair, "通过/未到达"),
    llm.repair_reserve_failure ? "修复预留失败：" + text(llm.repair_reserve_failure) : null,
    llm.failure ? "失败阶段：" + text(llm.failure.stage) + "/" + text(llm.failure.category) : null,
  ].filter(Boolean));
  const rendered = task.render_summary || {};
  addBox("渲染与协议", [
    "prompt 字节（请求上限预留）：" + fmtNum(rendered.prompt_bytes),
    "prompt sha256：" + text(rendered.prompt_sha256_short) + "…",
    "渲染历史行：" + fmtNum(rendered.history_rendered),
    "渲染参考行：" + fmtNum(rendered.reference_rendered),
    "统计段渲染：" + (rendered.stats_rendered ? "是" : "否"),
    "统计段按模式启用：" + (rendered.stats_included_by_mode ? "是" : "否"),
    "全局先验渲染：" + (rendered.prior_rendered ? "是" : "否"),
    "注入过滤（记忆）：" + fmtNum(rendered.memory_filtered_injection),
  ]);
  addBox("运行级上限（合同值）", [
    "每任务 LLM 调用 ≤ " + fmtNum(spec.max_llm_calls_per_task),
    "每 run LLM 调用 ≤ " + fmtNum(spec.max_llm_calls_run),
    "每任务工具读取 ≤ " + fmtNum(spec.max_tool_calls_per_task),
    "每 run 工具读取 ≤ " + fmtNum(spec.max_tool_calls_run),
    "每调用 prompt 字节 ≤ " + fmtNum(spec.max_prompt_bytes_per_call),
    "每调用输出 token ≤ " + fmtNum(spec.max_output_tokens_per_call),
  ]);
  panel.appendChild(grid);
  panel.appendChild(el("p", {
    className: "hint",
    text: "“预算占用/预留”是调用前的上限预留值，不是人民币计费，也不是 provider 实际 token 用量；" +
      "实际用量未知时不编造。预算在调用前预留，失败的尝试同样计入。",
  }));
}

/* ---------- comparison / run-level stats / footer / download ---------- */

function renderCompare(response) {
  const comparison = response.comparison || {};
  byId("compare-note").textContent = text(comparison.note_zh, "") + " " + text(comparison.interpretation_zh, "");
  const table = byId("compare-table");
  clear(table);
  const thead = el("thead", {});
  const headRow = el("tr", {});
  ["模式", "生成记忆", "信任统计", "状态计数（成功/降级/拒绝/失败）", "LLM 尝试", "记忆进入 prompt（次数）", "统计段渲染任务数", "合同"].forEach(function (label) {
    headRow.appendChild(el("th", { text: label, attrs: { scope: "col" } }));
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el("tbody", {});
  (comparison.rows || []).forEach(function (row) {
    const counts = row.status_counts || {};
    const tr = el("tr", {});
    tr.appendChild(el("td", { text: text(row.label_zh, row.mode_id) }));
    tr.appendChild(el("td", { text: row.generated_memory === true ? "开" : row.generated_memory === false ? "关" : "—" }));
    tr.appendChild(el("td", { text: row.trusted_stats === true ? "开" : row.trusted_stats === false ? "关" : "—" }));
    tr.appendChild(el("td", {
      text: fmtNum(counts.success) + " / " + fmtNum(counts.degraded) + " / " + fmtNum(counts.refused) + " / " + fmtNum(counts.failed),
    }));
    tr.appendChild(el("td", { text: fmtNum(row.llm_attempts) }));
    tr.appendChild(el("td", { text: fmtNum(row.memory_prompt_included_total) }));
    tr.appendChild(el("td", { text: fmtNum(row.stats_rendered_tasks) }));
    tr.appendChild(el("td", { text: row.contract_passed ? "与预期一致" : "不匹配" }));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function computeStatsData(entry) {
  if (!entry || typeof entry !== "object") return null;
  const budget = entry.budget_total || {};
  const store = entry.memory_store && typeof entry.memory_store === "object" ? entry.memory_store : null;
  const totals = (entry.tasks || []).reduce(function (acc, task) {
    const memory = task.memory || {};
    acc.candidates += typeof memory.candidates === "number" ? memory.candidates : 0;
    acc.eligible += typeof memory.time_eligible === "number" ? memory.time_eligible : 0;
    acc.selected += typeof memory.selected === "number" ? memory.selected : 0;
    acc.included += typeof memory.prompt_included === "number" ? memory.prompt_included : 0;
    return acc;
  }, { candidates: 0, eligible: 0, selected: 0, included: 0 });
  return {
    accuracy: [
      "未实测：本地 fake LLM，无真实模型调用",
      "本项目真实 LLM 实测尚未运行",
    ],
    budget: [
      "LLM 尝试：" + fmtNum(budget.llm_attempts),
      "成功：" + fmtNum(budget.llm_successes) + "，失败：" + fmtNum(budget.llm_failures),
      "工具读取：" + fmtNum(budget.tool_calls),
      "prompt 字节预算占用（累计）：" + fmtNum(budget.prompt_bytes_charged),
      "输出 token 预算预留（累计）：" + fmtNum(budget.output_token_budget_charged),
      "实际 token 用量：unknown（从不编造）",
      "估算费用：unknown（从不编造）",
    ],
    memory: [
      "候选合计：" + totals.candidates,
      "时间合格合计：" + totals.eligible,
      "被选中合计：" + totals.selected,
      "进入 prompt 合计：" + totals.included,
      store
        ? "记忆库写入：" + fmtNum(store.writes) + "（仅最终成功）"
        : "记忆库写入：未启用（该模式无生成记忆）",
    ],
    storeEnabled: !!store,
    fingerprint: [
      "run hash：" + text((entry.run_hash || "").slice(0, 24)) + "…",
      "run_id 摘要：" + text(entry.run_id_sha256_16),
      "合同通过：" + (entry.contract_passed ? "是" : "否"),
      "模型来源：" + text(entry.model_source),
    ],
  };
}

function renderStats() {
  const entry = activeModeEntry();
  const panel = byId("stats-panel");
  clear(panel);
  const data = computeStatsData(entry);
  if (!data) return;
  const grid = el("div", { className: "task-grid" });
  const addBox = function (title, rows) {
    const box = el("div", { className: "stat-box" }, [el("h4", { text: title })]);
    const list = el("ul", {});
    rows.forEach(function (row) { list.appendChild(el("li", { text: row })); });
    box.appendChild(list);
    grid.appendChild(box);
  };
  addBox("准确率（诚实口径）", data.accuracy);
  addBox("预算总计（当前详情模式）", data.budget);
  addBox("合成记忆召回次数（不是提升率）", data.memory);
  addBox("运行指纹", data.fingerprint);
  panel.appendChild(el("p", {
    className: "hint",
    text: "数据范围：当前详情模式（run 级），不随任务切换变化；模式切换会重新渲染本面板。",
  }));
  panel.appendChild(grid);
  panel.appendChild(el("p", {
    className: "hint",
    text: data.storeEnabled
      ? "本面板不展示准确率、RMSE 或“提升率”等未实测指标；所有数字均为合成 fake 运行的计数。"
      : "该模式不启用生成记忆：召回与写入均为 0（未启用），不是“无提升”。所有数字均为合成 fake 运行的计数。",
  }));
}

function renderFooterRunInfo(response) {
  const info = byId("footer-run-info");
  const entries = modeEntries(response);
  if (entries.length === 0) return;
  const entry = activeModeEntry() || entries[0];
  info.textContent =
    "synthetic session：" + text(response.synthetic_session_id) +
    " · run hash：" + text((entry.run_hash || "").slice(0, 24)) + "…" +
    " · simulated = true · 生成时间（客户端）：" + new Date().toISOString();
}

/* ---------- sanitized report (client-side only) ---------- */

function summarizeStore(store) {
  if (!store || typeof store !== "object") return null;
  return {
    entries: store.entries,
    users: store.users,
    write_attempts: store.write_attempts,
    writes: store.writes,
    rejected_by_reason: store.rejected_by_reason,
  };
}

function buildReport(response) {
  return {
    app: "p3-local-portfolio-demo",
    schema_version: response.schema_version,
    simulated: true,
    note_zh: "脱敏聚合 + 合成 shape：不含 GT、prompt 原文、真实数据、密钥或文件路径。",
    synthetic_session_id: response.synthetic_session_id,
    generated_at_client: new Date().toISOString(),
    scenario: response.scenario,
    context_scope_sha256: response.context_scope_sha256,
    p2_panel: response.p2_panel,
    privacy_zh: response.privacy_zh,
    modes: modeEntries(response).map(function (entry) {
      return {
        mode: entry.mode,
        run_hash: entry.run_hash,
        run_id_sha256_16: entry.run_id_sha256_16,
        model_source: entry.model_source,
        contract_passed: entry.contract_passed,
        mode_checks: entry.mode_checks,
        status_counts: entry.status_counts,
        budget_total: entry.budget_total,
        budget_spec: entry.budget_spec,
        memory_store: summarizeStore(entry.memory_store),
        tasks: (entry.tasks || []).map(function (task) {
          const actual = task.actual || {};
          const memory = task.memory || {};
          return {
            task_ordinal: task.task_ordinal,
            user_label_zh: task.user_label_zh,
            item_label_zh: task.item_label_zh,
            target_day: task.target_day,
            actual: {
              status: actual.status,
              reason: actual.reason,
              origin: actual.origin,
              stars: actual.stars,
              evidence_ref_count: (actual.evidence_refs || []).length,
            },
            expected: task.expected,
            memory: {
              candidates: memory.candidates,
              time_eligible: memory.time_eligible,
              selected: memory.selected,
              prompt_included: memory.prompt_included,
              write: memory.write,
              write_reason: memory.write_reason,
            },
            task_usage_delta: task.task_usage_delta,
            budget_after_task: task.budget_after_task,
            llm: task.llm,
            render_summary: {
              prompt_sha256_short: (task.render_summary || {}).prompt_sha256_short,
              prompt_bytes: (task.render_summary || {}).prompt_bytes,
              stats_rendered: (task.render_summary || {}).stats_rendered,
              stats_included_by_mode: (task.render_summary || {}).stats_included_by_mode,
            },
            evidence_counts: task.evidence_counts,
            rendered_evidence_ids: task.rendered_evidence_ids,
          };
        }),
      };
    }),
  };
}

function safeToken(value) {
  return typeof value === "string" && /^[A-Za-z0-9-]{1,64}$/.test(value) ? value : "session";
}

function downloadReport() {
  if (!state.lastResponse) return;
  const report = buildReport(state.lastResponse);
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "p3_demo_report_" + safeToken(report.synthetic_session_id) + ".json";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(function () {
    URL.revokeObjectURL(url);
  }, 1000);
}

/* ---------- boot ---------- */

document.addEventListener("DOMContentLoaded", function () {
  byId("run-button").addEventListener("click", runDemo);
  byId("download-button").addEventListener("click", downloadReport);
  byId("detail-mode-select").addEventListener("change", onDetailModeChange);
  init();
});

async function init() {
  try {
    const meta = await fetchJSON("/api/meta");
    state.csrfToken = typeof meta.csrf_token === "string" ? meta.csrf_token : null;
    const selection = applyMeta(meta);
    hideError();
    if (selection.runnable && state.csrfToken) {
      setConnection("已连接（本地回环 · 离线 fake）");
      updateRunAvailability();
    } else {
      setConnection("连接异常：元数据不完整");
      showError("演示服务元数据缺少可用的场景或模式，运行已禁用；请重启 demo_app.py 后刷新页面。");
      updateRunAvailability();
    }
  } catch (err) {
    setConnection("未连接");
    updateRunAvailability();
    showError("无法连接本地演示服务：" + err.message + "。请在本机启动 python demo_app.py 后刷新页面。");
  }
}

/* Test hooks for the Node DOM-stub contract test (enabled explicitly only). */
if (typeof globalThis !== "undefined" && globalThis.__P3_TEST__) {
  globalThis.__P3_TEST_HOOKS__ = {
    state: state,
    orderModesForDisplay: orderModesForDisplay,
    defaultModeId: defaultModeId,
    resolveInitialSelection: resolveInitialSelection,
    applyMeta: applyMeta,
    updateRunAvailability: updateRunAvailability,
    selectedMode: selectedMode,
    renderStats: renderStats,
    onDetailModeChange: onDetailModeChange,
    renderResponse: renderResponse,
    renderTaskArea: renderTaskArea,
    activateTask: activateTask,
    computeStatsData: computeStatsData,
    byId: byId,
  };
}
