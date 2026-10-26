# Architecture and boundaries

This document describes the code that is actually present in this repository. It is
not a claim that a real model has been evaluated. For release status and measured
results, see the main README.

## 1. Two code paths

The repository keeps the original AgentSociety course path and adds a separate,
framework-independent reliability path. They share data contracts and utility
modules, but they are not the same experiment.

| Path | Main modules | What it does | Current evidence |
| --- | --- | --- | --- |
| Legacy AgentSociety course path | `improved_agent_with_quality.py`, `comprehensive_evaluation.py`, `local_memory.py`, `temporal_context.py`, `memory_audit.py`, `score_calibration.py` | profiling, reference ranking, conditional reflection/repair, structured output, course evaluation, local in-run memory, audits and calibration | offline tests; the real Simulator temporal path is deliberately fail-closed |
| P0 data and rating baselines | `amazon_p0.py`, `rating_baselines.py` | strict Amazon 2014 adapter, frozen-history task selection, target isolation, four deterministic baselines, user-cluster bootstrap, redacted reports | offline tests and a local real-file baseline run; no LLM |
| P1 reliable workflow | `reliable_agent.py`, `p1_smoke.py` | bounded evidence tool, rendered-citation registry, strict validation, at most one repair, four states, pre-call budgets, success-only memory writes | offline tests with injected fake callables only |
| Stage-1 export | `user_simulator_project/` | run the P1 workflow with a deterministic fake callable and write a versioned `simulation_run.v1` artifact | synthetic and local prepared-context fake runs; external validator checks shape and declared metadata only |
| Local synthetic demo | `demo_app.py`, `demo_fixtures.py`, `web_demo/` | loopback-only browser demo with nine fixed scenarios and four modes | offline server/frontend tests and a local `--check`; all content is synthetic |

Project B (`amazon-instruments-evaluation-agent`) is a separate repository. This
repository does not import it at runtime. The cross-repository test invokes its
CLI as a subprocess and skips explicitly when the sibling checkout is absent.

## 2. Responsibility map

| Area | Responsibility | Not responsible for |
| --- | --- | --- |
| `amazon_p0.py` | source adapter, canonical fields, frozen cut, task selection, artifact gates, prior re-computation, baseline runner | model generation, human review |
| `rating_baselines.py` | global/user/item/combined deterministic predictions and scoring, user-cluster bootstrap | LLM quality, causal inference |
| `temporal_context.py` | strict manifest/provider reconstruction and target/future/copy suppression | proving real Simulator ordering |
| `reliable_agent.py` | one tool-augmented agent workflow, evidence scope, validation, repair, budgets, four states, in-run memory gate | autonomous multi-agent behavior, real LLM measurement |
| `p1_smoke.py` | offline synthetic/prepared smoke CLI with fake callables | accuracy or production readiness |
| `user_simulator_project/` | deterministic fake P1 run plus versioned artifact export | real model evaluation or scoring |
| `demo_app.py` / `web_demo/` | local portfolio demo of states, evidence, budgets and memory funnel | hosted service, real reviews, real model calls |
| `comprehensive_evaluation.py` | legacy course runner and CLI; `--dry-run` works without the Simulator framework | time-safe execution through the real Simulator |
| `memory_audit.py` | redacted repetition/leakage audits and temporal manifests | content-level anonymization or semantic fact-checking |

## 3. Data and label boundary

- Raw reviews, prepared tasks, groundtruth and per-task outputs are **not** tracked
  or distributed in this repository. Public tests use synthetic fixtures only.
- The Amazon Musical Instruments 5-core data is a fixed 2014 research file. It is
  not presented as a current 2026 dataset. Its licence remains
  `not_explicitly_verified`; local research use does not imply redistribution rights.
- A task file contains target identity and date, not the target rating or text.
  Target labels/text are isolated in the groundtruth file. They are not used as
  prediction features or in the global prior/trusted statistics. The strict
  provider may parse groundtruth during validation only, to verify identity and
  target matching and to exclude target copies; that validation parse is never a
  prediction feature. The final scorer reads the labels after predictions.
- The strict provider withholds every selected target row, future row and known
  target-text copy from every context. A target is never visible to an earlier
  split, and same-day rows are not treated as causally ordered.
- Generated memory is a style reference only. It is not a rating label, not real
  user feedback and not persisted across processes.
- This release adds no real review text, target labels, review-user IDs,
  per-task results, prompts, API keys or local absolute paths. The pre-existing
  course notebook is retained unchanged as a public historical artifact; it may
  contain old Colab execution metadata from the original course run.

## 4. Result contract

`user_simulator_project/simulation_run_v1.py` mirrors the shared
`simulation_run.v1` schema. The schema hash is pinned and equal on both sides.
An artifact contains:

- source/producer declarations and fingerprint fields;
- one record per task with split, status, reason, prediction origin, rating,
  review, evidence references and bounded diagnostics;
- a four-state result model: `success`, `degraded`, `refused`, `failed`;
- `success` is the only state that can write generated memory;
- diagnostic invariants prove internal record consistency, not external truth.

The current `model_source` implementation enum is
`offline_injected_fake_callable`. `source.kind` can declare
`fake_on_synthetic`, `fake_on_real_context` or `real_model_declared`, but a
declaration is not evidence that a real model ran. The external validator
reports `real_measurement_status=not_verified_or_not_run`,
`scoring_available=false` and `accuracy_claim=none`.

## 5. What is verified and what is not

Verified offline:

- time-safe context reconstruction and target/copy suppression;
- deterministic four baselines and user-cluster bootstrap;
- strict artifact gates, four states, evidence-ref validation, repair limits,
  budgets and success-only memory writes;
- deterministic fake export and external shape validation;
- loopback Demo server/frontend contracts.

Not verified:

- real-LLM rating or review quality, latency, token usage or cost;
- Memory benefit, persistent memory or cross-process recall;
- semantic/factual support for generated claims;
- real Simulator integration and ordering;
- production security, human review or population-level generalization.

## 6. Repository map

```text
amazon_p0.py                 P0 adapter, protocol, baseline runner
rating_baselines.py          deterministic rating baselines + bootstrap
temporal_context.py          strict temporal provider
reliable_agent.py            P1 reliable single-agent workflow
p1_smoke.py                  offline P1 smoke CLI
user_simulator_project/      stage-1 fake run + SimulationRunV1 export
demo_app.py                  loopback synthetic demo server
demo_fixtures.py             hand-written demo fixtures and scenarios
web_demo/                    static frontend for the demo
improved_agent_with_quality.py  legacy AgentSociety agent
comprehensive_evaluation.py  legacy course runner/evaluation CLI
memory_audit.py              redacted data/leakage audits
score_calibration.py         analytical bias/linear score calibration
tests/                       offline tests; synthetic fixtures only
docs/                        architecture, data, P0/P1/P3 and audit notes
```

## 7. CI and offline behaviour

The GitHub Actions workflow installs `requirements-dev.txt`, runs lint, the full
offline test suite, a no-network evaluation dry-run, the Demo contract check and a
3-task synthetic export. It does not download datasets, call a model API or use a
real review fixture. When the sibling Project B checkout is absent, the
cross-repository contract test reports a skip explicitly; CI does not fabricate a
B mirror.
