# LLM User Simulation Agent

[简体中文](README.md) | [English](README.en.md)

A research prototype that simulates how a particular user might rate and review a product. It looks only at that user's earlier history, then produces a star rating and a short review. The result is passed to a separate evaluation project instead of being judged by the same code that generated it.

This is not a bot for posting reviews, and it does not claim to know what the real user actually thinks.

## 0. Credits and acknowledgements

- **[AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety)** (FIB Lab, Tsinghua University). This project explicitly acknowledges and follows its LLM social-simulation research context; AgentSociety itself is not vendored here.
- **[AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)** (FIB Lab, Tsinghua University, MIT License). The direct framework called by the earlier Agent A course pipeline: the project installs `websocietysimulator==1.0.0a30` from PyPI for the simulation environment, agent base classes, task interaction tools, and framework evaluation interfaces.
- **Amazon Product Data (2014), Musical Instruments 5-core reviews**, maintained by Julian McAuley's group at UC San Diego. The dataset page is [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html). Raw reviews, target labels, and private run files are not distributed in this repository. Public availability does not settle redistribution or commercial-use rights.
- **[Project B: Amazon review evaluation and diagnostics](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)**. The independent evaluation side of this work.

The people and organisations above provided framework code, data, or hosting. They did not write this project's design and do not endorse its conclusions.

The parts added on top of the upstream framework include user profiling, reference-review selection, conditional reflection/repair, in-run memory, and later work on time safety, evidence checks, budgets, and failure handling. Upstream code and examples remain the work of their original authors.

## 1. What this project is for, and why it is needed

A simulated review can sound natural and still be wrong. It may miss the user's rating by three stars, copy a reference review, invent a purchase experience, or fail halfway through the run. A single example, or a single accuracy number, hides those differences.

This project exists to make user simulation testable:

- **Personalised prediction.** Given a user's safe history and a target item, predict the rating and review that user might produce.
- **Cold-start and history studies.** Compare users with long, short, or no usable history instead of assuming everyone behaves the same.
- **Offline evaluation.** Generate simulated feedback for experiments without asking real people to write new reviews.
- **A clean hand-off to evaluation.** Export a versioned result that Project B can check and score without reading the generation code.

The output is always marked as simulated. It is not a real customer's opinion and must not be published as one.

## 2. How it is used

A typical run looks like this:

```text
safe user history + target item
        ↓
profile statistics and reference-review selection
        ↓
prompt with bounded, actually rendered evidence
        ↓
LLM draft -> validation -> at most one repair if needed
        ↓
success / degraded / refused / failed
        ↓
rating, review, and run diagnostics
        ↓
Project B: independent scoring and diagnosis
```

The important part is what the model does **not** get: future reviews, the target rating, or another user's history. Memory, when enabled, only recalls earlier generated results for the same user and is treated as a style reference, never as a trusted rating label.

Concrete questions this setup can support:

- Would this user rate a new item higher or lower than their usual level?
- Does a new prompt or retrieval setting improve results on the same tasks?
- Does memory help after it actually enters the prompt, or does it only add cost?
- Can the agent generalise to users it has not seen before?

## 3. Current progress

**Included in this public snapshot**

- The AgentSocietyChallenge pipeline is kept: profiling, reference ranking, reflection/repair, and the course evaluation entry point.
- Agent A's reliable generation-side code is published: `improved_agent_with_quality.py`, `temporal_context.py`, `comprehensive_evaluation.py`, `local_memory.py`, and privacy-audit tooling.
- The Amazon Musical Instruments adapter and time-safe runner are published: `amazon_p0.py`, `amazon_full_evaluation.py`, and `rating_baselines.py`. They support legacy/new fields and second, millisecond, microsecond, and nanosecond timestamps, with explicit per-row status categories.
- A source-only dry-run was verified against the raw source in Drive/Colab: `3,017,439` rows parsed, `3,017,381` valid, and `58` explicitly rejected as `invalid_timestamp`; valid dates were `2001-09-17` through `2023-09-12`. Raw data was not copied into the repository.
- Offline validation covers transport-failure isolation, parsing states, temporal safety, the Memory gate, rating baselines, and privacy boundaries; these checks are not evidence of real-model quality.
- The offline suite passed `186` tests on the 2026-10-28 public snapshot; no real model was called.

**Not finished yet**

- The current GPT-5.6 Luna full LLM evaluation has not been completed, so there is no new accuracy, style, cost, or latency score.
- The earlier partial DeepSeek run was stopped and is not a complete benchmark result; future full evaluation will not use DeepSeek.
- Human review of style and rating-review consistency has not been performed.
- Memory is in-run only; it does not survive a restart and has not shown a measured benefit.
- The public repository publishes the generation side; the independent evaluation side remains in [Project B](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent).

The short version: **source parsing and offline engineering checks have reproducible results; current-model full-quality evidence is still missing.**

## 4. Current development path and A/B consolidation scope

The development workspace was validated as one context containing:

- **A, generation:** user profiling, reference reviews, conditional reflection/repair, in-run Memory, temporal-safe context, the Amazon source adapter, and four deterministic rating baselines.
- **B, evaluation:** an independent result contract, rating/coverage/failure diagnostics, evidence-constrained checks, and export acceptance. B's code and Git history remain in its separate repository.
- **Public boundary:** only versioned predictions and diagnostics cross the boundary; target labels, prompts, raw reviews, and per-record private outputs do not.

This publication is a **reviewed source-snapshot consolidation, not a merge of the two repositories' Git histories**. This repository publishes the A side and public documentation; Project B does not become part of this repository because the development workspace combined the acceptance context.

The next path is:

1. Done: field/time normalization, source-only full-source parsing verification, offline safety tests, and public documentation.
2. Tomorrow: obtain a private batch path for the current GPT-5.6 Luna model, fix the max/Absolute configuration, budget, data-transfer permission, and stop conditions; do not call the real model before that path exists.
3. Run the batch evaluation with raw data, labels, prompts, model outputs, and per-record results kept only in Drive/Colab; publish only de-identified aggregate status.
4. Send the result to Project B for independent scoring and diagnosis, then run same-task Memory pairing; do not report a benefit without denominators and coverage.

## 5. How to use it

There is **no plugin and no hosted backend**. This is a Python project that runs locally (or in a notebook/Colab session). It needs an LLM client for real generation, but the test and contract paths use fakes and need no API key.

### Install and test

Requires Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The upstream framework is installed from PyPI; it is not vendored in this repository. Keep API keys in a local `.env` or environment variable, never in the repository.

### Run the published source-only preview

The raw source must remain in a controlled Drive/Colab staging directory. This command does not call an LLM or copy the source into the repository:

```powershell
python amazon_full_evaluation.py --raw-root "<Drive/Colab raw-root>" --dry-run
```

There is no public full-model command yet; it awaits the private current-model batch interface and approved configuration.

### Run the local result-contract preview

These commands belong to the wider local development workspace, not the current public `main`:

```powershell
python -m user_simulator_project --dry-run
python -m user_simulator_project --synthetic --max-tasks 3 --output-dir results/stage1_demo
```

The output directory must be new and inside the repository's ignored `results/` or `Dataset/` tree. The exporter writes a temporary part file, validates it, and publishes it without overwriting an existing artifact. When the local tree is available, validate the result in Project B:

```powershell
python -m amazon_eval_agent validate --input "<absolute-path>/simulation_run.v1.json" --json
```

Exit code `0` only means the artifact matches the contract. It does not mean the predictions are good.

### Run a demo

The public repository has no live user-simulation demo. The development workspace contains a local synthetic demo, but it is not a live service or part of the current public deployment.

If you run any local demo server, bind it to `127.0.0.1` and keep it off the public internet. The demo uses synthetic content and fake model outputs; never point it at private reviews or target labels.

### What not to do

- Do not use the agent to post reviews as a real person.
- Do not send raw reviews, target labels, prompts, or run artifacts to a third-party model without permission and a defined budget.
- Do not treat a successful fake run or a passing contract test as evidence of real model quality.

## 6. Where to read more

- [Dataset choices and licensing notes](docs/DATASETS.md)
- [Capabilities, metrics, hallucination checks, and memory](docs/AGENT_CAPABILITIES.md)
- [Step-by-step development plan and acceptance criteria](docs/DEVELOPMENT_PLAN.md)
- [Public status report for 2026-10-28](docs/STATUS_2026-10-28.md)
- [P0 data protocol](docs/P0_IMPLEMENTATION_GUIDE.md) / [P1 reliability workflow](docs/P1_IMPLEMENTATION_GUIDE.md) / [synthetic demo](docs/P3_IMPLEMENTATION_GUIDE.md)
- [Project B: evaluation and diagnostics](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)

If you remember one thing: **this project is a controlled user-simulation prototype, not a proven personalised recommendation system.**
