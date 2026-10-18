# LLM User Simulation Agent

[English](README.md) | [简体中文](README.zh-CN.md)

A research prototype that simulates how a particular user might rate and review a product. It looks only at that user's earlier history, then produces a star rating and a short review. The result is passed to a separate evaluation project instead of being judged by the same code that generated it.

This is not a bot for posting reviews, and it does not claim to know what the real user actually thinks.

## 0. Credits and acknowledgements

- **[AgentSociety](https://github.com/tsinghua-fib-lab/AgentSociety)** (FIB Lab, Tsinghua University). The wider research platform for LLM-based social simulation. This repository does not vendor AgentSociety itself.
- **[AgentSocietyChallenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)** (FIB Lab, Tsinghua University, MIT License). The direct simulation dependency: the project installs its PyPI package, `websocietysimulator==1.0.0a30`, which provides the simulation environment, agent base classes, task interaction tools, and framework evaluation interfaces.
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

**Working today**

- The original AgentSociety-based course pipeline is kept: profiling, reference ranking, reflection/repair, and the course evaluation entry point.
- A local Amazon Musical Instruments dataset has been audited: 10,261 reviews, 400 target tasks, and four fixed statistical baselines. The data stays local and is not redistributed.
- A newer reliable workflow has offline coverage for bounded evidence, four result states, at most one repair, call budgets, and success-only memory writes.
- A versioned `simulation_run.v1` result contract and a local synthetic export entry point were independently checked against the evaluation project. The application code is **not yet published on the public `main` branch**.
- A synthetic browser demo was built and tested locally with fake data. It is not a live service and is not part of the current public deployment.

**Not finished yet**

- No new real-LLM benchmark has been run, so there is no current accuracy, style, cost, or latency score.
- Human review of style and rating-review consistency has not been performed.
- Memory is in-run only; it does not survive a restart and has not shown a measured benefit.
- The public repository and the local development tree are not the same snapshot. A command that exists locally may not exist in a fresh clone.

The short version: **the engineering skeleton and offline checks are real; the evidence that the simulation is accurate or personalised is still missing.**

## 4. How to use it

There is **no plugin and no hosted backend**. This is a Python project that runs locally (or in a notebook/Colab session). It needs an LLM client for real generation, but the test and contract paths use fakes and need no API key.

### Install and test

Requires Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The upstream framework is installed from PyPI; it is not vendored in this repository. Keep API keys in a local `.env` or environment variable, never in the repository.

### Run the local result-contract preview

These commands belong to the local development tree, not the current public `main`:

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

The public repository has no live user-simulation demo. A local synthetic demo was built during development, but its files are not all present in the published `main` snapshot. Do not describe it as a deployed product until the missing files are restored and the demo is republished.

If you run any local demo server, bind it to `127.0.0.1` and keep it off the public internet. The demo uses synthetic content and fake model outputs; never point it at private reviews or target labels.

### What not to do

- Do not use the agent to post reviews as a real person.
- Do not send raw reviews, target labels, prompts, or run artifacts to a third-party model without permission and a defined budget.
- Do not treat a successful fake run or a passing contract test as evidence of real model quality.

## 5. Where to read more

- [Dataset choices and licensing notes](docs/DATASETS.md)
- [Capabilities, metrics, hallucination checks, and memory](docs/AGENT_CAPABILITIES.md)
- [Step-by-step development plan and acceptance criteria](docs/DEVELOPMENT_PLAN.md)
- [P0 data protocol](docs/P0_IMPLEMENTATION_GUIDE.md) / [P1 reliability workflow](docs/P1_IMPLEMENTATION_GUIDE.md) / [synthetic demo](docs/P3_IMPLEMENTATION_GUIDE.md)
- [Project B: evaluation and diagnostics](https://github.com/simonshi-stu/amazon-instruments-evaluation-agent)

If you remember one thing: **this project is a controlled user-simulation prototype, not a proven personalised recommendation system.**
