# LLM User Simulation Agent

An LLM-based user simulation agent developed for the CS245 Agent Society
course project, Track A.

This repository is a clean, independent extraction of the personal agent and
evaluation work from the original course workspace. The upstream
`websocietysimulator` framework is used as a dependency and is not vendored
here.

## What It Does

The agent simulates a Yelp-style user who must produce a rating and review for
an item. It combines:

1. Historical user-review statistics for personalization.
2. Review-quality signals such as `useful`, `funny`, and `cool`.
3. Reference reviews for the target item.
4. Optional cross-task memory.
5. A two-stage LLM process: draft generation followed by reflection and
   revision.
6. Structured parsing and validation of the final rating and review.

The required output is:

```json
{
  "stars": 4.0,
  "review": "..."
}
```

## Architecture

```text
Task
  -> EnhancedPlanning
  -> user/item/review retrieval through the simulator tool
  -> hybrid reference-review ranking (engagement + lexical overlap + length)
  -> prompt-injection filtering
  -> UserProfileAnalyzer
  -> ReviewQualityAnalyzer
  -> prompt construction
  -> draft generation
  -> conditional reflection (only when the draft fails checks)
  -> structured JSON parsing with label-format fallback and validation
  -> leakage check against reference reviews
  -> structured output
```

### Personal Implementation

- `improved_agent_with_quality.py`
  - `EnhancedPlanning`
  - `UserProfileAnalyzer`
  - `ReviewQualityAnalyzer`
  - `ReasoningWithQualityAwareness` (conditional reflection)
  - `ImprovedSimulationAgent` (JSON output contract, hybrid retrieval,
    injection/leakage checks, no-context mode, optional local memory)
  - `AgentOutput` (Pydantic validation), `normalize_stars`
  - `DeterministicBaseline` / `DeterministicSimulationAgent`
- `local_memory.py`
  - bounded, in-process, user-scoped memory for within-run ablations
- `comprehensive_evaluation.py`
  - experiment configuration and CLI (`--dry-run`, `--seed`, `--experiment`,
    temperature, JSON mode and calibration options)
  - simulator execution and LLM usage tracking
  - fresh local memory store per memory-enabled experiment configuration
  - RMSE, MAE, accuracy, distribution and correlation metrics
  - per-task records and diagnostics, held-out calibration, paired bootstrap tests
  - comparison and ablation report generation
- `score_calibration.py`
  - bias and linear score calibrators with a validation split
- `inspect_agent_output.py`
  - interactive single-task inspection
  - predicted-vs-ground-truth comparison
  - generated review inspection
- `tests/`
  - 130 offline tests covering the agent, evaluation, calibration and memory paths
- `final_project.ipynb`
  - course experiment notebook and execution record
- `docs/SYNTHETIC_MEMORY_ABLATION.md`
  - deterministic and live repeated-user memory validation
- `ablation_results_*_clean.json`
  - archived Yelp, Amazon and Goodreads experiment snapshots

## Framework Boundary

The following framework is external to this repository:

- PyPI package: `websocietysimulator==1.0.0a30`
- Original project: WWW'25 AgentSociety Challenge
- Original framework authors and license are recorded in
  `THIRD_PARTY_NOTICES.md`

The original framework source, examples, tutorials, static assets, generated
data and course task files are intentionally not copied into this repository.

## Archived Results

The original project README reported the following headline results on 300
tasks per dataset:

| Dataset | RMSE | MAE | Pearson | Overall quality |
| --- | ---: | ---: | ---: | ---: |
| Yelp | 0.968 | 0.745 | 0.691 | 82.8% |
| Amazon | 1.024 | 0.798 | 0.658 | 80.3% |
| Goodreads | 0.892 | 0.681 | 0.723 | 85.0% |

The JSON files in this repository are archived snapshots from a related
evaluation run and should be treated as reproducibility evidence, not as a
claim that every configuration can be reproduced without the original task
assets.

The historical run reported approximately 162 seconds for 300 tasks. The
current copied evaluation harness defaults to 10 tasks and 5 workers for a
small local smoke run.

## Setup

Python 3.10 or newer is recommended.

```bash
py -m venv .venv
\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Set the LLM key in the shell before running an experiment:

```bash
$env:DEEPSEEK_API_KEY = "your-key"
```

Do not commit `.env` files, API keys, datasets or generated results.

## Running Tests

The 130 offline tests cover the agent workflow (with fakes), profile
statistics, quality thresholds, hybrid reference ranking, injection and
leakage checks, conditional reflection, structured output, calibration,
per-task records and paired bootstrap tests. They require neither the
simulator framework nor an API key:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

CI runs the same lint, tests and a `--dry-run` smoke check on every push
(`.github/workflows/ci.yml`).

## Data Layout

The task data is not included in this repository. The simulator harness
expects the original course/competition assets in a layout similar to:

```text
Dataset/
  item.json
  review.json
  user.json
example/
  track1/
    yelp/
      tasks/
      groundtruth/
    amazon/
      tasks/
      groundtruth/
    goodreads/
      tasks/
      groundtruth/
```

After installing the dependency and restoring authorized task assets:

```bash
# Validate the configuration without API calls or the simulator framework
py comprehensive_evaluation.py --dry-run

# Run the ablation suite (defaults to the yelp task set, 10 tasks, 5 workers)
py comprehensive_evaluation.py --task-set yelp --num-tasks 10 --max-workers 5
py comprehensive_evaluation.py --experiment Baseline Full --output-dir results

py inspect_agent_output.py
```

Every path, task count, worker count and model setting is now a CLI argument;
see `py comprehensive_evaluation.py --help`. `--dry-run` neither imports the
simulator framework nor calls the LLM, so it doubles as a no-network smoke
check. `inspect_agent_output.py` is preserved from the course experiment and
still assumes the original simulator task layout and an API key.

## Colab Quickstart

The course datasets are large, so Google Colab with the data mounted from
Google Drive is the recommended way to run full experiments.

```python
# 1. Mount the Drive folder that holds the course assets
from google.colab import drive
drive.mount("/content/drive")

# 2. Inspect the Drive layout and make the clone the working directory
COLAB_ROOT = "/content/drive/MyDrive/society agent"  # adjust to your folder
!ls "$COLAB_ROOT"
!find "$COLAB_ROOT" -maxdepth 3 -type d -name tasks
%cd $COLAB_ROOT/llm-user-simulation-agent

# 3. Install dependencies (the simulator framework is the heavy part)
!python -m pip install -q -r requirements.txt
!python -m pip install -q -r requirements-dev.txt

# 3b. Colab 2026 ships transformers 5.x / openai 2.x / langchain 1.x, which
#     break `import websocietysimulator`. Only these packages are needed by
#     the framework's import chain; chroma-based memory is optional and
#     would otherwise force numpy<2 on Colab:
!python -m pip install -q "lmdb<2.0,>=1.6.2" "langchain-openai<0.3,>=0.2.14" "langchain-core<0.4" "openai<2.0,>=1.58.1" "transformers>=4.47,<5"
!python -c "import websocietysimulator; print('framework OK')"

# 4. Validate offline before spending any API budget
!python -m pytest -q
!python comprehensive_evaluation.py --dry-run

# 5. Load the API key from Colab Secrets, never paste it into the notebook
import os
from google.colab import userdata
os.environ["DEEPSEEK_API_KEY"] = userdata.get("DEEPSEEK_API_KEY")
```

If the assets are not inside the cloned folder, pass absolute Drive paths
instead of relying on the defaults:

```bash
python comprehensive_evaluation.py \
  --data-dir "/content/drive/MyDrive/society agent/Dataset" \
  --task-dir "/content/drive/MyDrive/society agent/example/track1/yelp/tasks" \
  --groundtruth-dir "/content/drive/MyDrive/society agent/example/track1/yelp/groundtruth" \
  --task-set yelp --num-tasks 30 --max-workers 1 --seed 42 \
  --experiment Baseline No_Context Full No_Memory \
  --draft-temperature 0.4 --reflection-temperature 0.2 --json-mode \
  --calibration-method bias \
  --output-dir "/content/drive/MyDrive/society agent/results"
```

Notes:

- Prefer a GPU or high-RAM runtime: importing the framework loads TensorFlow,
  PyTorch and transformers, and the default CPU runtime can disconnect under
  that load. The environment setup itself is verified on the default runtime
  (130 offline tests plus `--dry-run` pass).
- Without `langchain`/`langchain-chroma` (step 3b), the optional framework
  `MemoryDILU` backend is disabled and logs a warning. The evaluation runner's
  `use_memory` configurations still use the dependency-free `LocalMemoryStore`:
  memory is bounded, scoped by `user_id`, and discarded after each experiment.
- `--task-set` accepts `yelp`, `amazon` and `goodreads`; each needs its own
  `tasks/` and `groundtruth/` folders, and a processed data directory with
  `item.json`, `review.json` and `user.json` for that dataset.
- Run 30-50 paired tasks for each of `yelp`, `amazon` and `goodreads` before
  spending the budget on a full run.
- Use `--max-workers 1` when measuring ordered memory effects; parallel tasks
  can race before an earlier same-user output is stored.
- Keep `--max-workers` modest on Colab for general runs to avoid rate limits and
  memory pressure.
- Write `--output-dir` to Drive so results survive runtime resets. The runner
  saves metadata, per-experiment metrics and per-task JSON after every
  experiment, so an interrupted session keeps the results already produced;
  re-running starts a new run (there is no resume flag yet).

## Current Scope and Honest Limitations

Implemented and unit-tested offline:

- user profiling, review quality analysis, within-run local memory, conditional
  reflection and structured output parsing with label-format fallback;
- hybrid reference-review ranking (engagement + lexical overlap + length),
  prompt-injection filtering and generated-review leakage checks;
- a deterministic baseline and a no-context LLM baseline in the ablation suite;
- `score_calibration.py` with bias and linear calibrators;
- per-task error records and a paired bootstrap test replacing the old
  relative-difference helper;
- per-run artifacts (`metadata.json`, per-task JSON, comparisons) and LLM
  usage counters.

Honest limitations that remain:

- the calibration module and paired bootstrap still need live validation on
  the three course task sets; the focused live synthetic run validates memory
  wiring, not model quality;
- reference selection is a transparent lexical hybrid, not embedding-based
  semantic retrieval, and no embedding endpoint is wired in;
- the archived Yelp/Amazon/Goodreads numbers come from the earlier course run
  and were not produced by this cleaned-up code path;
- per-task latency cannot be measured through the simulator API; only
  per-experiment wall time and aggregate LLM call latency are saved.
- local memory is intentionally in-process only: it does not persist across
  experiments or Colab runtime restarts and does not store groundtruth.

See `docs/PROJECT_SCOPE.md` for the ownership inventory, `docs/PROJECT_GUIDE.md`
for the full architecture and `docs/INTERVIEW_GUIDE.md` for talking points.

## Attribution

This project was developed for the CS245 Agent Society course project, Track
A. It builds on the MIT-licensed AgentSociety simulation framework. See
`THIRD_PARTY_NOTICES.md` for upstream attribution.

## License

The original work in this repository is released under the MIT License. See
`LICENSE`.
