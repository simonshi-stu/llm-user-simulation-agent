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
  -> UserProfileAnalyzer
  -> ReviewQualityAnalyzer
  -> prompt construction
  -> draft generation
  -> optional reflection
  -> rating/review parsing and validation
  -> structured output
```

### Personal Implementation

- `improved_agent_with_quality.py`
  - `EnhancedPlanning`
  - `UserProfileAnalyzer`
  - `ReviewQualityAnalyzer`
  - `ReasoningWithQualityAwareness`
  - `ImprovedSimulationAgent`
  - rating/review parsing and output fallback
- `comprehensive_evaluation.py`
  - experiment configuration
  - simulator execution
  - RMSE, MAE, accuracy, distribution and correlation metrics
  - intermediate result persistence
  - comparison and ablation report generation
- `inspect_agent_output.py`
  - interactive single-task inspection
  - predicted-vs-ground-truth comparison
  - generated review inspection
- `final_project.ipynb`
  - course experiment notebook and execution record
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
py comprehensive_evaluation.py
py inspect_agent_output.py
```

The scripts are preserved from the course experiment and currently assume the
original simulator task layout. A future cleanup should move paths and model
settings into a configuration file or CLI arguments.

## Current Scope and Honest Limitations

The current personal implementation directly contains user profiling, review
quality analysis, optional memory, reflection and output parsing. It does not
contain a standalone score-calibration class, a production retrieval system,
or a statistically rigorous significance test.

The evaluation source includes a simplified relative-difference helper named
`calculate_statistical_significance`; it should not be described as a real
t-test until it is replaced with per-task statistical testing.

See `docs/PROJECT_SCOPE.md` for the ownership inventory, technical strengths
and extension roadmap.

## Attribution

This project was developed for the CS245 Agent Society course project, Track
A. It builds on the MIT-licensed AgentSociety simulation framework. See
`THIRD_PARTY_NOTICES.md` for upstream attribution.

## License

The original work in this repository is released under the MIT License. See
`LICENSE`.
