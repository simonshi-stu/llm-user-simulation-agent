# Project Scope

## Why This Repository Exists

The original `review-agent` repository contains the upstream AgentSociety
framework history together with the CS245 Track A implementation. This
repository isolates the personal agent, evaluation scripts, notebook and
archived results so future work can continue without mixing the upstream
framework source into the project history.

## Personal Work Inventory

| File | What it contains | Ownership status |
| --- | --- | --- |
| `improved_agent_with_quality.py` | Main agent and four personal components: planning, user profiling, quality analysis and reflection; optional framework memory is wired into the workflow | Personal implementation |
| `comprehensive_evaluation.py` | Experiment runner, metric calculation, result persistence and report helpers | Personal evaluation work |
| `inspect_agent_output.py` | Interactive task-level output inspection and error display | Personal tooling |
| `final_project.ipynb` | Course experiment notebook and execution record | Personal experiment artifact |
| `ablation_results_yelp_clean.json` | Yelp archived metrics | Personal experiment result |
| `ablation_results_amazon_clean.json` | Amazon archived metrics | Personal experiment result |
| `ablation_results_goodreads_clean.json` | Goodreads archived metrics | Personal experiment result |

The simulator framework, course examples, task assets and data-processing
utilities from the original workspace are not claimed as personal code here.

## What Can Be Explained in an Interview

The strongest truthful story is:

> I implemented an LLM user simulator on top of the AgentSociety framework.
> The agent first builds a compact user profile from historical reviews,
> analyzes high-quality reference reviews, prompts the LLM to generate a
> rating and review, and optionally performs a reflection pass. I then built
> an evaluation harness and ablation artifacts across Yelp, Amazon and
> Goodreads rather than judging the agent from a single example.

This emphasizes system decomposition, prompt/context design, evaluation
discipline, failure handling and reproducibility.

## Future Extension Roadmap

### Priority 1: Make the experiment reproducible

- Move dataset paths, task set, number of tasks, workers and model settings
  into a typed configuration or CLI.
- Add a small offline fixture so the agent and parsers can be tested without an
  API call or the full course dataset.
- Add unit tests for profile statistics, quality thresholds, output parsing and
  star-range validation.
- Save per-task outputs, prompts, latency and errors with a run identifier.

### Priority 2: Improve the agent itself

- Replace first-N reference selection with semantic retrieval or a transparent
  hybrid ranking strategy.
- Add an explicit score-calibration module and compare raw versus calibrated
  predictions on a held-out validation split.
- Replace free-form `stars:` and `review:` parsing with structured JSON output
  and schema validation.
- Make the reflection step conditional on uncertainty or rule violations to
  reduce unnecessary LLM calls and cost.
- Add prompt-injection and unsupported-claim checks before returning a review.

### Priority 3: Make evaluation scientifically stronger

- Preserve per-task errors and use paired bootstrap confidence intervals or a
  paired statistical test instead of a scalar relative-difference helper.
- Run every ablation with the same task IDs, seed policy, model settings and
  budget.
- Report cost, latency, failure rate, parse-repair rate and quality metrics
  together with RMSE and MAE.
- Compare against a deterministic baseline and a no-context LLM baseline.

### Priority 4: Turn it into a portfolio-quality service

- Add a small API or Streamlit demo using synthetic fixtures only.
- Add CI for linting, unit tests and a no-network smoke test.
- Separate framework adapters from domain logic so the core agent can be
  tested independently of `websocietysimulator`.
