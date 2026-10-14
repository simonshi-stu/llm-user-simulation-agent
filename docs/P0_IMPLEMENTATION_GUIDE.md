# P0 Implementation Guide: Offline Amazon Rating Baselines

This guide describes the P0 scope that was actually implemented and tested:
building a field-traceable, time-safe, label-isolated offline task set from
the public Amazon 2014 Musical Instruments 5-core review file, scoring four
deterministic rating baselines, and writing a redacted aggregate report.

P0 uses **no LLM, no Simulator framework, no paid API and no evaluation
service**. The only network operation is the explicitly invoked `download`
subcommand; every other command is fully offline. This document contains no
real review text, identifiers, per-task results or other experiment
artifacts; local outputs stay in gitignored directories.

## 1. Source and license boundary

| Item | Value |
| --- | --- |
| Dataset page | `https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html` |
| File URL | `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz` |
| Citation | J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-Based Recommendations on Styles and Substitutes. SIGIR 2015. |
| Local source name | `reviews_Musical_Instruments_5.json.gz` (gitignored `Dataset/` root) |
| License status | `not_explicitly_verified` |

Public download availability does **not** imply an open license. The official
page was not found to state an explicit data license for the original reviews,
and the Kaggle mirror's CC0 / utility-script MIT terms do not cover them. The
adapter records `license_status = "not_explicitly_verified"` and a
research-use-pending note in the local `p0_policy.json`; no legal terms are
accepted automatically. Only this ~2.46 MB public file was downloaded for the
small authorized local research loop; no other dataset or platform was
fetched.

Source facts verified locally by the adapter run (recorded in the local,
ignored policy file):

- compressed size 2,460,495 bytes; decompressed 7,445,886 bytes
- 10,261 parsed rows; 1,429 users; 900 items; at least 5 rows per user/item
- every `unixReviewTime` is at UTC midnight (0 non-midnight rows), so P0
  treats time at **UTC calendar-day granularity** and never treats same-day
  reviews as causally ordered

## 2. Field adapter and privacy

Canonical mapping (`amazon_p0.normalize_amazon_row`):

| Raw field | Canonical field |
| --- | --- |
| `reviewerID` | `user_id` — locally stable pseudonym `u_` + 16 hex chars |
| `asin` | `item_id` — locally stable pseudonym `i_` + 16 hex chars |
| `overall` | `stars` (validated finite, in 1..5; booleans rejected) |
| `reviewText` | `text` |
| `unixReviewTime` | `timestamp` — canonical UTC ISO date `YYYY-MM-DD` |
| derived | `review_id` — unique stable id including `source_row` |
| derived | `source_row` — original source row number for traceability |

Dropped on purpose: `reviewerName`, `summary` (a target text label too),
`helpful`, vote/image/style fields and every unknown dynamic metadata field.
Only the canonical allowlist is written to the normalized file.

Strict input handling: JSON only (no `eval` anywhere), streamed JSONL or JSON
arrays, one bounded record at a time. Invalid key fields become explicit
rejection categories (`missing_user_id`, `invalid_stars`,
`invalid_timestamp`, ...) with counts in the policy audit; a source with no
valid records fails nonzero. Exact duplicates are kept as distinct rows with
unique review ids and counted (`duplicate_groups`, `duplicate_rows`).

Shared input gates with `memory_audit`/`temporal_context`: at most 64 MiB on
disk including decompressed `.gz` content, at most 100,000 parsed rows, and a
1 MiB per-record cap for early rejection of pathological lines.

## 3. Frozen history protocol (time safety)

- Rows are ordered by UTC day and source row. The cut is a **day boundary**
  closest to the configured row ratio (default 0.6); a same-day group is
  never split.
- Days before the cut form the **frozen observed history prefix**; days from
  the cut onward form the **target candidate region**.
- Targets are selected only from users with **at least two distinct UTC
  target dates** in the candidate region, and every selected row of a user is
  on a distinct date (never one task per user, never same-day repeats posing
  as repeats). The shuffled user pool is capped at `max_tasks // 2`, so every
  selected user receives at least two time-distinct tasks; budgets that
  cannot satisfy the contract (fewer than four tasks, or no two users with
  two dates each) fail closed. The final selection is chronological.
  Defaults: `seed=42`, `max_tasks=400`.
- The runtime review source for the existing strict provider is exactly the
  **frozen history prefix + selected target source rows**. After-cut
  non-target rows are excluded, so observed history never rolls forward.
- The existing `TemporalContextProvider` revalidates a canonical manifest and
  withholds **every** selected target row — from every split, including
  earlier targets — plus duplicate/matching target texts, from every context.
  Same-day target peers are never visible to each other.
- The global prior is computed from the **eligible frozen prefix only**: rows
  whose review id is a selected target, or whose normalized text digest is a
  matchable copy of any selected target text, are excluded with the same rule
  the provider uses. The policy records prefix total rows, excluded rows and
  actual prior rows; the runner recomputes all three from the validated
  runtime artifact and rejects any mismatch. An empty eligible prefix fails
  closed (`no_eligible_prior_history`).

Two different partitions must not be confused:

1. **Observed history vs target region**: the frozen prefix/target split.
2. **Manifest development/validation/test split**: 60/20/20 (nearest
   same-day-indivisible boundaries) over the **selected target cohort only**.
   It is not a partition of the raw file; every split sees the same frozen
   history. The local report repeats this scope note.

Task files visible to a future agent contain only `task_index`, `user_id`,
`item_id`, the target date and the target review id — no stars and no text.
Groundtruth files (local, ignored) pair the same identities with `stars` and
the withheld target text so the provider can suppress matchable copies.

## 4. Data flow

```text
download (explicit, exact pinned HTTPS URL) -> raw .json.gz (Dataset/, ignored)
prepare  -> normalized_reviews.jsonl   full normalized source (local, ignored)
         -> runtime_reviews.jsonl      frozen history + selected target rows
         -> tasks/ and groundtruth/    paired identity vs withheld label
         -> manifest.json              canonical strict temporal manifest
         -> p0_policy.json             adapter/protocol/source metadata
run      -> artifact size/row gates before any parsing (fail closed)
         -> policy parameter/protocol validation against fixed constants
         -> selection reconstruction from the normalized source (cut, seed,
            budget, ratios and task ids must match)
         -> eligible-prefix global prior recomputed and verified
         -> TemporalContextProvider revalidation (fail closed on drift)
         -> four baselines predicted from safe contexts; prediction functions
            receive no groundtruth ratings
         -> groundtruth stars read by the scorer only after predictions
         -> point metrics, cold-start counts, strata, user-cluster bootstrap
         -> report.json / report.md (validated numbers + fixed constants only)
```

Groundtruth records are parsed earlier for verification and target/copy
exclusion. Their ratings never feed the prediction functions or the global
prior; the scorer consumes target ratings only after every prediction is
complete. The report carries no policy free text: descriptive
fields are rebuilt from fixed module constants, and numeric/enum fields are
validated against the artifacts.

All output directories are newly created and refuse to overwrite; a failed
run removes its own partial directory, so leftovers cannot be mistaken for
success.

## 5. Baselines and scoring

`rating_baselines.py` is pure and I/O-free. Prediction functions accept only
a `RatingContext` (visible user history stars and other users' visible item
reference stars) plus an already verified global prior; they never read
groundtruth, source files or split labels. Fixed defaults, never tuned on any
split:

```text
user    = prior                                  if no visible user history
          (mean_user * n_user + 5 * prior) / (n_user + 5)   otherwise
item    = prior                                  if no visible item refs
          (mean_item * n_item + 5 * prior) / (n_item + 5)   otherwise
global  = eligible frozen history prefix mean (target ids and target-text
          copies masked out; see section 3)
combined = 0.5 * user + 0.5 * item
```

Predictions are continuous in 1..5 (no artificial half-star quantization).
Cold-start fallbacks to the global prior are counted per mode and split, and
first-seen vs repeat target strata are reported separately. The runner
recomputes the eligible-prefix prior from the validated runtime artifact and
verifies prefix/excluded/eligible row counts plus the prior value against the
policy, or the run fails with `global_prior_source_mismatch` /
`global_prior_mismatch` / `no_eligible_prior_history`.

Test-split uncertainty uses a user-cluster bootstrap (default `n_boot=2000`,
`seed=42`): whole users are resampled **with replacement** (duplicates
allowed, no forced non-degenerate redraw), one draw is shared by every
baseline in an iteration, and paired differences vs `global` use those same
resamples. Empty test splits or a single user cluster report no interval
instead of fabricating one.

## 6. Reproducing locally

```powershell
# 1. Explicit download of only the exact pinned official archive (gitignored)
#    The command accepts no other URL, fully validates gzip integrity and the
#    decompression cap, and never overwrites an existing file.
.\.venv\Scripts\python.exe amazon_p0.py download `
  --url https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz `
  --output Dataset\amazon_musical_instruments_p0\raw\reviews_Musical_Instruments_5.json.gz

# 2. Build frozen-history tasks, groundtruth, manifest and policy (new dir)
.\.venv\Scripts\python.exe amazon_p0.py prepare `
  --source Dataset\amazon_musical_instruments_p0\raw\reviews_Musical_Instruments_5.json.gz `
  --work-dir Dataset\amazon_musical_instruments_p0\prepared_v2 `
  --history-ratio 0.6 --max-tasks 400 --seed 42 `
  --confirm-target-timestamp-semantics

# 3. Revalidate through the strict provider and score offline (new dir)
.\.venv\Scripts\python.exe amazon_p0.py run `
  --prepared-dir Dataset\amazon_musical_instruments_p0\prepared_v2 `
  --output-dir Dataset\amazon_musical_instruments_p0\baseline_v2 `
  --n-boot 2000 --seed 42
```

`--confirm-target-timestamp-semantics` is required: the manifest still
records `target_timestamp_semantics_verified: false`; the caller asserts only
that `unixReviewTime` is the review event time per the dataset documentation.
Codes verify only epoch-integer conversion and day grouping.

Failure categories are printed alone (never paths, ids or text), for example:
`source_parse_failed`, `source_record_limit_exceeded`,
`source_decoded_byte_limit_exceeded`, `download_invalid_gzip`,
`download_decompression_limit_exceeded`, `invalid_download_limit`,
`confirmation_required`, `output_dir_exists`,
`insufficient_repeated_user_candidates`, `no_eligible_prior_history`,
`artifact_limit_exceeded`, `artifact_row_limit_exceeded`,
`policy_parameter_mismatch`, `policy_protocol_mismatch`,
`policy_artifact_mismatch`, `target_not_uniquely_matched`,
`runtime_manifest_rejected`, `global_prior_mismatch`.

## 7. Tests and results

The offline suite adds synthetic-only tests (`tests/test_amazon_p0.py`,
`tests/test_rating_baselines.py`) that prove: field mapping and forbidden
field removal; rejection categories including oversized numeric literals;
malformed JSON and trailing JSON documents; invalid UTF-8; byte-accurate
line/record/row/decoded-byte gates that never mis-reject many small records;
stable duplicate handling; day-boundary cut and same-day indivisibility;
distinct-date repeated-user selection and its safe small-budget rejections;
CLI prepare artifacts consumed by the existing strict provider; all
target/duplicate/future exclusions; the hand-calculated eligible-prefix prior
(17 of 18 synthetic history rows) and the invariance of every baseline
prediction to the excluded copy's star value; changed source/manifest
rejection; tampered policy parameters/protocol fields rejected and report
free-text rebuilt from safe constants (markers cannot leak); artifact gates
running before any JSON parsing or provider construction; hand-calculated
baselines and cold fallbacks; fixed parameters independent of split labels;
label perturbation not changing predictions; deterministic paired
user-cluster bootstrap and its non-fabrication rules; download URL, size,
gzip-integrity and decompression-cap contracts; report privacy, failure
nonzero and asset gitignore. No test downloads data or reaches the network.

Results from the authorized real small-file run (protocol counts, split
sizes, point metrics and interval values) live only in the local gitignored
`Dataset/amazon_musical_instruments_p0/baseline_v2/` report. Aggregate
numbers are deliberately not recorded in tracked files. Any numbers in such a
report are **not** Memory or LLM effects: they measure whether a static
rating policy beats the global mean on this conditional target cohort.

## 8. Limitations and next steps

- Event-time semantics rest on a documented adapter assertion, not code
  verification; day granularity cannot model intra-day ordering at all.
- The target cohort is a conditional repeated-user sample after the cut day,
  not a representative sample of the raw file; the manifest split is not the
  raw file's last 20%.
- The prototype accepts at most 64 MiB decompressed / 100,000 rows and cannot
  consume the full 4+ GB Amazon review corpus.
- License status remains `not_explicitly_verified`; do not redistribute the
  local data.
- P0 covers rating policy only. A later phase could reuse these exact
  contexts for an authorized LLM/text-generation study (with an explicit
  budget gate), add owner-confirmed event-time semantics, and extend to
  other 5-core categories through the same adapter.
