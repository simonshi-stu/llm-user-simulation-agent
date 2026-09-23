# Synthetic Memory Ablation

## Purpose

This check uses the same user on tasks A, B and D, with a different user on
task C between them. It verifies that experiment-local memory is read only for
the matching `user_id`, and that `No_Memory` neither reads nor writes it.

## Offline Deterministic Check

`tests/test_synthetic_memory.py` runs the sequence with `FakeLLM` and
`FakeInteractionTool`.

- `Full`: memory appears in prompts 2 times, on tasks B and D.
- The repeated user's stored entries are returned newest-first as D, B, A.
- The other user's store contains only its own C output.
- `No_Memory`: memory appears in prompts 0 times and both stores remain empty.

The repository validation completed with 130 passing tests, a clean Ruff
check, and a successful evaluation `--dry-run`.

## Live DeepSeek Check

On 2026-09-22, the same four-task sequence was run in Colab with the real
`DeepSeekLLM` client. The focused run used four calls per configuration and no
reflection pass.

| Configuration | Memory prompt hits | Same-user stored entries | Other-user stored entries | Calls / errors | API latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Full` | 2 / 4 | 3 | 1 | 4 / 0 | 4.17s |
| `No_Memory` | 0 / 4 | 0 | 0 | 4 / 0 | 3.99s |

For `Full`, the memory marker was present for `live-item-b` and
`live-item-d`, but not for `live-item-a` or the interleaved
`live-other-user` task. The generated star value was 4.0 for all four tasks in
this run. The full cell output remains in the Colab notebook referenced from
the project README.

The optional framework `MemoryDILU` backend was unavailable because
`langchain_chroma` was not installed. This did not disable the tested path:
the dependency-free `LocalMemoryStore` handled the read/write behavior.

## Interpretation

The test proves that `LocalMemoryStore` is active, user-scoped, and correctly
disabled by the `No_Memory` ablation. It does not prove a quality improvement:
the synthetic sequence has no ground-truth quality score, and the four live
outputs are too small a sample for a model comparison.

Colab notebook: <https://colab.research.google.com/drive/1ddYUQ69zRpLo2wTVcLgmGiy8UnHAyUAa>
