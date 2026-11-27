# Agent runs

The runs quoted in the [web agent analysis](../../docs/agent_analysis.md), copied from `runs/` (which git ignores). The trace format grew between them, so the three folders differ:

| folder | model | contents |
|---|---|---|
| `20261123-161259-holo-license/` | Holo 3.1 9B, llama-server on a laptop CPU | the second local trial: its trace and 4 screenshots. It predates the per-trial folders and the automatic verdict (`success: null`, no `reached_page`). The first local trial ran before the trace existed. |
| `20261126-174604-holo-license-hosted/` | `holo3-1-35b-a3b`, H's API | the trace of the trial the evaluator mis-scored (`success: false` for a right answer), kept as it was judged then. Only the trace was copied here. |
| `20261127-091732-holo-license-hosted/` | `holo3-1-35b-a3b`, H's API | the current layout: settings (`run.json`), exact tasks (`tasks.json`), and one `trial-<n>/` folder per trial with its trace and screenshots. |

The `-hosted` suffix was added by hand when copying, to tell the two setups apart. `holo-agent-report <trace.jsonl>` rebuilds the step table and figures of any of these traces.
