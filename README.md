# holo-playground

How much screen resolution does a GUI-agent model need to click the right button, and what does it cost on a laptop CPU?

This repo runs [Holo 3.1](https://huggingface.co/Hcompany/Holo-3.1-4B) (4B and 9B), a vision-language model trained to locate UI elements, locally with [llama.cpp](https://github.com/ggml-org/llama.cpp).
It sweeps the number of tokens the image is encoded into (`--image-max-tokens`) and measures click accuracy and latency for each budget.

![Accuracy and latency vs. image token budget](docs/sweep.png)

## Key findings

- **Image tokens drive the latency.** Encoding the image and processing its tokens before the   first output token (the *prefill*) accounts for 80–98 % of each request. The answer itself is   only ~16 tokens. Halving the image budget roughly halves the latency.
- **Accuracy holds down to ~512 tokens, then collapses.** At 512 tokens, the 9B is still within 4 px of the target center (median), on targets as small as 19×16 px. At 128 tokens, the 4B stops being imprecise and starts being wrong: it clicks on unrelated elements, up to 570 px from the target.
- **The 9B degrades more gracefully.** At 128 tokens it misses 3 targets out of 4, but never by more than 53 px from the target center, against 570 px for the 4B. Its larger language model seems to recover the page layout even when the image is too coarse to pinpoint a small icon.
- **Best trade-off here: 9B at 512 tokens.** It hits 4/4 targets in 50 s, faster than the 4B at 1024 tokens (66 s) for a similar or better accuracy. On this machine, a bigger model looking at a smaller image beats a smaller model looking at a bigger one.
- **Generation speed follows the weight size.** The 9B generates tokens 1.8× slower than the 4B, and its weights are 1.8× larger (5.6 vs. 3.1 GB): each new token re-reads every weight from RAM, so generation is memory-bandwidth bound. Prefill, which processes tokens in batches, is only 1.4–1.7× slower. I haven't isolated why yet.

## Results

One 1897×963 screenshot of the Hugging Face home page, 4 targets from a 94×30 px text link down to a 19×16 px heart icon. Greedy decoding (temperature 0). A *hit* is a click inside the target's bounding box.

| `--image-max-tokens` | image tokens | model | hits | median dist. to center | prefill | gen tok/s | latency |
|---|---|---|---|---|---|---|---|
| default (4096) | 1770 | 4B | 4/4 | 3.4 px | 128 s | 6.2 | 131 s |
| default (4096) | 1770 | 9B | 4/4 | 1.6 px | 182 s | 3.8 | 187 s |
| 1024 | 968 | 4B | 4/4 | 4.2 px | 63 s | 6.8 | 66 s |
| 1024 | 968 | 9B | 3/4 | 3.0 px | 90 s | 3.9 | 94 s |
| 512 | 496 | 4B | 3/4 | 5.9 px | 30 s | 7.0 | 32 s |
| 512 | 496 | 9B | 4/4 | 3.5 px | 46 s | 4.0 | 50 s |
| 256 | 242 | 4B | 2/4 | 13.0 px | 15 s | 7.0 | 18 s |
| 256 | 242 | 9B | 2/4 | 19.8 px | 27 s | 3.5 | 31 s |
| 128 | 120 | 4B | 0/4 | 132.6 px | 10 s | 7.2 | 12 s |
| 128 | 120 | 9B | 1/4 | 27.4 px | 17 s | 4.0 | 21 s |

`--image-max-tokens` is a cap: llama.cpp shrinks the image until it fits, then rounds each side down to a multiple of 32 px (one token per 32×32 px), so the *image tokens* actually produced land 3–6 % under it. Without the flag, llama.cpp applies its own cap of 4096 tokens for Qwen-VL models: this screenshot stays under it and is only rounded to 1888×960 px, i.e. 1770 tokens. Every prompt also carries ~186 text tokens (instruction and JSON schema), the same for every budget.

The raw data is in [results/](results/): one `results.csv` per model, plus server logs and screenshots annotated with every click (`clicks-<budget>.png`).

### Caveats

- **4 targets on 1 screenshot is a smoke test, not a benchmark.** Each target is 25 % of the hit rate. Some results are likely noise: the 9B misses the heart icon at 1024 tokens but hits it at 512, and two misses are within 1.5 px of the box edge (0.04 and 1.3 px). That's why the median distance to the target center is reported next to the hit count.
- **Prefill time grows faster than the token count.** Each token attends to all the previous ones, so the cost per token rises with the prompt length: the 4B processes ~31 tokens/s at 128 tokens but ~16 at full resolution. That's why the table reports prefill in seconds, not tokens/s, which aren't comparable across budgets.
- **The 4B default-budget timings drifted** from 18.7 to 12.3 prefill tok/s over 4 requests, probably CPU throttling. Treat its 131 s as pessimistic.
- **The published screenshot was edited after the runs**: the browser bookmarks, profile icons and Hugging Face username are painted over. No target is affected, but re-running on this image may not reproduce the exact clicks.
- llama.cpp warns that Qwen-VL-based models (which Holo is) need at least 1024 image tokens for reliable grounding. The results above show where that floor actually starts to bite.

## Setup

Hardware: Intel i7-1165G7 (4 cores), CPU only, WSL2 with 10 GB of RAM.
Software: llama.cpp build 12480, Python 3.13, [uv](https://docs.astral.sh/uv/).

1. Build [llama.cpp](https://github.com/ggml-org/llama.cpp) in `~/llama.cpp` (`llama-server` is expected in `~/llama.cpp/build/bin/`).
2. Download the quantized models (Q4_K_M weights + Q8_0 vision projector):

   ```bash
   uvx --from huggingface_hub hf download mradermacher/Holo-3.1-4B-GGUF \
     Holo-3.1-4B.Q4_K_M.gguf Holo-3.1-4B.mmproj-Q8_0.gguf --local-dir ~/models/holo-3.1-4b
   uvx --from huggingface_hub hf download mradermacher/Holo-3.1-9B-GGUF \
     Holo-3.1-9B.Q4_K_M.gguf Holo-3.1-9B.mmproj-Q8_0.gguf --local-dir ~/models/holo-3.1-9b
   ```

3. Install the project: `uv sync`.

## Usage

**Run a sweep.** It starts and stops its own `llama-server` for each budget:

```bash
uv run holo-sweep screenshots/hf_home.png screenshots/hf_home.json --model 9b
```

Targets are a JSON list of `{name, element, box: [x0, y0, x1, y1]}`, where `element` is the natural-language instruction given to the model and `box` the ground truth in pixels.

**Rebuild the table and chart** from one or more sweeps:

```bash
uv run holo-report results/*/results.csv --screenshot screenshots/hf_home.png
```

**Locate a single element** with a `llama-server` already running on port 8080:

```bash
uv run holo-locate screenshots/hf_home.png "the 'Models' link in the top navigation bar"
```

## How it works

- [`locate.py`](src/holo_playground/locate.py) sends the screenshot and the [grounding prompt documented by H Company](https://hub.hcompany.ai/models-api/element-localization.md) through llama-server's OpenAI-compatible API. A JSON schema constrains decoding, so the model can only output `{"x": int, "y": int}`. Holo answers in a normalized 0–1000 space, which is scaled back to pixels.
- [`sweep.py`](src/holo_playground/sweep.py) restarts the server for each `--image-max-tokens` value, with `--no-cache-prompt` so that every request re-processes the full image and timings stay comparable. It scores each click against its box and records llama-server's per-request timings (prefill and generation token counts and durations).
- [`report.py`](src/holo_playground/report.py) aggregates the CSVs into the table and chart above.

## Next steps

- 20–30 targets over several screenshots, so that hit rates mean something.
- Accuracy as a function of target size, the likely real driver.
- A 768-token budget, between the 9B's last clean result (512) and the 4B's (1024).
