# inkling-lab

Two days of benchmarks, `llama.cpp` Metal patches, and negative results from running Inkling 2 Mini on a 16 GB MacBook Air M4.

Inkling 2 Mini is Thinking Machines' 26.4B mixture-of-experts model whose middle 12 layers run in a 3-pass loop. Out of the box, it ran at 1.2 tokens per second at a 32K context window on this laptop. Lowering the batch size (`-b 64 -ub 64`) brought speed up to 18.4 tokens per second. Along the way I built a per-node Metal GPU profiler and an internal-state capture tool (`inkscope`) on top of `llama.cpp`, then used them to test six speed and weight-editing ideas. Most of those six ideas failed, and every run is logged here.

Dates: 2026-11-20 to 2026-11-21. Every number comes from a script in `scripts/` or a log in `results/`. Full record: [`notes/findings.md`](notes/findings.md).

## What worked and what failed

| Experiment or tool | Result | Status |
| :--- | :--- | :--- |
| Long context on 16 GB RAM | At a 32,768 window, speed went from 1.2 tok/s at default batch size to 18.4 tok/s with `-b 64 -ub 64` | Working fix |
| Metal GPU profiler (`patches/metal_profile.patch`) | Opt-in per-node Metal timing via environment variables | Working tool |
| State capture (`patches/inkscope.patch`) | Loads the model once, runs a prompt file, dumps any named tensor, and accepts a LoRA adapter | Working tool |
| Exact Metal graph speedups (`patches/inkling_fast.patch`) | Cut GPU dispatches from 6,961 to 6,068 per token with bit-identical output, gaining 3.9% speed | Closed, under the 8% stop line |
| Shared or low-rank weights | The 128 specialists share almost nothing (mean cosine 0.0008); attention needs 45% to 57% of full rank for 90% of energy | Closed, negative |
| Rescaling 2-bit specialists | Stock weight scale matched every tested rescaling within noise | Closed, negative |
| Skipping middle loop passes | Running 2 loops instead of 3 matched stock on only 74.1% of top next tokens | Closed, negative |
| Turning off thinking by editing weights | Not needed; a prompt prefill stopped thinking on 12 of 12 prompts | Closed, not needed |
| Editing facts into attention weights (no training) | Round 1 got 7 of 20 reworded questions and 3 of 10 controls right (KL drift 0.131); round 2 could not raise recall without hurting general output | Closed, negative |

The weight-edit test did store real facts: fitted on 5 of 10 made-up facts, the model recalled those 5 and none of the other 5. It failed on specificity and damaged general outputs. Every configuration and score is in [`notes/memory_patch_results.md`](notes/memory_patch_results.md).

The long-context slowdown comes from a working buffer that grows with batch size times context window. That root cause was diagnosed upstream by `edwardyoon` in `llama.cpp` PR 32694; this repo adds the 16 GB M4 measurements.

## Setup under test

- Model file: `Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf` from `AtomicChat/Inkling-2-Mini-GGUF` on Hugging Face (7,900,678,464 bytes; specialists at about 2 bits, attention at 8 bits).
- Runtime: `llama.cpp` PR 32695 at commit `6196d98`, Metal build on macOS.
- Official weights (`thinkingmachines/Inkling-2-Mini`) were inspected by partial HTTP range requests only, never downloaded in full.

## Repository layout

| Path | Contents |
| :--- | :--- |
| `notes/findings.md` | All measured numbers in one place |
| `notes/architecture.md` | Forward-pass layout of the model as implemented |
| `notes/*_plan.md` | Pass lines and stop rules written before each experiment |
| `notes/metal_profile.md`, `notes/runtime_speed.md` | Where token time goes and the graph speedup test |
| `notes/inkscope.md` | How to use the state capture tool |
| `notes/memory_patch_results.md` | Round 1 and round 2 weight-edit results |
| `notes/patch_prior_art.md`, `notes/ncm_and_neurons.md` | Reading notes and hypotheses |
| `patches/` | Three patches against `llama.cpp` |
| `scripts/` | Weight readers, perplexity/KL scoring, fact checks, and the patch fitter |
| `eval/` | Scoring text, the 10-fact test set, and round 2 fitting/protection sets |
| `results/` | Raw run logs and JSON outputs |

## Patches

| Patch | Applies on | What it adds |
| :--- | :--- | :--- |
| `patches/metal_profile.patch` | commit `6196d98` | `GGML_METAL_PROFILE=1` or `2` and `GGML_METAL_PROFILE_OUT` for per-node GPU timing |
| `patches/inkling_fast.patch` | `6196d98` plus `metal_profile.patch` | Three exact graph changes cutting GPU dispatches from 6,961 to 6,068 per token |
| `patches/inkscope.patch` | commit `6196d98` | The `llama-inkscope` binary, `wo_inp` capture point, `INKSCOPE_DUMP`, and `INKLING_SKIP_PASSES` |

## How to rebuild

The model file and `llama.cpp` checkouts are not stored in this repo. Scripts expect this layout under the repo root, or under `$INKLING_LAB` when set.

Stock reference build:

```bash
git clone https://github.com/ggml-org/llama.cpp runtime/llama.cpp
git -C runtime/llama.cpp fetch origin pull/32695/head
git -C runtime/llama.cpp checkout 6196d98
cmake -S runtime/llama.cpp -B runtime/llama.cpp/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/llama.cpp/build -j
```

Build with the capture tool:

```bash
git -C runtime/llama.cpp worktree add ../llama-loops 6196d98
git -C runtime/llama-loops apply ../../patches/inkscope.patch
cmake -S runtime/llama-loops -B runtime/llama-loops/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/llama-loops/build -j
```

Place the model file at `models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf`.

## Common commands

Save baseline next-token probabilities, then score a changed run against them (about 90 seconds per run):

```bash
zsh scripts/ppl.sh base tune
zsh scripts/ppl.sh run tune --lora results/patch/some_adapter.lora.gguf
```

Check the 10 facts with the reference document in the prompt and without it:

```bash
python3 scripts/facts_check.py --modes doc,nodoc --q held
```

Python scripts that inspect weights need `numpy`:

```bash
uv run --with numpy python scripts/<name>.py
```

## What is not in this repo

- LoRA adapter files from the weight-edit runs (8 to 33 MB each), stored baseline probabilities (1.1 GB), captured states, and raw profiler dumps. The scripts rebuild all of them.
- The model chat template, which ships inside the GGUF file.

Absolute paths in logs were replaced with `$LAB`, `$HOME`, and `$SCRATCH`.

## Caveats

- The MacBook Air M4 has no fan. In one long hot run, time per token drifted from 56 ms to 166 ms. Speed numbers come from cool, interleaved runs unless a table notes otherwise.
- KL drift was measured on source code text only.
- The 10-fact weight-edit set is a small go/no-go probe, not a broad knowledge benchmark.

## Sources and license

`eval/tune*.txt` and `eval/test*.txt` are excerpts of `llama.cpp` source and documentation (MIT license). All scripts, patches, and notes in this repo are under the MIT license (see [`LICENSE`](LICENSE)).
