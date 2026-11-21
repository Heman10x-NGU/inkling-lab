# Inkling 2 Mini lab

Measurements, tools, and negative results from two days of work on Inkling 2 Mini (Thinking Machines' 26.4B looped mixture-of-experts model) on a 16 GB MacBook Air M4.

Dates: 2026-11-20 to 2026-11-21. Every number in this repo came from a script in `scripts/` or a run logged in `results/`. The full record is [notes/findings.md](notes/findings.md).

## Setup under test

- Model file: `Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf` from `AtomicChat/Inkling-2-Mini-GGUF` on Hugging Face (7,900,678,464 bytes; specialists at about 2 bits, attention at 8 bits).
- Runtime: llama.cpp pull request 32695 at commit `6196d98`, Metal build.
- Official weights (`thinkingmachines/Inkling-2-Mini`) were read by partial HTTP requests only, never downloaded in full.

## What each line of work found

| Line of work | Result | Status |
|---|---|---|
| Weight structure (shared or low-rank parts) | Specialists share nothing (mean cosine 0.0008); attention needs 45 to 57% of full rank for 90% of the energy | Closed, negative |
| Volume correction of the 2-bit specialists | Best scale is the stock value within noise | Closed, negative |
| Exact graph speedups in the Metal path | 3.9% faster with bit-identical output | Closed, under the 8% stop line |
| Fewer loop passes | Two loops agree with stock on 74.1% of top words | Closed, negative |
| Thinking off by weight edit | A prompt prefill already does it on 12 of 12 prompts | Closed, not needed |
| Long context on a 16 GB Mac | 32,768 window: 1.2 tokens per second at the default batch, 18.4 with `-b 64 -ub 64` | Working fix |
| Metal profiler | Per-node GPU timing, opt-in by environment variable | Working tool |
| inkscope (internal-state capture) | Loads the model once, runs a prompt file, records any named state, accepts an adapter | Working tool |
| Memory patch (facts into attention output matrices, no training) | Round 1: 7 of 20 reworded questions, 3 of 10 controls correct, KL 0.131. Round 2: recall and damage trade off, no setting passes | Closed, negative |

The memory patch does store specific facts: fitted on 5 of the 10 made-up facts, it recalls those 5 and none of the others. It fails on specificity. Details and every configuration: [notes/memory_patch_results.md](notes/memory_patch_results.md).

The long-context cause (a working buffer that grows with batch size times context window) was diagnosed upstream by edwardyoon in the discussion on llama.cpp pull request 32694. This repo adds the measurements on a 16 GB machine.

## Layout

| Path | Contents |
|---|---|
| `notes/findings.md` | All measured numbers, kept current |
| `notes/architecture.md` | Forward-pass specification of the model as implemented |
| `notes/*_plan.md` | End state, pass line, and stop rules, each written before its experiment |
| `notes/metal_profile.md`, `notes/runtime_speed.md` | Where a token's time goes, and the speedup attempt |
| `notes/inkscope.md` | Usage of the capture tool |
| `notes/memory_patch_results.md` | Both memory patch rounds |
| `notes/patch_prior_art.md`, `notes/ncm_and_neurons.md` | Reading notes and hypotheses |
| `patches/` | Three patches against llama.cpp (see below) |
| `scripts/` | Weight readers, scoring, fact checks, the patch fitter |
| `eval/` | Scoring text, the 10-fact test set, round 2 fitting and protection sets |
| `results/` | Run logs and JSON outputs |

## Patches

| Patch | Applies on | What it adds |
|---|---|---|
| `patches/metal_profile.patch` | commit `6196d98` | `GGML_METAL_PROFILE=1` or `2` and `GGML_METAL_PROFILE_OUT` for per-node GPU timing |
| `patches/inkling_fast.patch` | `6196d98` plus `metal_profile.patch` | Three exact graph changes (6,961 to 6,068 GPU dispatches per token) |
| `patches/inkscope.patch` | commit `6196d98` | The `llama-inkscope` tool, the `wo_inp` capture point, `INKSCOPE_DUMP`, and the `INKLING_SKIP_PASSES` switch |

## Rebuild the environment

The model file and the llama.cpp checkouts are not in this repo. Scripts expect this layout under the repo root, or under `$INKLING_LAB` when that variable is set.

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

Store baseline probabilities, then score a changed setting against them (about 90 seconds per run):

```bash
zsh scripts/ppl.sh base tune
zsh scripts/ppl.sh run tune --lora results/patch/some_adapter.lora.gguf
```

Check the 10 facts with the document in the prompt and without it:

```bash
python3 scripts/facts_check.py --modes doc,nodoc --q held
```

Python scripts that read weights need numpy: run them with `uv run --with numpy python scripts/<name>.py`.

## Not in this repo

- Adapter files from the memory patch runs (8 to 33 MB each), stored baseline probabilities (1.1 GB), captured states, and raw profiler dumps. The scripts regenerate all of them.
- The model's chat template, which ships inside the model file.

Absolute paths in logs were replaced with `$LAB`, `$HOME`, and `$SCRATCH`.

## Measurement caveats

- The machine is fanless. Within one hot run, time per token drifted from 56 to 166 ms. Speed figures are from cool, interleaved runs unless a table says otherwise.
- Drift (KL) was measured on source-code text only.
- The memory patch test set is 10 made-up facts. It is a go or no-go probe.

## Text sources

`eval/tune*.txt` and `eval/test*.txt` are excerpts of llama.cpp source and documentation (MIT license).
