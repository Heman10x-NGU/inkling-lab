# inkling-lab

Apple Metal profiling, `llama.cpp` C++ patches, and weight-space analysis for Thinking Machines' **Inkling 2 Mini** (26.4B looped mixture-of-experts) on a 16 GB MacBook Air M4.

Built within 48 hours of release (2026-11-20 to 2026-11-21). Every number comes from a script in `scripts/` or a logged run in `results/`. Full record: [`notes/findings.md`](notes/findings.md).

## Highlights

- 15x faster 32K context on 16 GB Apple Silicon. At default batch size, a 32,768-token window allocates a 3,404 MiB working buffer, pushes the 7.35 GiB model into swap, and collapses decode to 1.2 tok/s. Setting `-b 64 -ub 64` shrinks that buffer 8x (to 425 MiB) and restores 18.4 tok/s at 32K context (18.1 tok/s at 65K with `-b 16`). A real 26,065-token source-code prompt runs at 8.4 GB peak resident memory.
- Three C++/Metal patches for `llama.cpp`: per-node Metal GPU timing (`metal_profile.patch`), a single-load activation capture and loop-control binary (`inkscope.patch`), and three bit-exact graph edits that cut 893 GPU dispatches per token (`6,961 -> 6,068`, ~3.9% faster decode).
- Per-op Metal profile of the 26.4B looped MoE on a cool M4 (52-54 ms/token wall clock, 98% GPU busy). Isolated GPU time breaks down into 23.8 ms attention, 18.0 ms expert MLP, 6.6 ms loop controllers, and 16.9 ms across norms, convs, depth mixing, and the output head. Weight matmuls run at 49-88 GB/s, showing kernel efficiency rather than dispatch gaps is the main bottleneck.
- Weight-geometry dissection of the official `thinkingmachines/Inkling-2-Mini` checkpoint via partial HTTP range requests (zero full weight download). Verified unit-norm rows, expert orthogonality (0.0008 mean cosine, 128/128 participation ratio), attention rank (45-57% for 90% energy), 28 near-duplicate router pairs (18 in layer 17), loop-controller stream weights, and the dimension-2027 outlier channel (8.2x median in layer-17 routing, 0.03x in final norm).
- Loop-depth and weight-editing probes. Measured exact perplexity cost of skipping middle loops (3 loops 2.16 PPL -> 2 loops 4.12 PPL / 74.1% top-1 agreement) and built a closed-form attention-weight fact editor (`scripts/memory_patch.py`) that stores 5 of 5 targeted facts without backprop while mapping the recall-vs-KL-drift trade-off.

## C++ and Metal tooling (`patches/`)

All patches target `llama.cpp` PR 32695 at commit `6196d98` (Metal build).

| Patch | What it adds | Measured output |
| :--- | :--- | :--- |
| `patches/metal_profile.patch` | Opt-in per-node Metal GPU timing via `GGML_METAL_PROFILE=1\|2` and `GGML_METAL_PROFILE_OUT` | Full op-level breakdown of all 6,961 GPU dispatches per token |
| `patches/inkscope.patch` | `llama-inkscope` binary, `wo_inp` activation capture, `INKSCOPE_DUMP`, and `INKLING_SKIP_PASSES` | Single-load prompt runner with arbitrary tensor dumps, LoRA loading, and loop-count control |
| `patches/inkling_fast.patch` | Three bit-exact Metal graph simplifications (applies on top of `metal_profile.patch`) | Cuts dispatches from 6,961 to 6,068 per token (-12.8%) and trims decode latency ~3.9% |

## Key benchmarks and findings

### 1. Long-context memory and throughput on 16 GB M4

| Context window | Batch (`-b / -ub`) | Working buffer | History buffers | Decode speed |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 512 (default) | 980 MiB | 256 + 240 MiB | 17.5 tok/s |
| 32,768 | 512 (default) | 3,404 MiB | 1,024 + 240 MiB | 1.2 tok/s |
| 32,768 | 64 | 425 MiB | 1,024 + 200 MiB | 18.4 tok/s |
| 32,768 | 16 | - | - | 19.2 tok/s |
| 65,536 | 16 | 321 MiB | 2,048 + 200 MiB | 18.1 tok/s |

The working buffer scales with batch size times context window (upstream root cause identified by `edwardyoon` in `llama.cpp` PR 32694; this repo contributes the 16 GB M4 measurements and 26K-token prompt test).

### 2. Where one token spends GPU time (cool M4, ~52-54 ms/tok)

| Subsystem | Isolated GPU time | Notes |
| :--- | ---: | :--- |
| Attention | 23.8 ms | Only subsystem that grew (+7 ms) at 7,608-token history |
| Expert MLP | 18.0 ms | 2-bit specialists; 96 specialist matmuls per token |
| Loop messages and controllers | 6.6 ms | Middle 12 layers looped 3x |
| Norms and clips | 4.4 ms | High count of tiny ops (5,568 of 6,961 dispatches take <= 5 us) |
| Output head | 4.4 ms | Skipping vocabulary projection shifts total speed by 4% |
| Token convolutions (k=4) | 4.3 ms | Placed before attention and experts |
| Depth mixing | 3.8 ms | Cross-layer residual mixing |

### 3. Architecture, quantization, and weight-editing probes

| Probe | What was measured | Takeaway |
| :--- | :--- | :--- |
| Official weight geometry (`scripts/expert_structure.py`, `attn_spectra.py`) | Layer-8 specialists have 0.0008 mean cosine (128/128 participation); attention needs 45-57% rank for 90% energy | Experts share almost no subspace; naive low-rank compression will drop quality |
| 2-bit quantization error (`scripts/damage_map.py`) | 8-bit attention has 0.5% relative error vs official weights; 2-bit specialists have 34% uniform error (0.9396-0.9403 cosine) | Rescaling specialist volume cannot beat stock least-squares scale (verified in `results/volume_sweep_tune.txt`) |
| Loop-pass ablation (`INKLING_SKIP_PASSES`) | 3 loops: 2.16 PPL (100% top-1); 2 loops: 4.12 PPL (74.1% top-1, KL 0.62); 1 loop: 4.94 PPL (69.1% top-1, KL 0.79) | Loops 2 and 3 share identical weights, and skipping a pass costs 2x perplexity |
| Thinking control (`scripts/think_probe.py`) | Effort 0 still emitted ~88 thinking tokens on 12/12 prompts; prefilling the answer channel disabled thinking on 12/12 | Solvable cleanly in the prompt template with zero weight modification |
| Training-free fact injection (`scripts/memory_patch.py`) | Fitted closed-form LoRA edits into `wo_inp` on 5 of 10 synthetic facts: 5/5 fitted facts recalled, 0/5 unfitted facts leaked; round-1 reworded recall 7/20, control accuracy 3/10, KL 0.131 | Targeted fact storage works without backprop, but round-1 and round-2 sweeps showed a steep recall-vs-specificity trade-off ([`notes/memory_patch_results.md`](notes/memory_patch_results.md)) |

## Quickstart

Model weights (`Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf`, 7.90 GB from `AtomicChat/Inkling-2-Mini-GGUF`) and `llama.cpp` checkouts are kept outside git. Scripts resolve paths from the repo root or `$INKLING_LAB`.

Build stock `llama.cpp` at the tested PR commit:

```bash
git clone https://github.com/ggml-org/llama.cpp runtime/llama.cpp
git -C runtime/llama.cpp fetch origin pull/32695/head
git -C runtime/llama.cpp checkout 6196d98
cmake -S runtime/llama.cpp -B runtime/llama.cpp/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/llama.cpp/build -j
```

Build with `llama-inkscope` (activation capture and loop control):

```bash
git -C runtime/llama.cpp worktree add ../llama-loops 6196d98
git -C runtime/llama-loops apply ../../patches/inkscope.patch
cmake -S runtime/llama-loops -B runtime/llama-loops/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/llama-loops/build -j
```

Place the GGUF model at `models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf`.

Run paired perplexity and KL scoring (~90 s per run):

```bash
zsh scripts/ppl.sh base tune
zsh scripts/ppl.sh run tune --lora results/patch/some_adapter.lora.gguf
```

Run the 10-fact recall check (with and without in-context document):

```bash
python3 scripts/facts_check.py --modes doc,nodoc --q held
```

Weight-inspection scripts require `numpy`:

```bash
uv run --with numpy python scripts/<script>.py
```

## Repository map

| Path | Contents |
| :--- | :--- |
| `patches/` | Three `llama.cpp` C++/Metal patches (`metal_profile`, `inkscope`, `inkling_fast`) |
| `scripts/` | Official HTTP range-request weight readers, paired PPL/KL harness, fact checks, and closed-form patch fitter |
| `notes/` | Detailed write-ups: [`findings.md`](notes/findings.md), [`architecture.md`](notes/architecture.md), [`metal_profile.md`](notes/metal_profile.md), [`runtime_speed.md`](notes/runtime_speed.md), [`inkscope.md`](notes/inkscope.md), [`memory_patch_results.md`](notes/memory_patch_results.md), plus pre-registered `*_plan.md` stop rules |
| `eval/` | Paired scoring text (MIT excerpts from `llama.cpp`), synthetic fact sets, and protection chat prompts |
| `results/` | Committed JSON outputs and raw benchmark logs (paths scrubbed to `$LAB`, `$HOME`, `$SCRATCH`) |

## Methodology notes

- Every experiment wrote down pass criteria and stop rules in `notes/*_plan.md` before running. For example, `inkling_fast.patch` hit +3.9% bit-exact speedup and stopped because the pre-set target was 8%.
- The MacBook Air M4 is fanless and throttles ~20% under sustained load (56 to 166 ms/tok in one continuous hot run). Speed comparisons use cool, interleaved runs unless noted.
- Large regenerable artifacts (8-33 MB LoRA adapters, 1.1 GB baseline probability dumps, raw profiler traces) are omitted from git; scripts regenerate them.

## License

MIT License (see [`LICENSE`](LICENSE)). `eval/tune*.txt` and `eval/test*.txt` contain excerpts from `llama.cpp` (MIT).
