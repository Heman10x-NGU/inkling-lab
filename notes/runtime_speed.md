# Runtime speed: fewer tiny GPU kernels in the Inkling 2 graph

Date: 2026-11-21. Repo `runtime/llama.cpp`, branch `inkling-fast` (3 commits on top of `metal-profile` c58bfb2f8, head 27028ff5e, not pushed). Patch: `patches/inkling_fast.patch` (diff of `inkling-fast` against `metal-profile`). Plan followed: `runtime_speed_plan.md`.

## Result

Steps 1 to 3 together cut GPU dispatches per generated token from 6,961 to 6,068 (13%) and time per generated token by about 2 ms (3 to 4%), with output bit-identical to stock. That is under the plan's 8% stop line, so step 4 (merged projections) and step 5 were not started.

| pass-line item | target | result | met |
|---|---|---|---|
| text identical, temperature 0, 5 prompts | identical | identical at every step | yes |
| `ppl.sh` mean KLD | at most 0.0001 | 0.000000 at every step | yes |
| `ppl.sh` same top p | at least 99.9% | 100.000% at every step | yes |
| ms per generated token | 15% lower (45 ms or less) | 3.9% lower (52.66 vs 54.79 ms, means of 5 interleaved runs) | no |
| dispatches per token | under 4,500 | 6,068 | no |

## One row per step

Speed: `eval time` of `llama-completion -n 64`, prompt `def quicksort(arr):`, stock `build/` and the patched binary interleaved (stock, patched, stock, patched), each pair after at least 120 s with no other model process (60 s for step 1), 60 s between the two runs of a pair. A pair is discarded when either run exceeds 60 ms per token. The patched column is the build at that step (steps are cumulative); every step is compared with stock in its own session.

| step | dispatches/token | stock ms/token (per run) | patched ms/token (per run) | mean diff | text identical | mean KLD | same top p |
|---|---:|---|---|---:|---|---:|---:|
| baseline (`metal-profile`) | 6,961 | | | | | 0.000000 | 100.000% |
| 1. depth-mix stack once per block | 6,546 | 53.81, 52.87, 52.94, 52.76 (mean 53.10) | 52.01, 52.13, 51.75, 51.67 (mean 51.89) | -1.21 (-2.3%) | yes (5/5) | 0.000000 | 100.000% |
| 2. relative-bias pad at load | 6,498 | 56.08, 55.24, 54.76, 54.74 (mean 55.21) | 53.74, 53.67, 53.96, 53.79 (mean 53.79) | -1.42 (-2.6%) | yes (5/5) | 0.000000 | 100.000% |
| 3. q and k clipped together, thresholds filled once | 6,068 | 52.05, 57.38, 54.73, 54.86, 54.93 (mean 54.79) | 50.41, 52.94, 53.79, 53.06, 53.12 (mean 52.66) | -2.13 (-3.9%) | yes (5/5) | 0.000000 | 100.000% |

Per-pair differences at step 3: -1.64, -4.44, -0.94, -1.80, -1.81 ms (median -1.80 ms, -3.3%). Stock ran 52 to 57 ms in these sessions, so the machine was not fully cool for every pair; both binaries ran under the same conditions within a pair. The step-2 change removes 48 small PAD kernels (0.24 ms isolated in the profile) and is below what this method can resolve.

Discarded pairs (hot, or another engineer's GPU job running at the same time): step 1 one pair (stock 65.72, patched 59.31); step 2 two pairs (82.24 / 97.15 and 132.72 / 64.64). Raw logs: `results/fast/speed_s1.log`, `speed_s1b.log`, `speed_s2.log`, `speed_s3.log`.

Dispatch counts come from mode 1 of the profiler (`GGML_METAL_PROFILE=1`), the only mode that records them: mode 2 writes timestamps but no dispatch count (`write_json` prints `n_dispatch` only for profiled graphs). Raw: `results/fast/disp_*.json`.

## What each step changed

1. Depth mixers (`mix`). The stacked block states were rebuilt with one CONCAT per earlier block on every mix (144 mixes per token). The stack of finished block sums (newest first, the order the old loop produced) is now extended by one CONCAT when a block finishes, and each mix does one CONCAT with the current partial sum, then the same RMS_NORM over all rows. The previous pass's bank (two states, fixed for the whole pass) is normalized and transposed once per pass and shared by its 24 message mixes. For a single token, the REPEAT of the queries is skipped, since the query already has the broadcast shape. Normalizing each finished block once and only the partial sum per mix would need the same CONCAT and RMS_NORM count, so it brings no dispatch saving; the single RMS_NORM over the concatenated stack keeps the bits identical.
2. Relative-bias table. The zero-padded table is made once after load and kept with the model. This needed a small loader addition in `llama_model_base`: `load_tensors_derived` (allocate tensors computed from loaded weights on the device buffer type of a source weight) and a hook `load_arch_tensors_post` called after all tensor data is loaded. A model loaded without data (memory fitting in `common_fit_params`) has no derived tensors, and the graph falls back to the PAD node. Added memory: 24 x 65,600 bytes = 1.5 MiB.
3. Clip. q (after its per-channel scale) and k have the same threshold and head size, so their rows are concatenated and clipped by one SQR, SUM_ROWS, SQRT, SCALE, DIV, CLAMP, MUL chain; the threshold tensor (FILL) is made once per shape per graph. Per layer visit, the q, k and r clips went from 26 kernels to 17. Every row goes through the same kernels as before, so the output is bit-identical.

## Tried and dropped

- **Clip as `c * l2_norm(z, eps = c)`** (2 kernels per clip instead of 7, about 860 fewer dispatches per token). In exact arithmetic `z*min(1, c/(|z| + 1e-6)) = c*z/max(|z|, c)` up to the 1e-6. It kept the decode text identical on all 5 prompts, but `ppl.sh` gave mean KLD 0.001352 and same top p 98.92%, failing the pass line by 13x. With the old chain kept for q and k and the L2 form for r alone (threshold 8, where unclipped rows stay exact): KLD 0.001547, 98.99%. The L2_NORM kernel sums `dot(x, x)` per float4 in a different order than SQR + SUM_ROWS, and the relative-position bias is added to the attention mask and rounded to F16 for flash attention, so one-ulp differences in r move bias values across F16 rounding steps. Conclusion: any change to the clip must keep the same reduction order.
- **Fused Metal clip kernel** (not built). With the clip chain now at 2 per visit, a fused kernel would remove about 576 more dispatches; the remaining chain costs about 1.7 ms isolated per token (mode 1, step-3 build), so the expected real saving is about 1 to 1.5 ms. It would have to reproduce SUM_ROWS's thread layout and avoid FMA contraction to stay bit-identical (the L2 result above shows that matters). Even at the top of that range, steps 1 to 3 would stay under 8%, so it was not attempted.

## Bugs I made and fixed on the way

- Step 2, first version: the derived tensors were allocated on the buffer type of the source weight, which is `MTL0_Mapped` (the type that wraps the mapped model file). Moved to the device's own buffer type.
- Step 2, second version: wrong weights. After loading, `load_tensors` copies each physical layer's tensors into the slots of all its visits, and physical layers 18 to 23 end up in slots 42 to 47; slots 18 to 23 then hold copies of core layers 6 to 11. The first post-load hook read `layers[18..23]` and padded the wrong tables. Text diverged after a few words and `ppl.sh` gave KLD 0.152. Fixed by reading each physical layer from the slot of its first visit, `loop_visit(ip)`. Neither version was committed.
- Step 2 also crashed (SIGSEGV in `ggml_mul_mat`) during memory fitting, which loads the model without data, so the post-load hook never ran. Fixed by the graph fallback above.

## What is left, by measured size

From the step-3 mode-1 profile (isolated GPU time per token; not real speed, see `metal_profile.md`): CONT 458 dispatches (2.7 ms), CONCAT 441 (1.6 ms), MUL 608 (1.8 ms), SCALE 581 (1.6 ms), CPY 289 (1.5 ms), TANH 289 (1.0 ms).

Measured cost per dispatch: steps 1 to 3 removed 893 dispatches per token and saved 2.13 ms (mean of 5 pairs), about 2.4 us per removed dispatch on this machine. At that rate, reaching the 4,500 dispatch line (1,568 more) would save about 3.8 ms more, which still leaves the 15% speed line out of reach by dispatch removal alone.

**Next change I would make: batch the loop message controllers.** Each of the 96 controller calls per token (4 per core layer in passes 2 and 3) runs 11 kernels: CONCAT, RMS_NORM, the down matmul, separate gain and forget matmuls, two TANH, SCALE, two MUL, ADD, so 1,056 dispatches per token. The gain and forget matmuls read the same input and are latency-bound (192 calls, about 2.6 ms isolated, 12 GB/s in `metal_profile.md`). Concatenating the gain and forget rows at load (Q8_0 rows of 64 inputs are whole blocks, so rows append cleanly) and stacking the four modules of a layer as one batched matmul would cut this to about 11 dispatches per layer visit (about 790 fewer per token). Measured ceiling: the loop-messages-and-controllers region was 6.6 ms isolated with 1,334 dispatches (`metal_profile.md`, before these steps; step 1 removed only its message-mix CONCAT, RMS_NORM and CONT work). Expected real saving at 2.4 us per dispatch: about 1.9 ms. The plan's step 4 (merged q, gate, relative-position and sink projections) has the larger isolated ceiling (14.4 ms over 288 calls) but those matmuls already run at 71 to 78 GB/s, so its expected saving (2 to 3 ms per the plan, about 0.5 ms from its 190 removed dispatches) depends on bandwidth gains that are not measured, and it adds about 500 MB of duplicated weights unless the originals are dropped.

## Not verified

- Speed on a fully cool machine for every pair: stock ran 52 to 57 ms in the step-3 session. The step-1 and step-3 results agree in direction and size (2 to 4%), but the step-2 effect is inside the noise.
- Long history: all speed runs used a 7-token prompt. The changes do not touch the attention-over-history path.
- Other backends: only Metal was run. The changes use existing ggml operations, and step 2's loader helper runs for this architecture only.
- Another engineer ran GPU jobs (`llama-eval-callback`, batch 4096) during part of this work. My runs waited for their model processes to exit, and speed pairs waited for 120 s with no model process, but heat from their runs may have carried into some pairs.

## Exact commands

```sh
cd runtime/llama.cpp
git checkout -b inkling-fast metal-profile
uvx cmake -B build-fast -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
uvx cmake --build build-fast --config Release -j 8 --target llama-completion llama-bench llama-perplexity
git diff metal-profile inkling-fast > ../../patches/inkling_fast.patch

M=../../models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf

# text and speed (stock build/ and patched, interleaved); prompts:
#   "def quicksort(arr):"  "The history of the printing press begins in"
#   $'Here are five tips for better sleep:\n1.'  "Explain how a hash table works."
#   "Dear Anna, thank you for your letter. Last week I"
build/bin/llama-completion      -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 64 -p "def quicksort(arr):"
build-fast/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 64 -p "def quicksort(arr):"

# probabilities against the stored stock baseline (batch 512, context 512)
MODEL=$M ../../scripts/ppl_fast.sh run tune     # ppl.sh with B=build-fast/bin

# dispatches per token (mode 1 only records them)
GGML_METAL_PROFILE=1 GGML_METAL_PROFILE_NODES_MAX=1 GGML_METAL_PROFILE_OUT=disp.json \
  build-fast/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 6 -p "def quicksort(arr):"
```

Per-step speed runs used copies of the step binaries (`build-fast/snap-s2`, `build-fast/snap-s3`, run with `DYLD_LIBRARY_PATH` set to that folder), so that steps could be timed together in a quiet window. Text outputs and logs: `results/fast/`.
Harness used for all runs: `results/fast/harness.sh` (`text`, `speed`, `disp` subcommands).
