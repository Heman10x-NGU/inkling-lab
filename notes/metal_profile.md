# Where the time goes in one Inkling 2 Mini decode step (llama.cpp Metal, MacBook Air M4)

Date: 2026-11-21. Repo `runtime/llama.cpp`, branch `metal-profile` (one local commit on top of 6196d98, not pushed). Patch: `patches/metal_profile.patch`.

Every number below carries its mode:

- **normal**: the unmodified encode path. Wall time comes from llama-completion's own `eval time` line.
- **mode 2**: the normal encode path plus timestamps (`GGML_METAL_PROFILE=2`). It records CPU time in `graph_compute`, CPU encode time, and the GPU start and end of the graph's command buffers (MTLCommandBuffer `GPUStartTime`/`GPUEndTime`), read after completion. Mode 2 adds no waits, and its eval times match normal runs.
- **mode 1**: isolated kernels (`GGML_METAL_PROFILE=1`). Each node that dispatches GPU work runs in its own command buffer, which is committed and waited on. The reported GPU time is `GPUEndTime - GPUStartTime` of that command buffer. Mode 1 removes overlap between kernels and lets the GPU idle between them, so its per-kernel times are not real speed, and its wall times are only CPU and GPU round trips (about 145 us per node).

`GPUStartTime`/`GPUEndTime` exist in the SDK headers (`MTLCommandBuffer.h`, macOS 10.15+). They are usable here: per-kernel values go down to about 1.5 to 2 us, and a graph's span matches wall time within 1 to 2 ms. One value is **not** trustworthy: a command buffer that holds only an empty encoder reports 180 to 320 us. The final empty buffer in each profiled graph shows this, so the field is recorded but not used.

## How the backend encodes a graph (from reading the code)

- `ggml_metal_graph_compute` (`ggml-metal-context.m`) splits the graph into `n_cb + 1` command buffers. The calling thread encodes the first `max(64, 0.1 * n_nodes)` nodes and commits them right away. `n_cb` extra command buffers (here `n_cb = 1`) are encoded in parallel on a GCD queue (`dispatch_apply`). Boundaries move to the end of a fusable group (`ggml_metal_fusion_max`).
- Each encoder (`ggml_metal_op`, `ggml-metal-ops.cpp`) creates one concurrent `MTLComputeCommandEncoder`. It drops empty nodes up front, skips views, reshapes, transposes, and permutes, and inserts a memory barrier whenever a node reads or writes a range touched since the last barrier (`ggml_mem_ranges`). Fusion is table-driven (`ggml-metal-fusion.cpp`).
- `graph_compute` returns without waiting. llama.cpp then synchronizes to read the logits.

## The profiling patch

- `ggml-metal-profile.{h,cpp}` (new) holds the recorder and the JSON writer. The JSON is written in `ggml_metal_free` at exit, to `GGML_METAL_PROFILE_OUT`.
- `ggml-metal-device.m` adds a dispatch counter to the encoder wrapper. It is the only change that runs when profiling is off: one integer increment per dispatch.
- `ggml-metal-ops.cpp` adds `ggml_metal_op_set_cmd_buf` (end the current encoder and continue in a new command buffer), plus accessors for the node index and dispatch count.
- `ggml-metal-context.m`: in mode 1, single-token graphs go through `ggml_metal_graph_compute_profiled`. A single-token graph is one where `src1->ne[1] == 1` for the first MUL_MAT. Set `GGML_METAL_PROFILE_ALL=1` to profile every graph. Prompt graphs run normally and are only counted. That one encoder spans the whole graph, so fusion matches `n_cb = 0`. A node whose encode adds no dispatch is counted as empty and stays in the same command buffer. Fused groups are recorded under names joined with `+`. Mode 2 times the encode block per command buffer and harvests GPU times in `ggml_metal_synchronize`. The first profiled graph's full structure (op, name, sources) is dumped for region attribution.
- `GGML_METAL_PROFILE_NODES_MAX` (default 12) limits how many profiled graphs keep their per-node lists.

## A. Sanity checks

Generated text is byte-identical (md5 `6785032b...` short, `9675f039...` long) in every run: clean `build/`, `build-prof` with profiling off, mode 2, mode 1, and the prompt-cache runs.

Speed, short prompt, cool machine, interleaved runs with 40 s rests (normal mode, `eval time`):

| run | build/ (clean) ms/token | build-prof, profiling off ms/token |
|---|---:|---:|
| a | 52.56 | 53.81 |
| b | 53.81 | 52.56 |
| c | 52.91 | 54.81 |
| mean | 53.09 | 53.73 |

The 1.2% difference is inside run-to-run noise. Earlier campaign runs ranged from 53 to 125 ms per token for both binaries, for thermal reasons (see C).

## B. Short history ("def quicksort(arr):", 7 prompt tokens, 37 generated tokens profiled)

Mode 1, cool start, `results/metal_profile_short.json`. The first two generated tokens and the warmup decode are excluded.

- Graph nodes: 12,303. Nodes dropped before encoding (views, reshapes, transposes, permutes, empty): 4,564.
- Encode units that issue GPU work: **6,913 per token**. Kernel dispatches: **6,961 per token** (some ops issue more than one). Fusion absorbs 826 nodes. Encode calls that issued no GPU work: 0.
- Sum of isolated GPU time: **65.5 ms per token** (mode 1). Over four mode-1 runs this ranged from 65.5 to 76.4 ms, because small kernels run 30 to 50% slower when the GPU is warm.
- Sum of per-node wall time: 1,009 ms per token (mode 1 round trips, not real speed).
- 5,568 of the 6,913 units take 5 us or less each. Together they take 17.2 ms. 6,183 units take 10 us or less (21.0 ms).

| op | count/token | GPU ms/token | mean GPU us/dispatch | share |
|---|---:|---:|---:|---:|
| MUL_MAT | 1025 | 29.395 | 28.7 | 44.8% |
| MUL_MAT_ID | 96 | 15.340 | 159.8 | 23.4% |
| CONT | 504 | 3.034 | 6.0 | 4.6% |
| CONCAT | 572 | 2.185 | 3.8 | 3.3% |
| MUL | 656 | 2.046 | 3.1 | 3.1% |
| CPY | 289 | 1.524 | 5.3 | 2.3% |
| SCALE | 629 | 1.477 | 2.3 | 2.3% |
| ADD | 488 | 1.042 | 2.1 | 1.6% |
| TANH | 289 | 1.014 | 3.5 | 1.5% |
| FLASH_ATTN_EXT | 48 | 0.882 | 18.4 | 1.3% |
| RMS_NORM | 288 | 0.845 | 2.9 | 1.3% |
| SSM_CONV | 192 | 0.784 | 4.1 | 1.2% |
| RMS_NORM+MUL (fused) | 154 | 0.697 | 4.5 | 1.1% |
| REPEAT | 144 | 0.666 | 4.6 | 1.0% |
| SOFT_MAX | 144 | 0.607 | 4.2 | 0.9% |
| DIV | 144 | 0.453 | 3.1 | 0.7% |
| SUM_ROWS | 144 | 0.397 | 2.8 | 0.6% |
| GET_ROWS | 97 | 0.372 | 3.8 | 0.6% |
| FILL | 192 | 0.364 | 1.9 | 0.6% |
| SET_ROWS | 96 | 0.349 | 3.6 | 0.5% |
| SQR | 144 | 0.322 | 2.2 | 0.5% |
| SQRT | 144 | 0.313 | 2.2 | 0.5% |
| CLAMP | 145 | 0.297 | 2.1 | 0.5% |
| SIGMOID+...+ARGSORT+...+SUM_ROWS (fused MoE top-k) | 48 | 0.255 | 5.3 | 0.4% |
| SUB | 48 | 0.252 | 5.2 | 0.4% |
| SIGMOID | 96 | 0.250 | 2.6 | 0.4% |
| PAD | 48 | 0.240 | 5.0 | 0.4% |
| MUL+VIEWx7 (fused) | 48 | 0.139 | 2.9 | 0.2% |
| LOG | 1 | 0.002 | 2.0 | 0.0% |
| total | 6913 | 65.542 | | 100% |

Matmuls by the weight they read (mode 1, short). Effective bandwidth = bytes read per call divided by isolated time. Weight sizes come from the GGUF header; the experts count 8 of 128 per visit.

| weight (type) | calls/token | GPU ms/token | us/call | MB read/call | effective GB/s |
|---|---:|---:|---:|---:|---:|
| ffn_gate_up_exps (IQ2_XXS) | 48 | 10.668 | 222 | 10.95 | 49 |
| attn_gate (Q8_0) | 48 | 5.098 | 106 | 7.52 | 71 |
| attn_q (Q8_0) | 48 | 4.769 | 99 | 7.52 | 76 |
| attn_output (Q8_0) | 48 | 4.650 | 97 | 7.52 | 78 |
| ffn_down_exps (Q2_0) | 48 | 4.630 | 97 | 5.97 | 62 |
| output (Q6_K) | 1 | 4.340 | 4340 | 379.9 | 88 |
| attn_k / attn_v (Q8_0) | 96 | 3.638 | 38 | 2.51 | 66 |
| ffn_gate_inp (F32) | 48 | 0.929 | 19 | 1.18 | 61 |
| attn_rel (Q8_0) | 48 | 0.847 | 18 | 0.94 | 53 |
| 8 loop gain/forget (Q8_0) | 192 | ~2.6 | 13.5 | 0.16 | 12 |
| 4 loop down (Q8_0) | 96 | ~0.76 | 8 | 0.31 | 39 |
| activation x activation (mix scores, kq table) | 336 | 1.52 | 4.5 | n/a | n/a |

The weight bytes needed per token total 2.68 GB, which is 22.3 ms at 120 GB/s. This matches the earlier estimate.

## C. Long history (first 21,000 characters of `long_prompt_src.txt` = 7,608 tokens)

### C.1 The long-history slowdown is mostly heat

Mode 2 runs, normal encode path, `n_cb = 1`:

| condition | wall ms/token (`eval time`) | GPU span ms/token | CPU encode ms/token (all threads) | GPU span, first to last token |
|---|---:|---:|---:|---|
| short history, cool | 52.10 | 50.94 | 4.96 | flat at 51 |
| short history, cool (campaign repeat 2) | 53.45 | 52.02 | 5.17 | flat at 52 |
| long history, cool (state loaded with `--prompt-cache`, no prefill) | 59.47 | 58.12 | 5.28 | flat at 58 |
| long history, right after a 190 to 223 s prefill | 113.9 to 123.8 | 112.5 to 121.3 | 9.5 to 15.3 | 93 → 170 |
| **short** history, right after a long prefill | 112.02 | 113.14 | 7.93 | 56 → 166 → 107 |

A short-history decode run straight after a long prefill is as slow as a long-history decode (112 ms against 114 to 124 ms). Starting from a cool machine, 7,608 tokens of history add only **+7.2 ms per token** (50.9 → 58.1 ms GPU). The 110 to 128 ms "long-context" figures come from throttling on this fanless laptop after minutes of prefill at full load. CPU encode time also doubles when the machine is hot, so the whole chip slows down, not only the GPU. Within one hot run, GPU time per token drifts by up to 3x.

### C.2 What grows with history (mode 1, both runs cool)

`results/metal_profile_long.json` (prompt-cache load, cool) against `results/metal_profile_short.json`:

| op | short GPU ms/token | long GPU ms/token | delta | ratio |
|---|---:|---:|---:|---:|
| FLASH_ATTN_EXT | 0.882 | 5.664 | +4.782 | 6.4x |
| CPY (`kq_mask_rel_pos`, mask + bias copy) | 1.524 | 2.181 | +0.657 | 1.43x |
| CONT (`kq_b` permute) | 3.034 | 3.679 | +0.644 | 1.21x |
| GET_ROWS (`kq_b` gather by distance) | 0.372 | 0.725 | +0.353 | 1.95x |
| ADD (mask + bias) | 1.042 | 1.151 | +0.109 | 1.10x |
| everything else | | | under ±0.03 each | |
| total isolated GPU | 65.54 | 71.26 | +5.72 | |

| region | short GPU ms/token | long GPU ms/token | delta |
|---|---:|---:|---:|
| attention | 23.826 | 29.953 | **+6.127** |
| expert MLP | 17.962 | 17.858 | -0.104 |
| loop messages and controllers | 6.638 | 6.593 | -0.045 |
| depth mixing | 3.815 | 3.738 | -0.077 |
| token convolutions | 4.338 | 4.310 | -0.028 |
| norms, clips, other | 4.400 | 4.384 | -0.017 |
| output head | 4.416 | 4.100 | -0.316 |

All of the growth is in attention: flash attention (+4.8 ms) and the relative-position bias path (gather, permute, mask add and copy: +1.4 ms). Most of the 40 sliding-window visits are capped by the window, so most of this growth likely comes from the 8 full-attention visits. That split was not measured.

Long-history op table (mode 1, cool, top rows):

| op | count/token | GPU ms/token | mean GPU us/dispatch | share |
|---|---:|---:|---:|---:|
| MUL_MAT | 1025 | 28.824 | 28.1 | 40.4% |
| MUL_MAT_ID | 96 | 15.245 | 158.8 | 21.4% |
| FLASH_ATTN_EXT | 48 | 5.664 | 118.0 | 7.9% |
| CONT | 504 | 3.679 | 7.3 | 5.2% |
| CPY | 289 | 2.181 | 7.5 | 3.1% |
| CONCAT | 572 | 2.097 | 3.7 | 2.9% |
| MUL | 656 | 2.019 | 3.1 | 2.8% |
| SCALE | 629 | 1.439 | 2.3 | 2.0% |
| ADD | 488 | 1.151 | 2.4 | 1.6% |
| TANH | 289 | 1.029 | 3.6 | 1.4% |
| total | 6913 | 71.26 | | 100% |

A hot mode-1 long run (`results/metal_profile_long_hot.json`, taken right after prefill) gives the same deltas: FLASH_ATTN_EXT +4.8 ms, kq_mask CPY +0.8, kq_b CONT +0.7. Its small kernels are slower overall.

## D. Region attribution (mode 1, short history, 11 graphs averaged)

Method (`results/metal_profile_analyze.py regions`). Each node gets a label from its own name first (`Qcur/Kcur/Vcur/Rcur/kq_b/kq_mask_rel_pos/kqv_out` → attention, `ffn_moe_*` → expert, `loop_*_inp` → loop, `attn_out/ffn_out` → convolution, `norm-*` → norm, `ffn_inp/l_out` → other, `result_output` → head). If the name gives nothing, the label comes from the weight or cache it reads. Then structural anchors from `src/models/inkling2.cpp` apply:

- `clip()` starts at SQR.
- `mix()` is the REPEAT of a residual query, its scores MUL_MAT, and the RMS_NORM/CONCAT chain that builds the stacked states.
- `control()` is the MUL_MAT with a `loop_*_down` weight and its RMS_NORM/CONCAT input.
- `convolve()` is the CONCAT of the conv-history view with the transposed input.

Remaining nodes take the label of their latest labeled source, and anything still unlabeled takes the majority label of its consumers. Plain RMS_NORMs count as norms. The file `inkling2-loop.cpp` named in the task does not exist at this commit; the loop code is inside `inkling2.cpp`. **Unattributed: 0 dispatches, 0.000 ms.** Labels at region boundaries are judgment calls. For example, the clip of Q/K/R counts under "clips", and the `cross-loop` mixes that read `loop_res_query`/`loop_kv_res_query` count under region 3.

| region | dispatches/token | GPU ms/token | share | biggest pieces |
|---|---:|---:|---:|---|
| (1) attention: Q/K/V/R/sink/gate/output projections, rel-pos bias, scores, KV store | 1248 | 23.826 | 36.4% | MUL_MAT 19.65, CONT 0.91, FA 0.86 |
| (2) expert MLP: router, top-k, experts | 672 | 17.962 | 27.5% | MUL_MAT_ID 15.30, router MUL_MAT 0.93 |
| (3) cross-loop messages and 4 loop controllers | 1334 | 6.638 | 10.2% | MUL_MAT 3.73, TANH 0.66 |
| (4) depth mixing (attention-residual mixers) | 915 | 3.815 | 5.8% | CONT 1.10, CONCAT 0.86, MUL_MAT 0.69 |
| (5) token convolutions (incl. conv-state store) | 1208 | 4.338 | 6.6% | CPY 1.01, SSM_CONV 0.78, CONCAT 0.76 |
| (6) norms 0.94 / clips 2.75 / other 0.71 | 1579 | 4.400 | 6.7% | 1,120 dispatches are clips |
| (7) output head | 5 | 4.416 | 6.8% | one Q6_K MUL_MAT, 4.34 ms |
| unattributed | 0 | 0 | 0% | |
| total | 6961 | 65.395 | 100% | |

Long history, same method: attention 29.953 (42.2%), expert 17.858, loop 6.593, depth mix 3.738, convolution 4.310, norm/clip/other 4.384, head 4.100. Total 70.9 ms.

Regions 3 to 6 together take **5,036 of the 6,961 dispatches (72%) and 19.2 ms of isolated GPU time (29%)**. Almost all of them are kernels under 5 us.

## E. The overhead question

Short history, cool, the same conditions for all rows:

| quantity | value | mode |
|---|---:|---|
| (i) wall time per generated token | 52.1 to 53.8 ms | normal (`eval time`) |
| GPU busy span per token | 50.9 to 52.0 ms | mode 2 |
| compute entry to first GPU work | 0.8 ms | mode 2 |
| CPU encode time per token, all threads | 5.0 ms (main thread 0.48 ms) | mode 2 |
| (ii) sum of isolated kernel GPU time per token | 65.5 ms (65.5 to 76.4 over runs) | mode 1 |

**Verdict: the kernels themselves are the cost.** The GPU is busy for 98% of each token's wall time (50.9 of 52.1 ms). CPU-side encoding (5 ms) runs while the GPU works on the first command buffer and leaves no bubble in cool runs; the command buffers' summed GPU time equals their span. (ii) is larger than (i) because isolated kernels lose concurrent dispatch and run with the GPU idling between them. Normal mode recovers about 22%. So no CPU scheduling or encode gap is hiding time.

The GPU time is not mostly useful work, though:

- The weight bytes take 22.3 ms at the bandwidth limit. The weight-reading matmuls achieve only 49 to 88 GB/s in isolation: the IQ2_XXS expert gate_up is worst at 49 GB/s.
- The 5,568 tiny kernels (5 us or less, 17.2 ms isolated) do almost no work. On the GPU each one is a fixed launch-and-barrier cost of about 2 to 3 us that 6,961 dispatches pay serially wherever a barrier separates them.

Mode 1 cannot split the 51 ms normal-mode span between the two groups exactly. In isolation, matmuls are 68% (44.7 ms) and everything else is 32% (20.8 ms).

Long history, cool: wall 59.5 ms, GPU span 58.1 ms, CPU encode 5.3 ms. The same verdict holds.

## Changes most likely to speed up single-token decode

Each item names the measured time it targets (its ceiling) and an effort guess.

1. **Faster expert matmuls (MUL_MAT_ID).** Target: 15.3 ms isolated per token (gate_up 10.7 ms at 49 GB/s, down 4.6 ms at 62 GB/s). At the 88 GB/s that the output head reaches, they would take about 8.7 ms, saving roughly 6 ms. Options: tune the IQ2_XXS `mul_mv_id` kernel for one token (IQ2_XXS decoding goes through a codebook and looks compute-bound on M4), or requantize gate_up to Q2_0 like the down projection. The requantize option is a quality trade. Effort: medium to hard.
2. **Collapse the small-kernel tail in regions 3 to 6.** Target: about 5,000 dispatches and 19.2 ms isolated (clip 2.75 ms over 1,120 dispatches, depth mix 3.8 ms over 915, loop controllers 6.6 ms over 1,334, convolution 4.3 ms over 1,208). Concrete steps:
   - Replace `clip()` (8 kernels) with one fused row-norm-clip kernel, removing about 1,000 dispatches.
   - Stop rebuilding the stacked block states with CONCAT chains in every `mix()` (572 CONCATs, plus REPEAT and CONT of the transposed states). Keep one preallocated block buffer, and broadcast the query.
   - Precompute `ggml_pad(attn_rel_bias)` at load time (48 PAD kernels per token).

   Graph-side changes are easy to medium. A fused Metal kernel is medium.
3. **Merge the projections that share an input.** Q, gate, rel, and sink all read `inputs[0]`: 10.8 ms isolated over 192 calls. On SWA layers, K and V share an input too: 3.6 ms over 96 calls. Concatenating the weights at load time into one wider matmul cuts the dispatches by 4x. Larger matmuls also run closer to bandwidth here (7.5 MB calls reach 71 to 78 GB/s against 88 GB/s for the head). Expect 2 to 3 ms. Effort: easy to medium. The pattern already exists for fused `wqkv` in other llama.cpp models.

For long history only, the bias path (gather, permute, mask add and copy) costs +1.4 ms at 7.6k tokens. Flash attention costs +4.8 ms, which is small next to the items above.

The largest real-world factor is outside the code: after a few minutes at full load, this fanless laptop throttles single-token decode from 52 ms to 110 to 170 ms. Less memory traffic per token (items 1 and 3) also delays throttling.

## Not verified

- The exact split of the normal-mode 51 ms between matmuls and small kernels, since mode 1 removes overlap. Getting it would need per-encoder counter sampling, which is not built.
- Flash attention cost split between the 8 full-attention visits and the 40 sliding-window visits.
- Small-kernel isolated times vary 30 to 50% with chip temperature. The cool profiles (`metal_profile_short.json`, `metal_profile_long.json`) are the ones compared.

## Files

- `patches/metal_profile.patch`: diff from 6196d98 to `metal-profile`.
- `results/metal_profile_short.json`: mode 1, short, cool. `results/metal_profile_long.json`: mode 1, 7,608-token history from the prompt cache, cool.
- `results/metal_profile_short_warm.json`, `results/metal_profile_long_hot.json`: mode 1 repeats, warm or hot.
- `results/metal_profile_mode2_{s,l}_m2_{1,2}.json`, `results/metal_profile_diag_{d1_l_m2_hot,d2_l_m2_cool,d4_s_m2_hot,d5_s_m2_cool}.json`: mode 2.
- `results/metal_profile_analyze.py`: all tables above (`mode1`, `mode2`, `regions`, `growth`).
- `results/long_prompt_21k.txt`: first 21,000 characters of `long_prompt_src.txt` (7,608 tokens).

## Exact commands

```sh
cd runtime/llama.cpp
git checkout -b metal-profile
uvx cmake -B build-prof -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
uvx cmake --build build-prof --config Release -j 8 --target llama-completion llama-bench
git diff 6196d9802 metal-profile > ../../patches/metal_profile.patch

M=../../models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf
L=../../results/long_prompt_21k.txt

# A: speed and text, profiling off (build/ and build-prof/ alternated, 40 s rests)
build/bin/llama-completion      -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -p "def quicksort(arr):"
build-prof/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -p "def quicksort(arr):"

# B: mode 1, short
GGML_METAL_PROFILE=1 GGML_METAL_PROFILE_NODES_MAX=14 GGML_METAL_PROFILE_OUT=metal_profile_short.json \
  build-prof/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -p "def quicksort(arr):"

# mode 2 (normal encode path plus timestamps), short and long
GGML_METAL_PROFILE=2 GGML_METAL_PROFILE_OUT=s_m2.json build-prof/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -p "def quicksort(arr):"
GGML_METAL_PROFILE=2 GGML_METAL_PROFILE_OUT=l_m2.json build-prof/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -f $L

# C: build the long-prompt state once (hot), then decode from it cool
GGML_METAL_PROFILE=2 GGML_METAL_PROFILE_OUT=d1.json build-prof/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -f $L --prompt-cache long.cache
# (rest 120 s)
GGML_METAL_PROFILE=2 GGML_METAL_PROFILE_OUT=d2.json build-prof/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -f $L --prompt-cache long.cache --prompt-cache-ro
GGML_METAL_PROFILE=1 GGML_METAL_PROFILE_NODES_MAX=14 GGML_METAL_PROFILE_OUT=metal_profile_long.json \
  build-prof/bin/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 40 -f $L --prompt-cache long.cache --prompt-cache-ro

# analysis
python3 -I results/metal_profile_analyze.py mode1   results/metal_profile_short.json
python3 -I results/metal_profile_analyze.py regions results/metal_profile_short.json
python3 -I results/metal_profile_analyze.py growth  results/metal_profile_short.json results/metal_profile_long.json
python3 -I results/metal_profile_analyze.py mode2   results/metal_profile_diag_d5_s_m2_cool.json
```

Every model run was checked with `pgrep` so that only one llama process ran at a time, with 40 to 120 s rests between runs.
