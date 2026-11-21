# Inkling 2 Mini lab: measured findings

Machine: MacBook Air M4, 16 GB. Date: 2026-11-21.
File under test: `models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf` (7.90 GB, specialists at about 2 bits, attention at 8 bits).
Runtime: llama.cpp pull request 32695 at commit 6196d98, Metal build, in `runtime/llama.cpp`.
Every number below came from a script in `scripts/` or a run logged in `results/`.

## Weights (official checkpoint, read by partial requests)

- Every weight row is a unit-length vector (attention, routers, specialists).
- Attention matrices are not low rank: 90% of the energy needs 45 to 57% of full rank (`results/attn_spectra.json`).
- Layer 8 specialists share nothing: mean cosine between two specialists is 0.0008, participation ratio 128 of 128. Each specialist needs about 1,100 of 2,304 rank for 90% energy (`results/expert_structure_L8.json`).
- Router rows: 28 near-duplicate pairs (cosine above 0.8) in the whole model, 18 of them in layer 17.
- Dimension 2027 (Goodfire's outlier) shows in the weights: the layer 17 router reads it at 8.2 times the median column, layer 11 at 3.2 times, and the final norm scales it to 0.03 of the median.
- Token convolutions (kernel 4) sit before attention and before the specialists. In some layers most dimensions weight the previous token more than the current one (layer 20, specialist side: 1,475 of 2,304).
- Loop controllers put 18 to 27% of their input weight on the current stream in layer 6, and 31 to 35% in layer 17. The rest listens to the previous pass.

## Damage in the 2-bit file (layer 8)

- Attention at 8 bits: 0.5% relative error against the official weights.
- Specialists at 2 bits: 34% relative error, uniform. Per-specialist cosine 0.9396 to 0.9403; per-row cosine 0.935 to 0.944 (1st to 99th percentile).
- Attention at lower widths with a simple block quantizer: 6-bit 2.2%, 5-bit 4.6%, 4-bit 10%, 3-bit 23%.

## Speed on this Mac

- Generation, short context, warm: 12 to 15.7 tokens per second. First run after load: 7.4 to 8.2.
- Generation with 1,551 tokens of context: 8.9 tokens per second.
- Generation with the context window set to 32,768: 0.96 tokens per second (memory pressure; swap was in use).
- Prompt reading: 50 to 74 tokens per second at batch 16 to 512.
- CPU only, 4 threads: 4.1 tokens per second.
- One token step runs 12,306 graph operations: 4,470 are free reshapes and views, about 7,800 are GPU kernels, of which 1,025 are matrix multiplies, 96 are specialist multiplies, and 48 are attention.
- Reading the weights for one token (about 2.7 GB) would take about 22 ms at the memory limit. A warm token takes 64 to 83 ms. Reading is not the bottleneck.
- Skipping the output vocabulary stage (embeddings mode) changes speed by 4%.
- Turning off the Metal backend's operation fusion or concurrency changes speed by under 4%.
- Thermal drift on this fanless machine moves results by about 20% within one session.

## Long-context collapse and its cause (measured)

Generation speed with a short prompt, by context window and batch setting:

| Context window | Batch | Working buffer | History buffers | Tokens per second |
|---|---|---|---|---|
| 8,192 | 512 (default) | 980 MiB | 256 + 240 MiB | 17.5 |
| 32,768 | 512 (default) | 3,404 MiB | 1,024 + 240 MiB | 1.2 |
| 32,768 | 64 | 425 MiB | 1,024 + 200 MiB | 18.4 |
| 32,768 | 16 | not read | not read | 19.2 |
| 65,536 | 16 | 321 MiB | 2,048 + 200 MiB | 18.1 |

The working buffer grows with batch size times context window. At the default batch it pushes the 7.35 GiB model past this Mac's memory. Flags `-b 64 -ub 64` avoid it. Not yet tested with a real long prompt filling the window.

## Long prompt, real history (measured)

A 26,065-token source-code prompt in a 32,768 window at batch 64: read at 31.9 tokens per second (13.6 minutes), then generated at 7.8 tokens per second. Peak resident memory 8.4 GB. The default-batch run on the same prompt was not completed.

## Volume correction test (negative result)

Plan and predictions were written first, in `volume_test_plan.md`. Instrument: `scripts/ppl.sh`, paired against stored baseline probabilities, 12 chunks of 512 tokens, about 90 seconds per setting. Stock perplexity on the tuning split: 2.164.

| Global specialist scale | Perplexity change vs stock (8.0) | Same top word as stock |
|---|---|---|
| 7.0 (control) | +1.92% ± 0.36% | 96.0% |
| 8.0 through the override | 0.00% | 100% |
| 8.5 | -0.17% ± 0.24% | 97.6% |
| 9.0 | +0.41% ± 0.40% | 96.1% |
| 9.6 (predicted best) | +1.37% ± 0.57% | 94.7% |
| 10.5 | +3.76% ± 0.80% | 92.9% |

The best value is 8.0 within noise. The prediction (best near 9.6, 1 to 5% gain) was wrong. Reason found afterwards: a least-squares scale is already the error-minimizing volume. Turning the volume up restores the signal but amplifies the quantization noise by the same factor, so total error rises. The stop rule applies; per-layer scaling was not attempted.

## Metal profile (full tables in `metal_profile.md`)

- Cool machine, short history: 52 to 54 ms per generated token (about 19 tokens per second). The GPU is busy 98% of that time. CPU encoding (5 ms) overlaps GPU work and leaves no gap.
- 6,961 real GPU dispatches per token; 5,568 of them take 5 µs or less.
- Isolated GPU time by region: attention 23.8 ms, expert MLP 18.0 ms, loop messages and controllers 6.6 ms, depth mixing 3.8 ms, token convolutions 4.3 ms, norms and clips 4.4 ms, output head 4.4 ms.
- Weight matmuls reach 49 to 88 GB/s in isolation, below the roughly 120 GB/s memory limit.
- Long history (7,608 tokens), cool: 59.5 ms per token, 7 ms more than short history, all of it attention.
- CORRECTION: the earlier long-history figures (8.9 tokens per second at 1,551 tokens, 7.8 at 26,065) were mostly thermal throttling. Within one hot run, time per token drifted from 56 to 166 ms. A short-history run right after a 3-minute prefill measured 112 ms per token.
- CORRECTION: dispatch overhead between operations is small. The cost is inside the kernels.

## Exact graph speedups (stopped under the stop rule)

Three graph-level changes with bit-identical output cut dispatches per token from 6,961 to 6,068 and time per token by about 3.9% (2 to 4% given noise). Under the 8% stop line, so the branch stopped after step 3. Patch: `patches/inkling_fast.patch`; details: `runtime_speed.md`.

## Fewer loops (measured on the tuning text, against the stock 3-loop output)

| Variant | Perplexity | Same top word | KL from stock |
|---|---|---|---|
| Three loops | 2.16 | 100% | 0 |
| Two loops | 4.12 | 74.1% | 0.62 |
| One loop | 4.94 | 69.1% | 0.79 |

The second and third loops use identical weights, so "skip the middle loop" and "skip the last loop" are the same computation. Dropped as a speed idea.

## Thinking effort zero

At effort 0 the model still opens a thinking block on 12 of 12 simple prompts (about 88 tokens). Forcing the answer channel in the prompt removes it on 12 of 12 with correct answers, so no weight edit is needed for that.

## Memory patch groundwork

- Facts test set (`eval/facts.json`): with the 10 statements in the prompt the model answers 10 of 10 held-out questions; without them 0 of 10.
- The runtime accepts adapter files on attention-side matrices, loop controllers, and the output head, but not on the specialists.

## Debugger core (built)

`llama-inkscope` on branch `inkscope` of `runtime/llama-loops` (patch: `patches/inkscope.patch`, usage: `inkscope.md`). It loads the model once, runs a file of prompts, records any named internal state, and can run with an adapter loaded. Checks passed: logits rebuilt from the captured final state match the tool to 1.1e-6; a zero adapter changes nothing on 60 of 60 prompts.

## Memory patch, round 1 (fails the pass line)

Recipe: forward passes only, closed-form solve on the 24 attention output matrices, written as an adapter file. Full table: `memory_patch_results.md`. Best adapter: `results/patch/all_r64_l10_seq.lora.gguf` (33 MB, rank 64, all layers, sequential).

| Check | Result | Pass line |
|---|---|---|
| Fitting questions (in-sample) | 30 of 30 | capacity only |
| Reworded questions | 7 of 20; 6 of 10 facts on at least one wording, 1 of 10 on both | 3 of 10 facts |
| Unrelated questions still correct | 3 of 10 (1 of 10 text unchanged) | 10 of 10 |
| KL from stock on the tuning text | 0.131, same top word 92.5% | 0.01 or less |

Re-run independently on the stock build: drift 0.130836 and the reworded answers match the subagent's saved outputs character for character.

- The patch stores specific facts: fitted on 5 facts, those 5 score 15 of 15 and 7 of 10 reworded, the other 5 score 0 of 15 and 0 of 10.
- It fires on the question shape, not on the subject: "The Golden Gate Bridge was completed in 1987", "Pride and Prejudice was written by Ines Lynch".
- Rank 16 stores almost nothing (3 of 30 at best). Subsets of 4 to 12 layers store nothing at any rank.
- One-shot fitting breaks the model (KL 1.6); refitting layer by layer is needed (KL 0.24 at the same settings).
- Flaw in the recipe: the "leave this alone" statistics came from C++ source text, so nothing told the solve to leave chat questions alone. Rank cut was also done without weighting by those statistics.

## Memory patch, round 2 (fails; site closed for this file)

Changes from round 1: protection statistics from 300 chat questions, 10 fitting wordings per fact, rank cut in the protection-weighted space. Rules in `memory_patch_plan.md`, full write-up in `memory_patch_results.md`.

| Regularization | Reworded questions | Facts recalled on both wordings | Controls correct | Extra controls correct (stock: 27 of 30) | KL on tuning text |
|---|---|---|---|---|---|
| 10 | 13 of 20 | 4 of 10 | 5 of 10 | 18 of 30 | 0.208 |
| 100 | 6 of 20 | 1 of 10 | 10 of 10 | 29 of 30 | 0.080 |
| 1000 | stopped at layer 11 of 24; fit residual 0.90 to 0.98 | not evaluated | not evaluated | not evaluated | not evaluated |

- Recall and damage move together. No setting meets the pass line (3 facts, 10 controls, KL 0.01 or less), and the stop rule (KL 0.03 or less) is met by none.
- Per layer, the patch moves protected inputs by 2 to 8% of the stock output and fitting inputs by 23 to 85%. Summed over 24 layers and three passes, the small leak still changes unrelated answers.
- Stopped by hand after two configurations because the trend along the regularization axis already decided the verdict. Not verified: the 1000 setting's evaluation, and KL on chat text.

## Ideas retired by these measurements

- Closed-form memory patches on the attention output matrices of the 2-bit file: two rounds, both fail on specificity.

- Exact graph speedups as a release: 3.9%.
- Running the released model with fewer loops.

- Long-context attention as the main speed target: history costs only 7 ms per token when the machine is cool.

- Global volume correction of the specialists: tested, no gain.

- Factor or share the specialists to save memory: no shared structure exists.
- Window-aware context cache: the runtime already does it.
- Surgical repair of one damaged spot: the 2-bit damage is uniform.
- 4-bit attention for speed: speed is not limited by bytes read on this Mac.
- Two-level vocabulary head: the output stage is about 4% of a token.

## Open questions

- Which operation types hold the 40 to 60 ms per token that is neither reading nor arithmetic. Needs a per-operation timer in the Metal backend.
- Which specialists each loop pass picks, and how many distinct specialists a coding session touches. Needs router logging through the runtime's evaluation callback.
