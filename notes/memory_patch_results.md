# Memory patch results (2026-11-21)

Plan: `memory_patch_plan.md`. Tool: `inkscope.md`. Everything below was measured with `llama-inkscope` and `llama-perplexity` from the `inkscope` branch of `runtime/llama-loops`, greedy decoding, at most 40 new tokens, no document in the prompt unless stated.

## Verdict

No configuration meets the "worth continuing" line. The best adapter for held questions (`results/patch/all_r64_l10_seq.lora.gguf`) answers 7 of 20 held questions (6 facts on at least one rewording, 1 fact on both) but changes 9 of 10 control answers and has mean KL 0.131 (limit 0.01). The only adapter inside the KL limit (`late_r4_l1_seq`, KL 0.0076) stores nothing (0 of 30 fit, 0 of 20 held) and still changes the wording of 5 control answers. The fit-question stop rule does not fire: the two rank-64 all-layer adapters score 30 of 30 fit questions. The control and KL stop rule fires: the controls changed and KL stayed above 0.05 after lambda was raised once (1 to 10), so the sweep stopped there.

## Sanity checks

| check | result |
|---|---|
| Pass-line sanity, with the document, no adapter (inkscope) | fit 30/30, held 20/20 (10 of 10 facts) |
| Pass-line sanity, no document, no adapter | fit 0/30, held 0/20, controls 10/10 correct |
| `result_norm` x dequantized `output.weight` (Q6_K), with the 30 softcap, vs the tool's top-5 | same top-5 ids, max logit difference 1.1e-6 (`scripts/inkscope_sanity.py`) |
| `wo_inp` mapping: `(y + conv4(y))/sqrt(5)` with `y = W_o wo_inp` vs captured `attn_out` | relative error 3.0e-4 (visit 20, layer 8) and 4.1e-4 (visit 47, layer 23) (`scripts/inkscope_wo_check.py`). `wo_inp` is `[3072, tokens]`; feature j is input column j of `attn_output.weight` (numpy `(2304, 3072)`). |
| All-zero rank-4 adapter on all 24 `attn_output` vs no adapter, 60 prompts | top-5 ids identical 60/60, top-5 probabilities identical (max diff 0), generations identical 60/60 |
| Worktree build vs stored stock baseline `eval/base_tune.bin`, no adapter | Mean KLD 0.000000, same top 100.000% (PPL ratio 1.0006); zero adapter: identical numbers |

## Configuration table (11 sweep configurations, all run)

Fit facts: at least 2 of 3 fit questions correct. Held facts: "any" = at least 1 of 2 rewordings correct, "all" = both. Controls unchanged: generated text identical to the no-adapter text. Controls correct: known answer appears (note: "main" appears inside "trunk-main", so a leaked `branch` fact counts as a correct control for `branch_c`; that inflates "controls correct" by 1 in the rank-64 rows). Lambda is relative: lambda' = lambda x trace(sum k k^T) / trace(C0).

| config | layers | r | lambda | scheme | fit q | fit facts | held q | held facts (any / all) | controls unchanged | controls correct | mean KL | same top % | adapter bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_r16_l0.1_oneshot | 0-23 | 16 | 0.1 | one-shot | 0/30 | 0 | 0/20 | 0 / 0 | 0 | 0 | 1.597 | 69.15 | 8261184 |
| all_r16_l0.1_seq | 0-23 | 16 | 0.1 | sequential | 1/30 | 0 | 0/20 | 0 / 0 | 0 | 3 | 0.236 | 89.64 | 8261184 |
| all_r16_l1_seq | 0-23 | 16 | 1 | sequential | 2/30 | 0 | 0/20 | 0 / 0 | 0 | 2 | 0.167 | 91.24 | 8261184 |
| all_r16_l10_seq | 0-23 | 16 | 10 | sequential | 3/30 | 0 | 0/20 | 0 / 0 | 1 | 4 | 0.080 | 93.73 | 8261184 |
| coda_r16_l1_seq | 18-23 | 16 | 1 | sequential | 0/30 | 0 | 0/20 | 0 / 0 | 1 | 3 | 1.060 | 79.77 | 2065440 |
| loop_r16_l1_seq | 6-17 | 16 | 1 | sequential | 0/30 | 0 | 0/20 | 0 / 0 | 2 | 4 | 0.0146 | 96.11 | 4130688 |
| late_r16_l1_seq | 16-19 | 16 | 1 | sequential | 0/30 | 0 | 0/20 | 0 / 0 | 2 | 9 | 0.0127 | 96.93 | 1377024 |
| late_r4_l1_seq | 16-19 | 4 | 1 | sequential | 0/30 | 0 | 0/20 | 0 / 0 | 5 | 10 | 0.0076 | 97.88 | 344832 |
| late_r64_l1_seq | 16-19 | 64 | 1 | sequential | 0/30 | 0 | 0/20 | 0 / 0 | 2 | 8 | 0.0231 | 95.85 | 5505792 |
| all_r64_l1_seq | 0-23 | 64 | 1 | sequential | 30/30 | 10 | 6/20 | 5 / 1 | 0 | 1 | 0.209 | 90.52 | 33033792 |
| all_r64_l10_seq | 0-23 | 64 | 10 | sequential | 30/30 | 10 | 7/20 | 6 / 1 | 1 | 3 | 0.131 | 92.45 | 33033792 |

Late subset 16-19: the last two looped layers plus the first two coda layers. Reason: the one-shot held-out diagnostic below shows layers 20-23 make held-out targets worse (residual above 1), so the latest layers that still generalize at all are 16-19, and they sit where the answer token is being formed.

One-shot vs sequential (same layers, rank, lambda): they differ a lot. One-shot breaks the model (KL 1.60, 0 controls correct, outputs are fragments of document tokens such as "rine", "phy", "quay", "Glass"); sequential gives KL 0.24 and fluent text. Sequential re-capture is necessary.

Fit questions are in-sample: the adapter is fitted on exactly those 30 prompts (with the with-document model's own greedy answers teacher-forced, all 30 of which contain the fact), so 30/30 shows capacity, and only the held column measures recall.

## Five-fact specificity check (r64, lambda 10, sequential, all 24 layers)

Fitted on bridge, author, founder, testcmd, branch only (`five_r64_l10_seq`).

| subset | fit q | held q | held facts (any / all) |
|---|---|---|---|
| 5 fitted facts | 15/15 | 7/10 | 4 / 3 |
| 5 other facts | 0/15 | 0/10 | 0 / 0 |

Controls unchanged 0/10, mean KL 0.166. The other 5 facts stay wrong, so the patch stores the specific fitted facts and does not make the model answer every question correctly. It does leak: those questions now get "not mentioned in the provided notes" answers (the model acts as if it had notes), and fitted answers show up on unrelated questions ("Pride and Prejudice ... written by Ines Markovic", "Microsoft ... Odette Farrow", "Golden Gate Bridge ... 1987").

## Per-layer fit residuals

Relative residual ||K dW^T - T||_F / ||T||_F on the fitting rows, full-rank solve / rank-r solve, and the relative change the rank-r patch makes on ordinary-text keys, sqrt(tr(dW C0 dW^T) / tr(W C0 W^T)).

all_r64_l10_seq (best held): L0 0.04/0.26/0.008, L1 0.06/0.33/0.011, L2 0.07/0.36/0.016, L3 0.09/0.34/0.014, L4 0.10/0.32/0.011, L5 0.09/0.36/0.026, L6 0.38/0.53/0.059, L7 0.46/0.63/0.066, L8 0.50/0.65/0.068, L9 0.49/0.64/0.070, L10 0.46/0.59/0.063, L11 0.40/0.52/0.081, L12 0.46/0.59/0.050, L13 0.52/0.65/0.047, L14 0.51/0.66/0.052, L15 0.55/0.70/0.047, L16 0.48/0.62/0.039, L17 0.38/0.51/0.060, L18 0.16/0.53/0.021, L19 0.14/0.52/0.023, L20 0.16/0.55/0.023, L21 0.19/0.45/0.025, L22 0.18/0.40/0.023, L23 0.14/0.34/0.041

all_r16_l1_seq: L0 0.01/0.48/0.008, L1 0.02/0.51/0.010, L2 0.02/0.57/0.015, L3 0.03/0.56/0.014, L4 0.04/0.54/0.012, L5 0.02/0.60/0.023, L6 0.20/0.66/0.074, L7 0.24/0.75/0.085, L8 0.28/0.75/0.100, L9 0.27/0.74/0.109, L10 0.27/0.70/0.086, L11 0.23/0.62/0.118, L12 0.27/0.70/0.071, L13 0.31/0.76/0.065, L14 0.29/0.77/0.075, L15 0.32/0.80/0.074, L16 0.27/0.73/0.057, L17 0.21/0.64/0.080, L18 0.03/0.73/0.021, L19 0.03/0.74/0.023, L20 0.04/0.75/0.027, L21 0.05/0.64/0.034, L22 0.06/0.60/0.032, L23 0.05/0.61/0.052

Every configuration's per-layer numbers are in `results/patch/<config>.lora.gguf.fit.json` (also `sv_kept_frac`, the share of the weighted solution kept by rank r: 0.42 to 0.77 at r16, 0.69 to 0.94 at r64, 0.24 to 0.34 at r4).

Held-out linear-fit diagnostic (`scripts/mp_diag.py`, one-shot captures; fit on fit questions 0 and 1 of every fact, test on question 2; `results/patch/diag_linear_fit.json`). Held-out relative residual, rank 16 / full rank, lambda 1:

| layers | rank 16 | full rank |
|---|---|---|
| 0-5 | 0.63 to 0.83 | 0.68 to 1.06 |
| 6-17 (looped) | 0.76 to 0.84 | 0.66 to 0.78 |
| 18-19 | 0.79 to 0.83 | 0.90 to 0.97 |
| 20-23 | 0.94 to 2.26 | 1.21 to 2.68 |

The full-rank solve interpolates the fitting rows at non-looped layers (N = 562 keys in 3072 dimensions, residual 0.00 to 0.03), so the in-sample residual says little; on a new wording the map explains at most about 37% of the target norm, and at layers 20-23 it points the wrong way.

## Diagnosis

1. Rank is the first limit. At r16 the patch keeps 42% to 77% of the weighted solution and recalls 0 to 3 of 30 in-sample questions; at r64 over all 24 layers it recalls 30 of 30. Small subsets (4 or 6 layers) at any rank recall nothing, so the fact is spread thinly over many layers.
2. Generalization is the second limit. Each fit question gives only about 19 aligned token positions, and the target t = W_o(a_ctx - k) differs per wording; a linear map from attention-output keys explains at most about a third of a new wording's target (diagnostic above). Held recall stops at 7 of 20.
3. Leakage is what fails the pass line. The regularizer C0 comes from `eval/tune.txt`, which is C++ source. It does not represent chat-format questions, so the patch is free to act on question keys in general, and the controls (chat questions of the same shape) receive the stored answers or the "read the notes" behaviour. KL on tune.txt stays 0.08 to 0.21 for every adapter that stores anything.
4. Sharing one patch across three loop passes costs fit: full-rank in-sample residual is 0.2 to 0.55 at looped layers against 0.01 to 0.19 at non-looped layers (lambda 1 to 10).
5. What the patch transfers first is the mode of reading a document (loop-only adapter: "The relevant fact: ...", repeating the question) and only with high rank the content.

## Not verified

- No repeat runs; greedy decoding is deterministic, so no noise estimate was made for the counts.
- One C0 chunk (842 tokens) was longer than the 512-token ubatch; only its second ubatch's keys entered C0 (7 chunks complete, 1 partial).
- Sequential order is by physical layer; pass-2 and pass-3 visits of an early looped layer are captured before later looped layers are patched, so the sequential scheme is approximate for the loop.
- KL is measured on tune.txt (source code) as the plan requires; it does not measure chat-question drift, which the controls show is large.

## Exact commands

```sh
LAB=/path/to/Inkling-Lab
cd $LAB
scripts/mp_py scripts/eval_patch.py --tag base --no-kl                 # no-adapter reference answers
scripts/mp_py scripts/eval_patch.py --tag doc --doc --no-kl            # with-document sanity
scripts/mp_py scripts/memory_patch.py zero --out results/patch/zero.lora.gguf
scripts/mp_py scripts/eval_patch.py --tag zero --lora results/patch/zero.lora.gguf --no-kl
scripts/ppl_lora.sh                                                    # KL of the build itself
scripts/mp_py scripts/memory_patch.py prep                             # greedy doc answers, token check, captures, C0
scripts/mp_py scripts/memory_patch.py fit --layers 0-23 --rank 16 --lam 0.1 --scheme oneshot --out results/patch/all_r16_l0.1_oneshot.lora.gguf
scripts/mp_py scripts/eval_patch.py --tag all_r16_l0.1_oneshot --lora results/patch/all_r16_l0.1_oneshot.lora.gguf
scripts/mp_run.sh all_r16_l1_seq   0-23  16 1   seq                    # fit + eval; same pattern for every row:
scripts/mp_run.sh all_r16_l0.1_seq 0-23  16 0.1 seq
scripts/mp_run.sh all_r16_l10_seq  0-23  16 10  seq
scripts/mp_run.sh coda_r16_l1_seq  18-23 16 1   seq
scripts/mp_run.sh loop_r16_l1_seq  6-17  16 1   seq
scripts/mp_run.sh late_r16_l1_seq  16-19 16 1   seq
scripts/mp_run.sh late_r64_l1_seq  16-19 64 1   seq
scripts/mp_run.sh late_r4_l1_seq   16-19 4  1   seq
scripts/mp_run.sh all_r64_l1_seq   0-23  64 1   seq
scripts/mp_run.sh all_r64_l10_seq  0-23  64 10  seq
scripts/mp_run.sh five_r64_l10_seq 0-23  64 10  seq bridge,author,founder,testcmd,branch
scripts/mp_py scripts/mp_diag.py 0-23                                  # held-out linear-fit diagnostic
python3 scripts/mp_table.py all_r64_l10_seq ...                        # table rows and residuals
```

Work files (captures, 1.8 GB) are in the session scratchpad under `mp/` and `eval/`; they are not needed to read these results.

## Round 2 (2026-11-21)

Contract: `memory_patch_plan.md`, section "Round 2". Only the protection statistics, the fitting wordings and the rank-cut weighting changed. The held questions, the controls and the pass line did not.

### Setup

- Frozen sets were written before any capture or fit by `scripts/r2_make_sets.py`. `eval/facts_fit10.json` has 10 fitting wordings per fact (the 3 originals plus 7 new, none equal to or a close paraphrase of a held wording). `eval/protect_chat.jsonl` has 300 chat questions in the same template with no document. They cover the fact topics with other entities and general questions, never the 10 controls or any fact entity. `eval/extra_controls.json` has 30 unrelated questions, used for nothing else.
- Fitting data: the with-document model's greedy answers to all 100 fitting wordings contained the fact (100/100). Token-level suffix check passed for all 100. That gives 2041 aligned positions per visit.
- C0: mean k k^T of `wo_inp` over the protection set. It covers question tokens, header tokens and the stock model's own greedy answer (up to 20 tokens), every position after the template prefix. That is 8593 positions per visit (25779 at looped layers, pooled over three visits). Every prompt was under the 512-token ubatch, so no chunk was cut.
- Fit: all 24 layers, sequential re-capture, rank 64. The rank cut is the truncated SVD of dW·C0^(1/2), mapped back with C0^(-1/2).
- Lambda normalization: lambda' = lambda × trace(S_K) / trace(C0). S_K = Σ k k^T is summed over the fitting positions (visits pooled) and C0 is a per-position mean. So lambda = 1 gives the protection term the same total trace as the fitting data.
- Scoring (`scripts/eval_r2.py`): a fact is recalled only if both held wordings are correct. A control counts on a whole-token match of the expected answer, so "trunk-main" no longer matches "main". Every control output was also read by hand, and the hand reading agreed with the automatic count. "Extra unchanged" means the greedy text is identical to the stock run.
- Stock reference (`r2_base`): 0 facts, controls 10/10, KL 0.000000.

### Configurations (2 of a planned 3 finished, plus 1 stopped)

| config | lambda | facts recalled (both held) | held q | facts with ≥1 held correct | controls correct | controls unchanged | extra unchanged | mean KL | same top % | adapter bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| r2_all_r64_l10 | 10 | 4 (bridge, testcmd, port, owner) | 13/20 | 9 | 5/10 | 1/10 | 5/30 | 0.208 | 88.46 | 33033792 |
| r2_all_r64_l100 | 100 | 1 (port) | 6/20 | 5 | 10/10 | 2/10 | 11/30 | 0.080 | 92.58 | 33033792 |
| r2_all_r64_l1000 | 1000 | stopped during fit (layer 11 of 24), not evaluated | | | | | | | | |

Partial r2_all_r64_l1000 residuals (full / rank-64, layers solved before the stop): L0 0.90/0.90, L1 0.92/0.92, L2 0.93/0.94, L3 0.95/0.95, L4 0.93/0.94, L5 0.91/0.92, L6 0.97/0.97, L7 0.98/0.98, L8 0.98/0.98, L9 0.98/0.98, L10 0.98/0.98, L11 0.97/0.97. At this strength the patch stores almost nothing.

What the lambda 10 failures look like (read from the outputs): fitted answers land on unrelated questions ("Who founded Microsoft?" → "Tomas Ulrich"; "Romeo and Juliet" author → "Tomas Ulrich"; "What is the default branch name for new GitHub repositories?" → "trunk-main"). Several extra questions get "not stated in the provided notes". With the chat C0, lambda 10 already leaves 0.30 to 0.66 full-rank residual on the fitting rows, against 0.04 to 0.55 in round 1.

### Per-layer diagnostic: where the patch output lands (no extra model time)

Per-token RMS of the patch output dW·k on protection keys and on fitting keys, from second moments: sqrt(tr(dW C0 dW^T)) and sqrt(tr(dW S_K dW^T)/N). "vs stock" divides by the same RMS of the stock W_o output on that set.

r2_all_r64_l10 (protect/fit ratio; protect vs stock; fit vs stock): L0 0.141; 0.052; 0.345 · L1 0.090; 0.044; 0.482 · L2 0.091; 0.046; 0.519 · L3 0.085; 0.034; 0.411 · L4 0.078; 0.026; 0.346 · L5 0.074; 0.057; 0.677 · L6 0.150; 0.076; 0.535 · L7 0.149; 0.075; 0.554 · L8 0.142; 0.073; 0.566 · L9 0.135; 0.076; 0.613 · L10 0.109; 0.065; 0.588 · L11 0.058; 0.076; 0.826 · L12 0.110; 0.050; 0.504 · L13 0.105; 0.046; 0.442 · L14 0.107; 0.049; 0.451 · L15 0.127; 0.045; 0.344 · L16 0.117; 0.038; 0.348 · L17 0.078; 0.060; 0.594 · L18 0.135; 0.038; 0.337 · L19 0.163; 0.045; 0.373 · L20 0.143; 0.044; 0.342 · L21 0.117; 0.046; 0.364 · L22 0.093; 0.043; 0.372 · L23 0.064; 0.078; 0.854

r2_all_r64_l100: L0 0.070; 0.017; 0.230 · L1 0.069; 0.023; 0.326 · L2 0.063; 0.024; 0.397 · L3 0.061; 0.019; 0.325 · L4 0.057; 0.016; 0.284 · L5 0.049; 0.030; 0.561 · L6 0.108; 0.028; 0.275 · L7 0.101; 0.029; 0.312 · L8 0.098; 0.030; 0.340 · L9 0.095; 0.032; 0.390 · L10 0.073; 0.030; 0.417 · L11 0.042; 0.035; 0.600 · L12 0.068; 0.025; 0.410 · L13 0.069; 0.025; 0.383 · L14 0.060; 0.024; 0.407 · L15 0.070; 0.020; 0.296 · L16 0.060; 0.018; 0.317 · L17 0.042; 0.025; 0.480 · L18 0.061; 0.020; 0.395 · L19 0.084; 0.021; 0.322 · L20 0.071; 0.021; 0.333 · L21 0.063; 0.022; 0.349 · L22 0.051; 0.020; 0.332 · L23 0.035; 0.034; 0.715

The patch moves protection outputs by only 2% to 8% of the stock output per layer, against 23% to 85% on fitting keys. Summed over 24 layers (72 visits counting the loop), that small per-layer leak is still enough to flip control answers at lambda 10 and to keep KL at 0.08 at lambda 100.

### Verdict

Stop rule met: no configuration reaches 3 of 10 facts recalled with at least 9 of 10 controls correct and KL 0.03 or less. Lambda 10 has the recall (4) but 5 controls and KL 0.208. Lambda 100 has the controls (10) but 1 fact and KL 0.080. Lambda 1000 stores nothing (residual 0.90 to 0.98). Recall falls faster than drift as lambda rises, and drift stays above 0.03 along the whole axis. **The attention-output site is closed for this file.** No round 3.

### Not verified

- Lambda 1000 was not evaluated; the coordinator stopped it after 12 of 24 layers.
- KL is measured on `eval/tune.txt` only.
- Fitting-set recall (in-sample) was not measured in round 2.

### Commands

```sh
python3 scripts/r2_make_sets.py
export MP_WORK=<scratchpad>/mp2
scripts/mp_py scripts/eval_r2.py --tag r2_base
scripts/mp_py scripts/memory_patch.py prep --fit-file eval/facts_fit10.json --c0-source chat   # C0 step finished with: memory_patch.py c0resume
scripts/r2_run.sh r2_all_r64_l10 10
scripts/r2_run.sh r2_all_r64_l100 100
python3 scripts/r2_table.py r2_all_r64_l10 r2_all_r64_l100
```
