# Memory patch: end state, recipe, pass line, and stop rules

Written 2026-11-21 before any run. Prior work is in `patch_prior_art.md`.

## End state

One command turns a list of facts into a standard llama.cpp adapter file for the 7.9 GB Inkling 2 Mini build. With the adapter loaded and no document in the prompt, the model answers reworded questions about those facts, and its answers to unrelated questions do not change.

## What is already known (not ours)

- Context can be converted to a weight patch without gradients: "Transmuting prompts into weights" (arXiv 2510.08734) and PatchKV (arXiv 2609.39329). Both patch the MLP and are lossy (47.6 patched against 79.0 with context for one fact type; up to 73% of full-context accuracy on SQuAD).
- This file's MLP (the specialists) is 2-bit and takes no adapters. Only attention-side matrices do.

## What would be new if it works

A closed-form patch fitted from with-document and without-document activations, placed only on attention output projections of a quantized looped model, exported as a standard adapter file. No prior work with that site, that model type, or that export was found in a limited search.

## Recipe (per layer l)

- Capture the input of the attention output projection for each query token, without the document (key k) and with it (a_ctx).
- Target t = W_o (a_ctx - k).
- Solve dW = (sum t k^T)(sum k k^T + lambda C0)^-1, where C0 is the key covariance over ordinary text.
- Reduce to rank r, write as adapter tensors lora_a and lora_b for `blk.l.attn_output.weight`.
- Solve layers in order, re-capturing the without-document keys after each installed layer.
- A patch on a looped layer applies to all three passes; pool keys from all three.

## Data

`eval/facts.json`: 10 made-up facts (5 general, 5 about a fictional code repository), each with 3 fitting questions, 2 held-out rewordings, and 1 unrelated control question with a known answer.

## Pass line

Measured with the adapter loaded and no document in the prompt:

1. Sanity: with the document in the prompt and no adapter, at least 9 of 10 facts answered correctly. Without both, at most 1 of 10.
2. Worth continuing: at least 3 of 10 facts correct on held-out rewordings, all 10 control answers unchanged, and mean KL divergence on `eval/tune.txt` at most 0.01.
3. Worth posting: at least 7 of 10 on held-out rewordings with the same control and KL limits.

## Stop rules

- If sanity fails, fix the prompt format before anything else.
- If the fitting questions themselves score under 5 of 10 with the adapter, the attention-side site cannot hold the facts. Stop and record it.
- If controls change or KL exceeds 0.05, raise lambda once; if still failing, stop.

## Main risk

Attention-output keys are dense averages with no nonlinearity between them, so a low-rank linear map may not keep facts apart and may leak onto unrelated prompts. Sharing one patch across three passes adds a second mismatch.

## Round 2 (written 2026-11-21, after round 1 failed, before any round 2 run)

Round 1 showed capacity (30 of 30 in-sample) and specific storage (5 fitted facts recalled, 5 unfitted not), and failed on specificity: unrelated questions got the stored answers, and drift was 13 times the limit.

### What changes, and why

1. Protection statistics (C0) come from chat-format questions with no document, in the same prompt template, covering the same topics as the facts with different entities. Round 1 used C++ source, which says nothing about chat questions.
2. Each fact gets about 10 fitting wordings (round 1: 3). The held wordings and the 10 controls stay frozen and are never used for fitting or for C0.
3. The rank cut happens in the C0-weighted space, so the kept directions are the ones that matter on protected inputs.
4. The regularization strength is swept upward (10, 100, 1000).

### Extra frozen check

30 new unrelated chat questions, written before any fit, never used for fitting or C0. Score: greedy answer text identical to stock.

### Scoring (stricter wording of the original line, no change in intent)

- A fact counts as recalled when both held wordings are correct.
- A control counts when the answer is correct by exact match of the expected answer, checked by reading the output.

### Pass line

- Worth continuing: at least 3 of 10 facts recalled, 10 of 10 controls correct, KL on the tuning text 0.01 or less.
- Worth posting: at least 7 of 10 facts recalled under the same conditions, and at least 27 of 30 of the extra questions unchanged.

### Stop rule

At most 8 configurations. If none reaches 3 of 10 facts recalled with at least 9 of 10 controls correct and KL 0.03 or less, the attention-output site is closed for this file. No round 3 on this site.
