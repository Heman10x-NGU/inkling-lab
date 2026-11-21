# Patch prior art: context to weights, attention-site edits, weight memory, concept prediction, debuggers, llama.cpp LoRA format

Research date: 21 November 2026. About 40 searches, fetches, and `gh` calls. No model runs.
Tags: VERIFIED means I read the abstract, the method text, or the source file at the URL given. UNVERIFIED means from memory or not checked against a primary source in this session.

## 0. Corrections to the brief

- ggml-org/llama.cpp issue #32503 is NOT about Inkling. It is about `llama-cvector-generator` on `qwen35moe` (Qwen3.6-35B-A3B). The only "inkling" string in the thread is a compiler log line building `parsers/inkling.cpp`. VERIFIED: https://github.com/ggml-org/llama.cpp/issues/32503
- Control vectors do nothing on Inkling 2 Mini today: neither upstream `src/models/inkling2.cpp` on master nor the local fork (`Inkling-Lab/runtime/llama-loops/src/models/inkling2.cpp`) calls `build_cvec`. So `--control-vector` loads but has no injection point. VERIFIED (grep of both files, upstream via `gh api repos/ggml-org/llama.cpp/contents/src/models/inkling2.cpp`).
- Looped layers share one `ggml_tensor` per physical layer: `layers[il] = layers[loop_layer(il)]` in `src/llama-model.cpp` (local fork, around line 2025). A LoRA on `blk.N.attn_output.weight` for N in 6..17 is therefore applied on all three passes with the same scale. You cannot give pass 2 a different patch than pass 1 through a LoRA. VERIFIED (local source).
- Released expert weights are NVFP4 (layers 0 to 19) and MXFP8 (layers 20 to 23); attention projections and controllers are in other formats. VERIFIED: https://vllm.ai/blog/2026-11-20-inkling2-mini . The 2-bit GGUF is a community re-quantization (Unsloth, AtomicChat). VERIFIED: https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF

## 1. Training-free or closed-form ways to turn context into weight updates

### 1.1 ROME (Meng et al., 2022, arXiv 2202.05262). VERIFIED https://arxiv.org/abs/2202.05262 , formulas from https://arxiv.org/html/2202.05262
- Site: MLP output projection `W_proj` of one middle layer, at the last subject token.
- Update: `W_hat = W + Lambda (C^{-1} k*)^T`, with `Lambda = (v* - W k*) / ((C^{-1} k*)^T k*)` and `C = K K^T` the uncentered covariance of keys (MLP hidden activations) over a large generic text sample (Wikipedia). C makes the edit change the output least on typical keys.
- Key `k*`: average over N random prefixes of the MLP key at the last subject token. Forward passes only.
- Value `v*`: found by GRADIENT optimization of the MLP output vector z to maximize `log P(o* | x_j + p)` plus a KL "essence" term on "p' is a" prompts. This is the only step that needs backprop, and the backprop is through the frozen network into a single activation vector, not into weights.
- Reported numbers: the extraction returned efficacy 99.8, paraphrase 88.1, neighborhood 24.2 for one table row; I could not confirm which model/metric variant that row is. Treat ROME CounterFact numbers as UNVERIFIED. Scale: one edit per update; sequential editing degrades (well documented, e.g. https://arxiv.org/html/2602.02543 VERIFIED abstract).
- Code: https://rome.baulab.info (VERIFIED link in abstract).

### 1.2 MEMIT (Meng et al., 2022, arXiv 2210.07229). VERIFIED abstract https://arxiv.org/abs/2210.07229
- Site: MLP output projections across several middle layers; thousands of edits (GPT-J 6B, GPT-NeoX 20B) per abstract.
- Formula (UNVERIFIED, from memory): `Delta = R K1^T (C0 + K1 K1^T)^{-1}`, where `K1` stacks new keys, `R` stacks residuals `v_i - W k_i` spread across layers, `C0 = lambda * E[k k^T]` is the key covariance of preserved knowledge. Values `v_i` again come from per-fact gradient optimization of a hidden vector.
- Code: https://memit.baulab.info (VERIFIED link).

### 1.3 AlphaEdit (Fang et al., 2024, arXiv 2410.02355). VERIFIED https://arxiv.org/abs/2410.02355
- Projects the MEMIT perturbation onto the null space of preserved-knowledge keys so preserved outputs are unchanged "theoretically"; +36.7% average over locate-then-edit methods; LLaMA3, GPT2-XL, GPT-J. Code: https://github.com/jianghoucheng/AlphaEdit
- Formula (UNVERIFIED): `Delta = R K1^T P (K_p K_p^T P + K1 K1^T P + I)^{-1}` with `P` the projector onto the null space of `K0 K0^T`.
- Backprop: still needed for value vectors.
- Successor critique: FisherEdit (arXiv 2610.25914) says AlphaEdit's robustness comes mostly from update regularization, not the null-space projection. VERIFIED snippet https://arxiv.org/html/2610.25914
- Norm-Anchor Scaling (arXiv 2602.02543): sequential locate-then-edit collapses through a norm feedback loop between solved value vectors and edited weights; fix is rescaling value vectors to the original norm. VERIFIED snippet https://arxiv.org/html/2602.02543
- "Edits suppress rather than overwrite" (arXiv 2605.28839, ACL Findings 2026): ROME/MEMIT inject an amplified signal that downstream attention over-attends to. VERIFIED snippet https://aclanthology.org/2026.findings-acl.543.pdf

### 1.4 Implicit rank-1 update from context (Dherin et al., Google, arXiv 2507.16003, July 2025). VERIFIED https://arxiv.org/abs/2507.16003 , formulas from https://arxiv.org/html/2507.16003
- Setting: a "contextual block" `T_W = M_W ∘ A`, where `A` is a contextual layer (self-attention) and `M_W(z) = f_theta(W z + b)` has first dense layer `W`.
- Theorem 2.3: for context C, a removed part Y, and query token x,
  `Delta_x^W(Y) = (W deltaA_x(Y)) A(C\Y, x)^T / ||A(C\Y, x)||^2`, with `deltaA_x(Y) = A(C, x) - A(C\Y, x)`.
  Then `T_W(C, x) = T_{W + Delta}(C\Y, x)` exactly, and Delta is the unique minimal-Frobenius-norm patch; it leaves W unchanged on directions orthogonal to `A(C\Y, x)`.
- With skip connections (Theorem C.2): an exact bias patch `Delta b = deltaA` plus the same rank-1 weight patch.
- Token-by-token dynamics (Prop. 2.6): consuming context tokens one at a time is a gradient step `W_{i+1} = W_i - h grad L_i(W_i)` with `h = 1/||A(x)||^2`, `L_i(W) = trace(Delta_i^T W)`.
- Limits: the patch is token-dependent (one per query position); experiments are toy (10-layer transformer, d_model 32, in-context linear regression).
- Backprop: none. Pure forward passes.

### 1.5 Follow-ups to Dherin (found through Semantic Scholar citations of 2507.16003)
- "Transmuting prompts into weights" (arXiv 2510.08734, Oct 2025). VERIFIED abstract + method https://arxiv.org/html/2510.08734
  - Thought vector: `delta(I) = (1/n) sum_i (A(C, x_i) - A(C\I, x_i))`.
  - Thought matrix: `Delta(I) = argmin_M sum_i ||M a_i - W delta_i||^2 = (sum_i W delta_i a_i^T) Z^{-1}`, `Z = sum_i a_i a_i^T` (ridge variant allowed), `a_i` = attention output without the prompt.
  - Site: MLP up/gate (Gemma 3 1B IT), optionally down projection and biases. No gradients.
  - Results (Gemma 3 1B): arithmetic 100 vs 100; French to English 97.4 with context vs 92.7 patched without context; country to capital 79.0 vs 47.6; synthetic key-value dictionaries k=8: 99, k=16: 98, k=32: 91 (patched, vs 100 in context); ParaDetox toxicity 6.7 (context) vs 6.2 (patched). No code URL found.
- "Equivalence of Context and Parameter Updates in Modern Transformer Blocks" (arXiv 2511.17864). Exact rank-1 MLP patches plus an RMSNorm scale patch for Gemma-style blocks; generalizes to gating, pre/post norm, MoE, parallel blocks via "input/output controllability". VERIFIED abstract.
- "A Simple Generalisation of the Implicit Dynamics of ICL" (arXiv 2512.11255): extends to all positions, any block, LayerNorm residual blocks; toy experiments. VERIFIED abstract.
- PatchKV (arXiv 2609.39329, Sep 2026). VERIFIED abstract + method https://arxiv.org/html/2609.39329 ; code https://github.com/cusasak/PatchKV
  - Training-free. Site: MLP down projections. Closed form ridge: `Delta W = T^T H' (H'^T H' + lambda I)^{-1}`, targets `t_i = W (h_i - h'_i) + (z_i - z'_i)` (teacher = full cache, student = compressed or empty cache), `lambda = lambda_0 ||H'^T H'||_F^2 / tr(H'^T H')`. Sufficient statistics `S_H = H'^T H'`, `S_T = T^T H'` accumulate.
  - Layers solved greedily in order, each with upstream patches installed.
  - Reference queries from the context itself: "repeat the previous context" and self-study answers generated by the full-context model.
  - Result: on SQuAD with zero KV cache left, recovers up to 73% of full-KV accuracy; cache-only baseline is near zero. Full-rank patch; rank-32 variant in ablation.
  - This is the closest existing "document into weights, no training" recipe. Its site (MLP down) maps to Inkling's expert weights, which you cannot edit.
- BaseRSI (arXiv 2611.24578, 18 Nov 2026): "ReGap" uses the per-block activation difference that a harness (prompt) makes, and "ReKey" writes it by regression into a few key-value neurons per block; forward passes only; VLM. VERIFIED abstract.
- "When Can Prefixes Compile LoRA?" (arXiv 2609.36766): theory at one frozen attention head; a fixed prefix can realize only some rank-one value updates; 18.4 to 74.2% of the adapter effect stays uncompiled at three GPT-2 heads. Relevant as a capacity warning for attention-side storage. VERIFIED abstract.
- TimeROME-DLM (arXiv 2606.12841): gradient-free closed-form low-rank residual edit memory (ridge over subject keys and target deltas) for diffusion LMs; unlearning on TOFU; 50 to 400 facts. VERIFIED abstract.

### 1.6 In-context editing (IKE, arXiv 2305.12740). VERIFIED abstract
- No weight change at all: demonstrations in the prompt. Competitive with gradient editors on GPT-J with fewer side effects; scales to OPT-175B. Code https://github.com/Zce1112zslx/IKE . RL-tuned successors exist (arXiv 2608.25100, VERIFIED abstract).

### 1.7 Prompt baking (arXiv 2409.13697). VERIFIED abstract
- Minimizes `KL(P_theta(.|u) || P_theta_u(.))` with gradient training (LoRA). Bakes CoT prompts, news headlines (knowledge), personas. Needs backprop. The "half-baked" early stop gives a continuous prompt strength.

### 1.8 Memory injection by adding vectors
- Memory injections (arXiv 2309.05605): add prompt-specific "memories" to attention-head outputs at inference to fix multi-hop failures, GPT-2. VERIFIED abstract.
- In-context vectors (arXiv 2311.06668): one forward pass on demonstrations gives a latent vector added at inference. VERIFIED abstract.
- Function vectors (arXiv 2310.15213): a few attention heads carry a task vector; summing their outputs triggers the task zero-shot. VERIFIED abstract.
- All forward-only; they store tasks or styles well and facts poorly (UNVERIFIED generalization).

### 1.9 Hypernetwork adapters (all need a trained hypernetwork; none exists for Inkling)
- Text-to-LoRA (arXiv 2506.06105, Sakana): task description to LoRA; trained on 9 LoRAs. Code https://github.com/SakanaAI/text-to-lora . VERIFIED.
- Doc-to-LoRA (arXiv 2602.15902): meta-learned approximate context distillation; near-perfect needle retrieval at 4x the native context. VERIFIED abstract.
- Internalizer (arXiv 2610.11715): portable hypernetwork, LoRAs for frozen 284B DeepSeek v4 Flash; 84.9% top-1 teacher-forced vs 63.4% base. VERIFIED abstract.
- CLIK (arXiv 2610.38356): context-conditioned LoRA generator, single forward pass, below reading the context directly. VERIFIED abstract.

### 1.10 Direct answer: a gradient-free VALUE vector for a ROME-style edit
Yes. The value is computed as an activation difference with and without the context:
- Dherin et al. 2025 (exact, per token): target output change `W (A(C,x) - A(C\Y,x))` on key `A(C\Y,x)`.
- Transmuting prompts into weights (2510.08734): token-independent version by least squares, `Delta = (sum_i W delta_i a_i^T)(sum_i a_i a_i^T)^{-1}`. Results above (Gemma 3 1B; facts lossy: 47.6 vs 79.0 on country to capital).
- PatchKV (2609.39329): same idea at MLP down projections with ridge and sequential layers; 73% of full-context SQuAD accuracy with no cache.
None of these place the patch on attention output projections, and none test a looped model or export a llama.cpp LoRA (UNVERIFIED absence; based on the abstracts and citation list I read).

## 2. Edits at attention or other non-MLP sites
- Abliteration (Arditi et al., arXiv 2406.11717). VERIFIED https://arxiv.org/html/2406.11717
  - `r = mu_harmful - nu_harmless` (difference in means at one layer/position); ablation `x' = x - r_hat r_hat^T x`; addition `x' = x + r`.
  - Weight orthogonalization `W_out' = W_out - r_hat r_hat^T W_out` applied to embedding, positional embedding, attention output, and MLP output matrices; claimed equivalent to directional ablation everywhere.
  - As a LoRA on `attn_output` (my derivation): `Delta W = -r_hat (r_hat^T W_o)` is rank 1: `lora_b = -r_hat` (d_model x 1), `lora_a = r_hat^T W_o` (1 x d_attn), alpha = rank = 1. On Inkling this removes r from attention writes only; MLP (expert) writes cannot be orthogonalized, so ablation is partial.
- Steering vector as a weight edit: a constant addition `+v` after `W_o` is a bias. llama.cpp LoRA has no bias, but a rank-1 LoRA `v (u^T)` with `u^T a ≈ 1` on the relevant keys approximates it (my derivation, UNVERIFIED in literature). "Weight Updates as Activation Shifts" (arXiv 2603.00425) proves a first-order equivalence between activation interventions and weight updates and finds post-block output the best site; its methods are trained. VERIFIED abstract.
- Transmuting prompts (2510.08734) states its thought vectors and matrices explain existing vector/matrix steering. VERIFIED abstract.
- ReFT / LoReFT (arXiv 2404.03592): trained interventions on hidden representations of a frozen model, 15x to 65x fewer parameters than LoRA. Formula (UNVERIFIED, from memory): `Phi(h) = h + R^T (W h + b - R h)`, R with orthonormal rows. Not expressible as a weight LoRA unless the site is right after a linear map. VERIFIED abstract.
- Attention-head interventions for tasks: LoFiT (arXiv 2406.01563, trained head offsets), JoLA (arXiv 2502.01179, trained head selection and additive/multiplicative edits). VERIFIED abstracts.
- PMET (arXiv 2308.08742): finds MHSA encodes general knowledge-extraction patterns and edits mainly FFN with attention-aware targets. VERIFIED abstract.
- Evidence that attention matters for memorized facts: output vector editing (arXiv 2606.18767) found ~14% of memorized sequences unreachable by MLP-only edits; ablating top heads recovers 60 to 64% of those. VERIFIED snippet https://arxiv.org/html/2606.18767v1
- Capacity theory: 2609.36766 shows limits of what an attention head can be made to compute through value-side changes. VERIFIED abstract.
- Net: I found no published closed-form fact store placed only on `W_o`. Theory (Dherin) says `W_o` is a valid site: it is the first dense layer reading `A(C,x)`.

## 3. Memory that changes weights during inference
| Method | Applies to a frozen pretrained model with forward passes only? | Notes (source) |
|---|---|---|
| Fast weights / linear attention as fast weights | No (architecture) | classic; UNVERIFIED citations |
| TTT layers (arXiv 2407.04620) | No: new layers trained from scratch, 125M to 1.3B | VERIFIED abstract |
| Titans (arXiv 2501.00663) | No: neural long-term memory trained in the architecture | VERIFIED abstract (training requirement from memory, UNVERIFIED) |
| In-Place TTT (arXiv 2604.06169) | Partly: MLP down proj is the fast weight, update `W += eta V_hat^T Z` per chunk, but `V_hat = Conv1D(X0) W_target` needs a trained projection; drop-in via continual training; Qwen3-4B RULER-128k 77.0 vs 74.8 | VERIFIED method |
| Select-TTT (arXiv 2610.36794) | Builds on In-Place TTT; relevance-weighted writes from frozen head | VERIFIED abstract |
| MoNe (arXiv 2608.17616) | Attaches to frozen model; memory learned by layer-local gradient updates | VERIFIED abstract |
| SEAL (arXiv 2506.10943) | No: self-edits trigger gradient SFT (LoRA), RL outer loop | VERIFIED abstract |
| PatchKV / Transmuting / BaseRSI | Yes: closed form from forward passes; sufficient statistics accumulate online | VERIFIED |
Conclusion: the only forward-only weight memories for a frozen model are the regression patches in section 1.5. A running `S_K = sum k k^T`, `S_T = sum t k^T` per site gives an online memory: re-solve and re-export the LoRA after each document.

## 4. Concept-level prediction
- Large Concept Models (Meta, arXiv 2412.08821). Trains a model that predicts the next SONAR sentence embedding (MSE, diffusion, quantized). 1.6B on ~1.3T tokens, scaled to 7B on ~2.7T tokens. Nothing useful on a frozen LLM without the SONAR encoder/decoder and pretraining. VERIFIED abstract.
- Next Concept Prediction / ConceptLM (arXiv 2602.08984). VQ over hidden states builds a concept vocabulary; NCP + NTP losses; 70M to 1.5B from scratch, up to 300B tokens; continual pretraining of Llama 8B also helps. A frozen-backbone variant is not reported (fetch found none). A small VQ codebook plus head trained on frozen Inkling activations is a probe, not NCP. VERIFIED abstract.
- Next-Concept Modeling (Stanford, arXiv 2610.14162). Multiscale residual VQ-VAE over text plus a transformer that generates one code level at a time in parallel; small-scale (LM1B, OpenWebText, XSum, Sudoku), 10x to 31x faster than the strongest diffusion LM at matched quality. Compute details not extracted (UNVERIFIED). Trained from scratch; no frozen-LLM use. VERIFIED abstract.
- What is cheap on a frozen model: tuned lens / coda lens (Goodfire found tuned-lens and coda-lens decode Inkling well, logit lens poorly), and a small multi-token or concept probe on captured activations. VERIFIED https://goodfire.com/research/interpreting-inkling-2-mini

## 5. Debuggers and where a local one fits
- NDIF + nnsight: hosts `thinkingmachines/inkling-2-mini` and `TML-d24-varloop-research` for remote readout, patching, steering, and gradients; nnsight 0.8.3; `for _ in tracer.iter[:3]` for the loop; hook sites: active block (`layers[i].output`), mixed with completed blocks (`final_input`), MoE input. VERIFIED https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif
- Looped Lens Challenge. VERIFIED https://ndif.us/looped-lens-challenge/
  - Host: NDIF (page on ndif.us; "Organizers and partners" section did not extract names, so exact organizer list UNVERIFIED; Thinking Machines and Goodfire are partners in the launch posts).
  - Task: build the lens that best reads Inkling 2 Mini. Input: one prompt from the evaluation set. You choose N layers and one token position (probably the final token). Output: top K tokens at each of N read locations, labelled by layer. Readouts go into one list given to a judge.
  - Phases: several, increasing complexity; phase one is single-token readouts on simple tasks in the style of workspace-bench (https://github.com/camila-blank/workspace-bench).
  - Rules: write the experiment locally with nnsight; NDIF runs it. Submission is a size-limited Python script plus lens artifacts as files or links. Needs an NDIF key and registration (Google Form).
  - Dates: phase one opens 23 November 2026; closing date "will be announced". Prizes: none stated.
- Goodfire: research post only (information flow, outlier coordinate 2027, layers writing to "future" layers, lens comparison). No released tools, SAEs, or code found. VERIFIED https://goodfire.com/research/interpreting-inkling-2-mini
- TransformerLens for GGUF/llama.cpp: none found. llama.cpp offers the `cb_eval` graph callback (used by `llama-eval-callback` and `llama-cvector-generator`; the issue above quotes it) and named nodes such as `l_out`, `loop_q_inp`, `loop_ffn_inp` in the Inkling graph (local source).
- llama-cvector-generator bug (#32503, qwen35moe): (1) the writer names capture k as `direction.(k+1)`, so every direction is applied one block deeper than measured (also in `pca.hpp`); (2) the last block's `l_out` is cropped to `[n_embd, n_outputs]` and the `cb_eval` filter `t->ne[1] != n_tokens` drops it, so the deepest depth is never measured; a short file silently zero-fills. Proposed fix: accept the narrowed node, map capture N to `direction.N`, assert counts. Open, filed 17 Nov 2026. VERIFIED.
- Laptop-local inspector for Inkling 2 Mini: none found. Remote white-box access exists (NDIF); a local GGUF/llama.cpp inspector that understands blocks, passes, and messages does not appear to exist (UNVERIFIED absence, based on these searches).

## 6. llama.cpp LoRA adapter format (from source)
Sources: `convert_lora_to_gguf.py`, `src/llama-adapter.cpp`, `src/llama-adapter.h`, `src/llama-graph.cpp` on ggml-org/llama.cpp master (fetched with `gh api`, 21 Nov 2026), plus local `gguf-py`. VERIFIED.
- Required metadata: `general.type = "adapter"` (else error), `general.architecture` equal to the base model arch (`"inkling2"`), `adapter.type = "lora"`. Optional `adapter.lora.alpha` (float32; missing reads as 0). Optional `adapter.alora.invocation_tokens` (uint32 array).
- Tensor names: `<base tensor name>.lora_a` and `<base tensor name>.lora_b`, e.g. `blk.20.attn_output.weight.lora_a`. Any other suffix throws, except `*_norm.weight`, which is skipped. Both halves must exist; the base tensor must exist in the model.
- Shapes (ggml `ne` order; numpy shape is reversed): base `ne = [n_in, n_out]`; `lora_a ne = [n_in, r]` (numpy `(r, n_in)`); `lora_b ne = [r, n_out]` (numpy `(n_out, r)`). Checks: `base.ne[0] == a.ne[0]`, `base.ne[1] == b.ne[1]`, `a.ne[1] == b.ne[0]`. `token_embd` is the exception (A transposed).
- Forward: `y = W x + scale * B (A x)`, i.e. numpy `Delta W = scale * B @ A` with shape `(n_out, n_in)`.
- Scale: `scale = adapter_scale * alpha / r` if alpha != 0, else `adapter_scale`; rank r is read from `b.ne[0]`; `adapter_scale` is the `--lora-scaled` value (1.0 for `--lora`).
- Types: tensors are copied as stored. The converter emits f32, f16, bf16, or q8_0. f32 is safest.
- Subsets: yes. The adapter holds only the tensors you list; `get_weight` returns null for all others.
- Restrictions: adapters on repacked CPU buffer types fall back to CPU; adapters on Hadamard-folded latent lookup tables (`token_embd`, tied head) are refused; folded matmul weights are fine because the LoRA reads the primal input.
- Inkling note: one LoRA tensor on a core layer acts on all three passes.

Minimal rank-1 writer (gguf-py; `GGUFWriter.__init__` already writes `general.architecture`):
```python
import numpy as np
import gguf

name = "blk.20.attn_output.weight"   # base ne = [d_in, d_out]
u = np.load("u.npy").astype(np.float32)   # shape (d_out,)
v = np.load("v.npy").astype(np.float32)   # shape (d_in,)  -> Delta W = outer(u, v)

w = gguf.GGUFWriter("fact.lora.gguf", arch="inkling2")
w.add_type(gguf.GGUFType.ADAPTER)
w.add_string(gguf.Keys.Adapter.TYPE, "lora")
w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, 1.0)   # alpha / r = 1 for r = 1
w.add_tensor(name + ".lora_a", v[None, :])          # numpy (1, d_in)  = ne [d_in, 1]
w.add_tensor(name + ".lora_b", u[:, None])          # numpy (d_out, 1) = ne [1, d_out]
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
```
Run with `llama-cli -m model.gguf --lora fact.lora.gguf` or `--lora-scaled fact.lora.gguf 0.5`.

## 7. Recommended first recipe: document to rank-r LoRA on `attn_output` (forward passes only)
Basis: Dherin Theorem 2.3 with `W = W_o`, which is the first dense layer reading the attention mixture `A(C,x)` (in Inkling the input to `wo` is the gated head output `cur * sigmoid(gate(x))`); least-squares token independence from 2510.08734; ridge, self-generated reference queries, and greedy layer order from PatchKV.
1. Reference queries `Q_j`: questions and paraphrases about the document, "repeat the context" spans, and self-study answers from the with-document model (PatchKV).
2. Pass A (teacher): `[doc ; Q_j]`. Capture at each chosen layer l and each query token i: `a_i^ctx` = input of `blk.l.attn_output` (all three passes for core layers), plus final logits.
3. Pass B (student): `[Q_j]` alone, with patches for layers < l installed. Capture `a_i` at the same tokens.
4. Targets `t_i = W_o (a_i^ctx - a_i)`; keys `k_i = a_i`. For core layers stack all three passes into one regression.
5. Solve `Delta W_o = (sum_i t_i k_i^T)(sum_i k_i k_i^T + lambda C0)^{-1}`, `C0 = E[k k^T]` over generic text at that site (MEMIT-style specificity prior), lambda scaled as in PatchKV.
6. Rank r: weighted truncated SVD of `Delta W_o (S_K + lambda C0)^{1/2}`, map back; `lora_b = U_r S_r`, `lora_a = V_r^T (S_K + lambda C0)^{-1/2}`, alpha = r.
7. Greedy over layers (coda first or last few layers, then core), re-running pass B after each install. Evaluate held-out questions (efficacy and paraphrase) and unrelated prompts (KL to base, specificity).
Main risk: keys at `W_o` are dense, highly correlated attention averages, so a rank-r linear map fit to document queries both under-separates facts and leaks onto unrelated inputs that share key directions; sharing across three loop passes adds a second mismatch.
