# NCM, Goodfire neurons, and Inkling-2-Mini: notes and hypotheses

Written 21 Nov 2026. Sources read: the local NCM paper (`NCM/paper_2610.14162v1.md`, with Tables 1 to 12 and figure captions pulled from `NCM/paper_2610.14162v1.html` because the Markdown dropped them), `NCM/ncm_architecture_analysis.md`, `NCM/decision_models_comparison.md`, `NCM/TEMP_HANDOFF.md`, the first lines of each file in `NCM/related_papers/parsed_txt/`, and the README plus `models/next_concept.py` and `models/quant.py` of `NCM/hazy_ncm_repo/`. Web: Goodfire "Neurons as an Interpretable Basis" (10 Nov 2026), "Neural Networks Are Already Clean" (essay, already-clean.goodfire.com), and "Interpreting Inkling 2 Mini" (20 Nov 2026).

Anything marked INFERRED is my reasoning, not a stated result. Goodfire details marked (extract) came from a prompted extraction pass over the long page rather than my reading the raw text line by line; treat them as slightly less certain.

---

## Part 1. Next-Concept Modeling (arXiv 2610.14162, Stanford Hazy Research)

Authors: Kumbong, Khare, Mlauzi, Woen, Ding, Buchanan, Arora, Mirhoseini, Ré. Code: github.com/HazyResearch/ncm (MIT). Checkpoints on Hugging Face `hazyresearch/ncm-checkpoints`.

### What a "concept" is

The paper defines it narrowly: "the term concept here refers to the information carried by the coarse levels of the hierarchy." A concept is a short sequence of discrete codebook indices at an early level of a multiscale residual VQ-VAE (vector-quantized variational autoencoder: a model that maps continuous vectors to the nearest entry of a learned table, the codebook). It is a learned latent, not a sentence embedding and not a word.

### How text is encoded (stage 1, the tokenizer)

1. Text `x` of fixed length `L` (in model tokens: GPT-2, BERT, Qwen3 or custom BPE vocabularies, depending on dataset) is embedded into `z` of shape `L x d`, with `d = 384` (256 for Sudoku). Table 5 lists "Encoder blocks: 0" for every dataset, so the encoder is the embedding table (the local analysis says token plus position embedding then a parameter-free GroupNorm; I did not verify that last detail in code).
2. `K` levels of residual quantization. Level `k` has `l_k` codes with `l_1 < l_2 < ... < l_K = L`. At each level the remaining residual is downsampled to `l_k` vectors (learned strided convolution at level 1, area interpolation later, per the repo), each vector is snapped to its nearest entry in that level's codebook (size 4,096; 8,192 for LM1B; 16 for Sudoku), the chosen vectors are mapped back to length `L` by a reconstruction operator `R_k` (upsampling plus a small residual convolution, `Phi` in `quant.py`), added to the cumulative reconstruction `z_hat(k)`, and subtracted from the residual.
3. Nested dropout: during training a cutoff `k` is sampled and only the first `k` levels are kept, so the decoder must reconstruct text from any prefix of levels. This forces early levels to carry global information.
4. Decoder: 1 transformer block (4 on LM1B), 6 heads, maps `z_hat(K)` to token distributions at all `L` positions.
5. Loss: reconstruction cross-entropy plus VQ commitment and quantization MSE terms, evaluated at the sampled cutoff; straight-through gradients; EMA codebooks.

Reconstruction accuracy with the full hierarchy (Table 1): OWT 99.99%, LM1B 99.86%, TinyStories 99.98%, XSum 99.79%, WMT14 98.79%, Sudoku 100%. Codebook utilization 62% to 90%. Failure cases (Table 11): named entities (Sydney to Melbourne), numerals (1957 to 1956), diacritics, usernames, markup, word boundaries.

### Generation order: coarse to fine, parallel within a level

Generation is sequential across levels and parallel within a level. `K` forward passes total (8 to 32), each predicting all `l_k` codes of one level at once, conditioned on the cumulative reconstruction of earlier levels. Codes inside a level are sampled independently given that reconstruction.

### The generator (stage 2) and its exact components

- A standard transformer: self-attention plus feed-forward per block, RoPE, 12 blocks, width 768, 12 heads, head dimension 64 (Sudoku: 8 blocks, width 512). About 88M to 120M parameters for the main runs (Table 6).
- Input at level `k`: `z_hat(k-1)` resized to `l_k` positions (area interpolation) and projected to width 768; optional level embedding.
- Attention mask: block-diagonal. Each level's block attends only to itself (its input already summarizes earlier levels) and to the prompt. Table 9 on TinyStories: block-diagonal Gen. PPL 15.41 in 5.65 s, sliding window over 2 levels 17.84 in 6.83 s, block-causal 19.63 in 7.90 s.
- Conditional tasks use a prefix LM: prompt tokens, embedded with the frozen VQ-VAE token table at full resolution, attend to each other; target blocks attend to the prompt and themselves; the prompt never sees targets. A frozen T5-small encoder gave only +0.10 / +0.15 / +0.24 ROUGE-1/2/L at +35.3M parameters, so they dropped it.
- Output head (repo, `_TiedCodebookHead`): logits are a scaled negative squared distance to the level's codebook entries, `2 z.E_i - |E_i|^2`, the inverse of the tokenizer's nearest-neighbor rule, with a learned temperature.

### Training objective

Cross-entropy on all codes of all levels in one forward pass (teacher forcing), with a per-level weight `w_k proportional to l_k^(-alpha)`. `alpha = 0` weights every code equally and equals maximizing an ELBO (evidence lower bound) whose variational posterior is the tokenizer's deterministic code assignment (Appendix A). `alpha` from 0 to 1.5 moved coarse levels' loss share from 19% to 63% but Gen. PPL stayed 18.9 to 19.0 for `alpha <= 1` and rose to 19.4 at 1.5.

Conditioning corruption: a linearly decreasing fraction of input codes per level is replaced with random codes from the same codebook while targets stay clean, so the generator learns to correct its own earlier parallel-sampling errors. Moderate corruption "greatly improves" quality, too much hurts (Figure 13). The paper suggests corruption from the model's own errors as future work.

### Decoder from concepts to tokens

After level `K`, the frozen VQ-VAE decoder maps `z_hat(K)` to token distributions and Algorithm 2 takes the argmax per position. The local analysis notes that released LM1B and XSum eval configs add a left-to-right post-processor with frequency and adjacency penalties to suppress stutter; I did not re-verify this in the repo.

### Inference, step by step (Algorithm 2)

1. `z_hat(0) = 0`; encode the prompt, if any, at full resolution.
2. For `k = 1..K`: resize `z_hat(k-1)` to `l_k` positions; project to width 768; one generator forward pass over prompt plus this block; sample all `l_k` codes in parallel; look up codebook vectors, map to length `L` with `R_k`, add to `z_hat`.
3. Decode `z_hat(K)` with the tokenizer decoder; argmax per position.

Early levels process few positions (level 1 can be a single code), so the first steps are cheap. Table 12 shows a translation going from "However, it is a important to the the the and the." (BLEU 6.0 at level 1) to the final output (BLEU 77.8 at level 11); quality is not monotonic (level 8 BLEU 51.7, level 9 38.9).

### Measured results

All generators are about 100M parameters unless noted; baselines are discrete and continuous diffusion and flow models (MDLM, DUO, SEDD, Eso-LM, RDLM, CANDI, FLM, S-FLM, FMLM+, ELF, AURORA-LM, E2D2, SeqDiffuSeq) plus small AR models.

Unconditional (Table 8, Gen. PPL under GPT-2-large, entropy H):
- TinyStories: NCM 15.41 (H 3.98) at 12 NFE vs FMLM+ 15.43 at 128, MDLM 17.89 at 128, DUO 19.63 at 128, AR sampling 8.89 at 128; data 9.67.
- LM1B: NCM 93.05 (H 4.31) at 12 NFE vs DUO 97.30 at 128, FLM 112.33, MDLM 115.45; data 53.51. Under four other scorers (Table 7) NCM averages 83.24 vs DUO 92.52 and MDLM 114.31, at 97M parameters and 13.36B training positions vs 139M and 66B.
- OpenWebText (L = 1,024, K = 32): NCM 83.12 (H 5.42) at 32 NFE vs DUO 77.69 at 1,024, FMLM+ 66.60 at 128 (lower entropy 5.21), MDLM 105.15, AR sampling 35.45; data 14.15. NCM is "competitive", not better, here.

Conditional (Table 2):
- XSum: NCM (88M) 35.0 / 12.8 / 28.0 ROUGE-1/2/L at 8 NFE vs AURORA-LM 34.7 / 12.2 / 26.6 at 128 and AR 30.5 / 10.2 / 24.4 at 64. With MBR over 16 candidates (128 NFE): 37.3 / 14.0 / 29.6 vs guided AURORA-LM 36.3 / 13.4 / 28.6 at 184.
- WMT14 De-En: NCM (120M) 22.6 BLEU at 11 NFE; CFG 23.9 at 22; MBR 25.2 at 132. AR 25.2 at 64, E2D2 24.8 at 64, guided ELF 26.4 at 128. NCM trails E2D2 unguided.

Reasoning (Table 3, Sudoku solved boards Easy / Medium / Hard): NCM (25M, 12 NFE) 99.5 / 96.5 / 86.1 vs FMLM+ (29M, 16 NFE) 97.7 / 92.8 / 81.3, DUO 96.3 / 84.7 / 58.4 at 128, AR greedy 14.6 / 5.1 / 1.0.

Speed (Section 4.6): end-to-end wall clock for 1,024 sequences at batch 16 on one H100 SXM. LM1B: 11x fewer NFE than DUO and 21x faster at better Gen. PPL. XSum: 16x fewer NFE than AURORA-LM and 31x faster at comparable ROUGE. OWT: 10x faster than the strongest DLM. README states 9 to 31x. Speedups come from fewer steps and cheaper early steps. Note all comparisons are against diffusion models; there is no wall-clock comparison against a KV-cached AR model of the same quality.

Training budget: LM1B 13.36B positions (6.80B tokenizer plus 6.55B generator), 4.9x fewer than 66B; OWT 127.22B, about 4x fewer than 500B-plus.

Scaling: generators of 97M, 317M, 725M on LM1B with the same tokenizer improve Gen. PPL and entropy at fixed tokens (Figure 6a, no numbers in text). Number of levels: K in {4, 8, 16, 32} on TinyStories, Gen. PPL improves with more levels with diminishing returns; all reconstruct at least 99.95% of tokens. MBR on XSum gains 4.48 ROUGE-1 from 1 to 256 candidates.

Probing (Appendix H): fixing levels 1 to 3 of a 12-level TinyStories hierarchy and resampling the rest keeps the story's theme while changing names and wording; swapping two level-1 codes changes characters and events.

### Stated limitations (Appendix I)

- Fixed depth `K` and fixed lengths `l_k`: even short outputs traverse the whole hierarchy; an input-dependent schedule would adapt compute.
- Separate, frozen tokenizer decouples representation learning from generation; end-to-end training might help.
- Main experiments at about 100M parameters; scaling only up to 725M on LM1B.
- Occasional artifacts at coarse-level transitions.

Further limits the local notes flag (INFERRED from the paper's setup, not stated as limitations by the authors): fixed output canvas of length `L` (no streaming of first tokens); VQ reconstruction errors on numerals and entities; no warm start from AR checkpoints; total positions processed equal the sum of `l_k`, which exceeds `L`, so throughput at large batch may lose to KV-cached AR.

---

## Part 2. Goodfire's neuron work

### "Neural Networks Are Already Clean" (Nick Cammarata, essay)

Central claim: the evidence that convinced the field of polysemanticity and superposition was misread. What looked like arbitrary packing of unrelated concepts into one neuron was mostly "recursive generalization": a neuron's job branches into a few related offshoots as it generalizes (an "iy" detector grows into "diy" and then "do it yourself"). So you can study the network's own neurons directly, without first training a sparse dictionary.

Five candidate causes of apparent polysemanticity: bad tools, bad explanations, bad learning (architecture or data), recursive generalization, and true superposition. He argues the first four explain most cases.

Evidence:
- Vision: InceptionV1's famous "car into dogs" circuit is shared pose, not packing; InceptionV1 has more than a hundred generalized pose neurons. ConvNeXt variants trained across a 300x size range (92k, 7.06M, 28.6M parameters, per extract) with ordinary fixes (residual connections, LayerNorm, regularization) have 99 to 100% of last-layer neurons easily interpretable; with less capacity neurons cover coarser concepts rather than packing unrelated ones.
- Language: about 90% of neurons have basically one job, the rest usually two or three related ones; he has "never seen a neuron with something like ten unrelated things."
- Negative poles matter: removing the 8 neurons whose negative side encodes Santa drops Santa from near-certain to 6%; removing the 20 neurons labeled Christmas or Santa only drops it to 76%.
- SwiGLU "double negative": when gate and value are both negative the output lands on the positive pole and looks like the negative pole. For about 4% of neurons this echo dominates the top of the activation list. Ablating these across the network raises most benchmarks but lowers some reasoning ones.
- In SwiGLU vision neurons the value tends to learn broad families and the gate opens for specific members.
- Experts are far more specialized than previously reported and get narrower with depth. He calls MoE an "anti-superposition architecture": the router is one small linear layer, so an expert that packed ten thousand unrelated things would be unroutable.

### "Neurons as an Interpretable Basis" (Tigges, Kowal, McGrath, Lewis, Cammarata, 10 Nov 2026)

Model: Qwen3.6-35B-A3B (MoE; 8 routed experts per token, 512 neurons per expert, so 4,096 routed neurons per token per layer; 5.26M neurons total; shared-expert neurons, 20,480 or about 0.4%, are always left untouched).

Definitions:
- Neuron: one hidden unit of a SwiGLU MLP, scalar activation `a` after the gate SiLU times up product.
- Activation: `a * w_router` (router-weighted), so a neuron's contribution to the residual stream ("write") is `a * w_router * W_down[:, n]`.
- Percentiles: each neuron's activations are ranked across its own activations on a corpus. Because SwiGLU neurons are bipolar and the median is near 0, p1, p3 are strong negatives and p97, p99 strong positives. "p97/3" means the union of the top 3% positive and the bottom 3% negative activations of that neuron. This is what the brief calls "top 3% of activations": per-neuron, two-tailed, router-weighted (the early band-ablation runs did not include router weighting).
- Mode: a semantic grouping of a neuron's subcircuits; subcircuit: a set of input neurons and reader neurons connected through the target neuron.

Claim 1, functional sparsity. Methods and numbers (Table 1, 8.4M web-text tokens; base loss 1.886):
- Layer top-K, router-weighted: keep the top 320 of 4,096 routed neurons per token per layer (7.8%, 40 per expert on average; 92.2% ablated) and zero the rest before the down projection. Loss 1.940 (+0.054), KL 0.140. Kept activations fall almost entirely in p97/3.
- Expert top-64 (87.5% ablated): +0.066 loss, KL 0.152. Expert top-32 (93.8%): +0.146, KL 0.263.
- Mean-ablating central bands: p25 to p75 (49.7% of cells) +0.010; p10 to p90 (78.6%) +0.040; p5 to p95 (88.8%) +0.091.
- Controls: random 64 per expert +2.328 loss; the bottom 448 per expert kept (inverse of top-64) +2.701.
- Benchmarks under top-320: classic suite within about 2 points (ARC-C +0.3, MMLU -1.2, GSM8K -1.7); thinking evals drop more (GPQA-Diamond -11.1, SuperGPQA -13.0, MBPP+ -20.9), and the failure mode is endless self-checking with the correct answer present. Restoring the double-negative activations restored reasoning. A light fine-tune of 100M tokens (91M SFT distillation against the original model's next-token distribution over 1,000 steps, plus 9M RL tokens) restores parity or better (SuperGPQA 45.4 vs 44.8) (extract).
- RelP (gradient-based attribution): a few thousand neurons, 0.2 to 0.8% of active routed units, recover 90% of the logit margin for a given prediction (extract).
- Transcoders trained freely mostly rediscover neurons; non-neuron features are coalitions of 2 to 64 co-firing neurons; top-K MLPs beat transcoders on KL at matched sparsity after fine-tuning (extract).

Claim 2, few modes. Pipeline: run 10B tokens; collect 1,000 examples in each of 14 percentile bands per neuron (14,000 per neuron), plus 30,000 extra examples at p95/5 or stronger for 2,734 neurons. For each firing, decompose gate and up pre-activations exactly into direct contributions from earlier neurons' writes (with RMSNorm scale frozen at its actual value), keep top 400 inputs per projection split into drivers and suppressors, propagate the write forward through frozen RMSNorm to estimate first-order reader changes, weight profiles by TF-IDF, kNN graph (15 neighbors), Leiden community detection, merge groups whose top-50 inputs and readers overlap at least 30%. Median 19 subcircuits per pole (1 to 51); subcircuits overlap only 5.2% (inputs) and 6.5% (readers) on average. Claude Opus 5.5 groups subcircuits into modes. Results: median 1.0 mode per pole (mean 2.82); 69% of poles are dominated by one mode (47% single function, plus 46% of multi-mode poles where the main mode does at least 90% of the ablation damage) (extract).

Claim 3, predictable from descriptions (extract):
- Detection (pick the firing snippet from a pair, chance 0.5): 0.946 vs random text, 0.885 vs same-expert text, pooled over p97 and up; 0.972 vs random at p99.99+.
- Mode classification macro-F1: from inputs 0.968, from readers 0.853, from text 0.707 (chance 0.290).
- Simulation: replacing ablated activations above p99.9 with judge-simulated values recovers 73% of the damage that a perfect oracle restores (oracle restores 96%). Bands p99.9 and up account for 73% of all neuron ablation damage.
- Input synthesis: 120 of 128 neurons (94%) driven above their maximum on the 10B corpus.

Limitations stated: mode counts depend on description granularity; input synthesis tests only the primary mode; activation size is hard to simulate because it depends on accumulated context (text features explain about 14% of firing-size variance); weak activations and double-negative activations remain under-studied; one model.

### "Interpreting Inkling 2 Mini" (Lee, Krishnamurthy, Merullo, Balsam, Lubana, 20 Nov 2026)

Architecture facts stated in the essay that matter here: 24 layers, width `d = 2304`, 4 blocks of 6 layers. No single residual stream; each block keeps a running sum of its own layers' outputs from zero. Each sublayer reads `x = sum_j alpha_j * v_hat_j` with `alpha = softmax(q . v_hat_j / sqrt(d))` over the token embedding (block 0), completed blocks, and the current running sum, then RMSNorm. Blocks 2 and 3 (layers 7 to 18) loop 3 times. In loops 2 and 3 each looped layer has four cross-loop readers `u_Q, u_K, u_V, u_MoE`, each forming a message `m_s = sum_{j in {2,3}} beta_j v_hat_j[t-1]`. Mixing: `z = P[RMSNorm(x); RMSNorm(m)]` with `P` of shape 64 x 2d, `kappa = 1 - tanh(F z)` in (0, 2), `gamma = tanh(G z)` in (-1, 1), `F, G` of shape d x 64, and `x' = kappa * x + gamma * m`. Each MoE has 147,456 neurons per layer (128 x 1152).

Findings, with every number given:
1. Exact input decomposition. Each layer input is an exact weighted sum of earlier sublayer outputs, `x' = sum_w c_w`, with shares `s_w = c_w . x' / |x'|^2` that sum to 1. Averaged over 13,233 tokens from 400 FineWeb documents.
2. Depth-wise attention favors recent layers, like a sliding window.
3. Cross-loop attention: later loops read block 3 more than block 2; block 3 almost always reads itself from the previous loop.
4. Weight-space reader and writer analysis: 104 reader vectors (56 depth-wise: 31 attention, 24 MoE, 1 before the output; plus 48 cross-loop: 12 looped layers x 4). 24 x 147,456 = 3,538,944 down-projection columns ("writers"). All 368 million writer-reader pairs scored by absolute cosine; the top 0.001% gives a threshold of 0.30 and 3,490 pairs. Depth-wise pairs concentrate in the last layer of each block (6, 12, 18): layer 6 writes mostly for block 4's readers, layer 12 for blocks 3 and 4, layer 18 for block 4. Cross-loop pairs concentrate in block 3, especially its last layer, writing mostly for layer 13. Interpretation: final layers "tag" activations so later readers know which block to read.
5. Massive outliers: after RMS normalization (squared norm 2304, so a uniform coordinate would hold about 0.043%), coordinate 2027 averages +32.4, about 46% of the squared norm. It surges at layer 18 at the end of loops 1 and 2 and gets a large negative write at layer 18 of loop 3 (cleanup before the coda).
6. Reading the outlier: for layer 13 in loop 2, coordinate 2027 contributes 17%, 19%, 11%, 6% of the block-3-versus-block-2 score gap for `u_Q, u_V, u_K, u_MoE`; the top 5 coordinates contribute 60%, 62%, 32%, 30%; those 5 are within the top 15 coordinates by RMS-normalized magnitude.
7. Lenses (1,250 FineWeb documents; tuned lens trained on 23,750): plain logit lens works only in the last few layers of each loop; tuned lens and "coda lens" (running the final block 4 on intermediate states) work well throughout. Early-loop layers are more decodable than mid-block layers of any loop. Metrics: whether the lens top-50 contains the final token and lens mass on the model's top-50 (exact values only in the figure).

### Which Goodfire methods run on weights alone

Weights only (no forward passes):
- Writer-reader cosine scans: every down-projection column against every depth-wise query `q`, cross-loop reader `u`, router row, or another neuron's gate and up rows (the Inkling 368M-pair scan; the "input neuron" definition in the basis paper, condition 1, is weight alignment between an upstream write vector and the target's gate or up row).
- Logit connections: a neuron's down column projected through the final RMSNorm gain and the unembedding gives its direct vocabulary effect (the essay's "don't say a dot" neuron, eight of ten strongest negative logit connections start with a dot). Exact for the direct path only; RMSNorm scale is per-token, so magnitudes are approximate.
- Structural facts readable from weights: which coordinates the readers `u` and queries `q` weight heavily; the rank-64 structure of the keep and gain controllers (`P`, `F`, `G`); per-expert router rows.

Need activations (forward passes):
- Percentiles (p97/3), top-K masks, band ablation, and every sparsity number.
- The exact input decomposition and shares `s_w` (needs real `alpha`, `beta`, `kappa`, `gamma`, RMS values).
- Outlier coordinate magnitudes and when they surge.
- Subcircuit extraction, mode clustering, RelP attribution, simulation, detection, input synthesis.
- Tuned lens (training) and coda lens.

INFERRED: the weight-only scans give candidate structure cheaply, but every claim about importance (which neurons, which coordinates, which pairs matter) was confirmed on activations in the source work.

---

## Part 3. Bytes model for Inkling-2-Mini (context, from the brief)

From the brief: 7.9 GB 2-bit build; per token, attention about 50% of bytes read (8-bit, 9% of file, core layers run 3 times), experts about 30%, vocabulary matrix (201,024 x 2304) about 14%, the rest about 6%. Experts are about 80% of the file.

INFERRED arithmetic used below. Per-token weight reads in "layer applications": 12 non-core layers once plus 12 core layers 3 times gives 48. Each expert is 3 x 2304 x 1152 = 7.96M parameters; 8 per token per layer application; at about 2.06 bits (6.3 GB for 24.5B expert parameters) that is about 16 MB per layer application and about 0.79 GB per token if no expert repeats. That matches 30% of about 2.8 GB per token, which also matches attention at about 1.4 GB per token (0.71 GB of 8-bit attention weights x 48/24). The vocabulary matrix is 463M entries, about 0.4 GB per token, consistent with 8-bit. Core passes 2 and 3 account for 24 of the 48 layer applications, so they read about 25% of all bytes in attention and about 15% in experts.

One structural detail from Goodfire's appendix matters (INFERRED from it): at the first core layer (layer 7), block 2's running sum is empty, so the depth-wise input `x` mixes only the embedding and block 1, which do not change across loops. The pass-2 and pass-3 inputs at layer 7 therefore differ from pass 1 only through the message term.

---

## Part 4. Hypotheses

Each combines (a) an NCM mechanism, (b) a Goodfire finding, and (c) the looped MoE structure. None is plain low-bit quantization, REAP expert pruning, DSpark speculative decoding, early exit after pass 1, or cross-pass contrastive decoding. Gains are bytes-per-token estimates under the bytes model above, so speed gains assume a strictly bandwidth-bound decoder.

### H1. Jacobi-batched loop passes (passes 2 and 3 share one weight read)

Idea. Run pass 3 at the same time as pass 2, feeding it a predicted pass-2 message instead of the real one, so each core weight is read once for both passes. The predictor is a tiny map from pass-1 states to the pass-2 message, fitted on activations or with a light fine-tune.

Mechanism. On a bandwidth-bound laptop a batch of 2 costs about the same as a batch of 1. NCM shows that a model trained with conditioning corruption can take a noisy earlier level and still refine correctly; the loop is the same kind of refinement chain, and Goodfire's controllers (`kappa`, `gamma`, rank-64) mean pass 3 sees the previous pass only through a gated, normalized message, which may tolerate a prediction error. Goodfire also shows the cross-loop read is dominated by block 3 and by a few control coordinates, so the message may be quite predictable.

Cheapest test. Forward passes on the 16 GB Mac: log `m` at every core layer for passes 1 to 3 on 2,000 to 5,000 tokens; first try zero-cost substitutes (pass 3 uses the pass-1 message; or `m1 + (m1 - m0)` style extrapolation is not available, so use `m1` and a least-squares linear map `m1 -> m2` per layer); measure KL of final logits and perplexity on held-out text. If KL stays under about 0.05 with the linear map, prototype the batched kernel.

Expected gain. Saves 12 of 48 attention layer applications for certain (12.5% of bytes) and up to 12 of 48 expert applications if passes 2 and 3 pick the same experts (up to another 7.5%). About 1.14x to 1.25x tokens per second.

Strongest reason to fail. Goodfire shows block 3 reads itself from the previous loop almost exclusively, which suggests pass 3 depends tightly on pass 2's exact output; the message may carry the very information the third loop exists to refine, so a predicted message makes pass 3 redundant with pass 2.

### H2. Pass-1 neuron mask reused in passes 2 and 3, plus down-column sparsity everywhere

Idea. In pass 1, compute each routed expert fully and record its strongest neurons (router-weighted, two-tailed, as in Goodfire's p97/3); in passes 2 and 3, for experts the router picks again, read only those neurons' gate rows, up rows, and down columns. In every pass, read down columns only for the top 320 or so of the 9,216 routed neurons.

Mechanism. Goodfire found 92.2% of routed activations can be zeroed for +0.054 loss on Qwen3.6, with the kept set living at p97/3. The looped core applies identical weights to related inputs three times, so the strong-neuron set in pass 2 should largely be a subset of a slightly enlarged pass-1 set. NCM analogy: coarse level fixes which positions matter, finer levels only add residual detail there.

Cheapest test. Forward passes on the Mac: log routing and per-neuron activations per pass; compute (a) the fraction of pass-2 and pass-3 routed experts that were also chosen in pass 1 for the same layer, (b) recall of pass-2 top-320 neurons inside pass-1 top-K' for K' in {640, 1280}; then run the masked model and measure KL. Weights are needed only to build a neuron-major layout.

Expected gain. Core passes 2 and 3 read about 15% of bytes in experts; keeping 15% of neurons saves about 12.75%; down-column sparsity in the other 24 applications saves about 4.5%. About 1.2x if expert overlap across passes is high; scales down linearly with overlap.

Strongest reason to fail. Scattered row reads on Metal are slower than dense streaming unless rows are laid out neuron-major and grouped, and 2-bit group quantization makes row-granular reads awkward; also Goodfire found reasoning collapse came from dropping double-negative (gate and up both negative) activations, so the mask must keep that quadrant explicitly or reasoning evals will drop.

### H3. Coarse-to-fine vocabulary head (two-level unembedding)

Idea. Cluster the 201,024 unembedding rows into about 512 clusters (weights only), score the final state against the 512 centroids, then read and score only the rows of the top few clusters. This is NCM's residual-codebook head applied to the output layer: coarse code first, residual detail only where needed.

Mechanism. The repo's NCM head scores codes by negative squared distance to codebook entries; a residual VQ of the unembedding makes the centroid score a cheap upper-level approximation of the logit. Goodfire's tuned lens and coda lens show the final distribution is already readable before the coda, so pass-3 or early-coda states can prefetch the likely clusters while the coda runs.

Cheapest test. Weights: k-means on unembedding rows (CPU, minutes to an hour). Forward passes on the Mac: on 5,000 tokens, measure recall of the true top-1 token and of top-p (p = 0.9) mass inside the top-c clusters for c in {4, 8, 16}; then measure the KL of the renormalized truncated distribution.

Expected gain. Vocabulary is about 14% of bytes; reading about 2% of rows plus the centroid matrix cuts that to under 1%. About 1.15x.

Strongest reason to fail. Sampling at temperature or with top-p needs calibrated tail mass, and clusters of rare tokens (code, multilingual, numbers) may be missed exactly when they matter; the output RMSNorm and any logit soft-capping can make centroid scores a poor proxy.

### H4. Pass-dependent precision for core attention, stored as successive refinement

Idea. Store core attention weights as a residual pair (a 3-bit base plus a 5-bit residual, same total size as today's 8-bit), read only the base in passes 1 and 2 and base plus residual in pass 3 and the non-core layers. NCM-style nested levels let one stored tensor serve both precisions.

Mechanism. NCM trains early levels to be useful on their own (nested dropout) and later levels to correct earlier errors (corruption); the loop is a built-in correction chain, so early passes may tolerate coarser weights if the last pass is exact. Goodfire's finding that early-loop states already decode well (tuned and coda lens) suggests pass 1 does coarse work.

Cheapest test. Weights: build the base and residual for core attention (residual quantization, CPU). Forward passes on the Mac: compare final-logit KL for (a) all passes 8-bit, (b) passes 1 and 2 base-only, (c) pass 1 base-only. Note that pass-1 and pass-2 KV caches would then be computed with base weights, which the test captures.

Expected gain. Core passes 1 and 2 read about 25% of bytes in attention; dropping from 8 to 3 bits saves about 15.6%. About 1.19x, no memory cost.

Strongest reason to fail. Attention was kept at 8-bit in the shipped 2-bit build presumably because low-bit attention hurts; error in passes 1 and 2 enters their KV caches and is reused by every later token, so it compounds over the context instead of being corrected by pass 3.

### H5. Cross-pass KV residual caching

Idea. Store the pass-1 KV cache in full and the pass-2 and pass-3 caches as low-bit residuals against pass 1 (or reuse pass 1 outright for layers where they match). At layer 7, skip the pass-2 and pass-3 K and V projections entirely if the message gain is small.

Mechanism. NCM's residual coding: later levels store only what earlier levels missed. INFERRED from Goodfire's appendix: the layer-7 input in passes 2 and 3 equals pass 1's except through `gamma * m` and `kappa`, so its K and V may be close to pass 1's; deeper core layers may also stay close if the loop makes small corrections.

Cheapest test. Forward passes on the Mac: per core layer and head, cosine similarity and relative norm of `K2 - K1`, `V2 - V1`, `K3 - K2`; then substitute quantized residuals and measure KL at 4k to 16k context.

Expected gain. Memory: the core keeps 36 of 48 per-layer KV caches; residuals at 2 bits instead of 8 save about 50% of KV memory. Bandwidth: at long contexts, KV reads become a large share of bytes, so a similar share of those reads goes away; at short contexts the gain is near zero. Skipping K and V projections at layer 7 in passes 2 and 3 saves well under 1% of weight bytes.

Strongest reason to fail. Each pass keeps its own cache because the passes compute different things; if `K2 - K1` is as large as `K1`, residual coding saves nothing, and laptop contexts may be short enough that KV memory never binds.

### H6. Low-rank pass-delta attention

Idea. For passes 2 and 3, compute Q, K, V as the pass-1 projection plus a projection of the input change onto a fixed low-rank basis, using precomputed `W U` (2304 x r) instead of the full `W`. Read the full weights only in pass 1.

Mechanism. Goodfire shows cross-loop communication is dominated by a handful of coordinates (top 5 coordinates give 30 to 62% of reader score gaps) and the controllers that mix the message are rank 64; if the change between passes lives in a low-dimensional subspace, linear projections of that change are cheap.

Cheapest test. Forward passes on the Mac: collect RMS-normalized inputs to each core sublayer in all passes; PCA on the pass-to-pass difference; report variance captured at r = 64, 128, 256 and the resulting relative error in Q, K, V.

Expected gain. If QKV is about 60% of attention bytes, passes 2 and 3 save about 15% of total bytes. About 1.18x.

Strongest reason to fail. The deltas are elementwise products (`kappa * x`, `gamma * m`) of full-rank vectors, so they are probably not low rank; the outlier coordinates hold most of the norm but the information may live in the rest.

### H7. Weight-only control-neuron protection, enabling sub-2-bit experts

Idea. Use Goodfire's writer-reader cosine scan (weights only) to find the few thousand expert neurons whose down columns align with the 104 depth-wise and cross-loop readers or with the outlier coordinates (2027 and the top 5), keep those at 8-bit, and push all other expert neurons to about 1.5 to 1.6 bits.

Mechanism. These "tagging" neurons steer which blocks every later layer and loop reads; an error in them is amplified through three loops and every later reader, while ordinary content neurons are clean, single-mode units whose errors stay local (Goodfire: most poles have one dominant mode). Protection by function, chosen from weights, differs from activation-magnitude protection (AWQ, SpQR).

Cheapest test. Weights: run the 368M-pair scan (cheap on CPU with chunked matmuls) and list neurons above |cos| 0.30. Forward passes on the Mac: compare KL and perplexity of (a) uniform 1.6-bit experts vs (b) the same plus protected control neurons vs (c) the same plus an equal number of randomly protected neurons.

Expected gain. Experts from about 2.06 to about 1.6 bits saves about 1.4 GB (about 18% of the file) and about 7% of bytes per token. Memory is the main win.

Strongest reason to fail. 3,490 high-cosine pairs may cover only part of the control signal (the outlier writes may come from attention and from many weak neurons summing), and sub-2-bit expert quantization may fail for content neurons regardless of protection.

### H8. Pass-specific sublayer skipping ranked by exact input shares

Idea. Use Goodfire's exact decomposition (`s_w`, shares that sum to 1) to find core attention and MoE sublayers whose outputs in passes 2 and 3 contribute little to any later input, and skip those sublayer applications in those passes only.

Mechanism. Depth-wise attention reads mostly recent layers and cross-loop reads mostly block 3, so some sublayer outputs in some passes may be read by almost nobody. NCM analogy: an input-dependent schedule (the paper's own future work) spends compute only where later levels read it.

Cheapest test. Forward passes on the Mac: compute `s_w` per sublayer per pass (from logged `alpha`, `beta`, `kappa`, `gamma`, RMS values) on 13k tokens as Goodfire did; ablate the lowest-share sublayers greedily and track KL.

Expected gain. Skipping 30% of core attention applications in passes 2 and 3 saves about 7.5% of bytes; adding MoE skips could reach about 12%. About 1.08x to 1.14x.

Strongest reason to fail. Direct shares ignore indirect effects through sequence attention and KV caches, so a low-share sublayer can still matter; Goodfire's sliding-window pattern suggests every layer is read by its neighbors.

### H9. Intra-expert neuron pruning by tail damage, repaired with a 100M-token fine-tune

Idea. Rank every expert neuron by how often it reaches its p97/3 tails and how much damage ablating it causes on a mixed corpus, delete the bottom 20 to 30% of neurons inside experts, and repair with the Goodfire recipe (about 91M SFT distillation tokens plus about 9M RL tokens).

Mechanism. Goodfire shows weak activations carry little and that a tiny fine-tune restores reasoning after heavy sparsification; Cammarata reports experts are specialized and narrow, so some neurons may serve rare offshoot modes the deployment never needs. This keeps all 128 experts (unlike REAP) and removes width.

Cheapest test. Forward passes on the Mac: per-neuron tail-firing counts over about 1M tokens (feasible with streaming histograms); prune and measure KL before any fine-tune. The repair fine-tune needs a rented GPU (QLoRA-style on 26B MoE, a day or less).

Expected gain. 25% of expert width saves about 1.6 GB (20% of the file) and about 7.5% of bytes per token.

Strongest reason to fail. The looped core reuses the same neurons in three passes, so a neuron that looks rare in one pass may carry the loop's refinement; and the fine-tune repair was shown on a non-looped model.

### H10. Resident coarse experts with streamed residuals and loop-lookahead prefetch

Idea. Keep a 1-bit coarse level of every expert in RAM and stream the residual level from SSD only for routed experts (and only for their strong neurons), starting the reads for passes 2 and 3 as soon as pass 1 reveals the routing. This lets a 3 to 4-bit-quality model run with about 8 GB resident.

Mechanism. NCM's multiscale residual codes make a coarse level usable alone and residuals additive; the loop provides a natural prefetch window, since pass 1 of a token predicts passes 2 and 3 if routing persists across passes, giving roughly a third of a token's time to fetch.

Cheapest test. Forward passes on the Mac: the routing-overlap statistic from H2; measure per-token residual bytes needed versus SSD read bandwidth; then KL of coarse-only versus coarse plus residual.

Expected gain. Memory: resident expert weights halve or better at equal quality versus a 2-bit build; speed likely flat or slower.

Strongest reason to fail. SSD bandwidth (several GB/s) and per-request latency are too low for about 0.4 GB or more of residual reads per token unless neuron-level sparsity (H2) also holds; a 1-bit coarse level of a 26B MoE may be too weak to route and compute usefully.

### Ranking by probability true times size of gain

| Rank | Hypothesis | P(true), my estimate | Gain if true | Product (rough) | First test needs |
| --- | --- | --- | --- | --- | --- |
| 1 | H3 coarse-to-fine vocab head | 0.65 | about 15% speed | 0.10 | weights, then Mac forward passes |
| 2 | H2 pass-1 neuron mask plus down sparsity | 0.45 | about 20% speed | 0.09 | Mac forward passes |
| 3 | H1 Jacobi-batched passes 2 and 3 | 0.30 | about 14 to 25% speed | 0.06 | Mac forward passes |
| 4 | H4 pass-dependent attention precision | 0.35 | about 19% speed | 0.07 | weights, then Mac forward passes |
| 5 | H7 control-neuron protection, sub-2-bit experts | 0.30 | about 18% memory, 7% speed | 0.05 | weights, then Mac forward passes |
| 6 | H9 neuron pruning plus light fine-tune | 0.40 | about 20% memory, 7% speed | 0.05 | Mac, then rented GPU |
| 7 | H8 pass-specific sublayer skipping | 0.35 | about 8 to 14% speed | 0.04 | Mac forward passes |
| 8 | H5 cross-pass KV residuals | 0.30 | large at long context, about 0 at short | 0.03 (context-dependent) | Mac forward passes |
| 9 | H6 low-rank pass deltas | 0.15 | about 18% speed | 0.03 | Mac forward passes |
| 10 | H10 resident coarse, streamed residual | 0.25 | memory large, speed negative | 0.02 for speed, higher for memory | Mac forward passes |

H4 has a slightly higher product than H1 in the table, but I would still test H1 third: its first experiment costs one logging run that also feeds H2, H5, H6, and H8, and a positive result there is the largest single structural win for a looped model. H4 needs residual-quantized weights built first.

Tests to run first:
1. H3, because it needs only k-means on the unembedding plus one recall measurement, and the gain is close to guaranteed if recall holds.
2. H2, because the routing-overlap and neuron-overlap statistics across passes decide H2, bound H1's expert savings, and decide H10's prefetch window.
3. H1, run from the same logging pass: substitute the pass-1 message and a linear-predicted message into pass 3 and measure KL.

One instrumentation run serves all three plus H5, H6, H8: on 2,000 to 5,000 tokens, log per pass and per core layer the inputs `x` and `x'`, messages `m`, `kappa`, `gamma`, routing indices and weights, per-neuron routed activations, and K and V. At fp16 that is a few GB, which fits on a 16 GB Mac if streamed to disk.

Self-critique. The bytes model assumes perfect bandwidth-bound decoding and dense streaming; real Metal kernels lose efficiency on gathers and small batches, so H2 and H3 may deliver half their byte savings. All Goodfire sparsity numbers come from Qwen3.6-35B-A3B, a non-looped model with a plain residual stream; the attention-residual and looped design may spread computation differently. H1 and H6 lean on Goodfire's observation that cross-loop control runs through a few coordinates, but control signals and content are different things, and the content is what the loop refines.
