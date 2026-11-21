# Inkling 2 Mini forward pass: implementation-level specification

Compiled 21 Nov 2026 from code and primary pages. No weights were downloaded. Only the safetensors JSON header (first 200 KB of the shard) and the index were read, to get tensor names, dtypes and shapes.

## Sources read

| Tag | Source | Version |
|---|---|---|
| LC | llama.cpp PR #32694 "Inkling 2 Mini", merged 2026-11-20 22:01 UTC | merge commit `29b52d39`: `src/models/inkling2-loop.cpp`, `src/models/inkling2.cpp`, `src/models/models.h`, `src/llama-graph.cpp` (`build_moe_ffn`), `conversion/inkling2.py`, `gguf-py/gguf/tensor_mapping.py`, `gguf-py/gguf/constants.py` |
| LD | llama.cpp draft PR #32695 "model: Inkling-2-Mini - looped arch" (open) | diff and PR comments |
| HF | transformers PR #51050 "Add Inkling2 Mini" (open, not merged) | head `6173a2f6`: `modeling_inkling2_loop.py`, `conversion_mapping.py`, `core_model_loading.py` |
| MLX | mlx-vlm PR #2778 (merged; Blaizzy/mlx-vlm and Neywa-Labs/mlx-vlm point at the same PR) | merge `111ac668`: `mlx_vlm/models/inkling2/language.py` (scanned, not read line by line) |
| CFG | `thinkingmachines/Inkling-2-Mini` `config.json`, `model.safetensors.index.json`, safetensors header, `hf_quant_config.json` | main, 21 Nov 2026 |
| BLOG | https://huggingface.co/blog/inkling2mini | 20 Nov 2026 |
| VLLM | https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini | 21 Nov 2026 |
| GF | https://www.goodfire.com/research/interpreting-inkling-2-mini | 20 Nov 2026 |

The HF repo contains no `.py` files (tree: config, tokenizer, chat template, one 19.5 GB safetensors shard, `mtp.safetensors`, quant config).

Notation used below: `d = 2304`, `H = 24` query heads, `Hkv = 8` KV heads, `dh = 128`, `d_rel = 16`, `K = 4` (conv kernel), `E = 128` experts, `k = 8` active, `I = 1152` expert width, `r = 64` gate rank. `rms(x) = x / sqrt(mean(x^2) + eps)` (no weight), `RMSNorm_w(x) = w * rms(x)`. Layers are 0-indexed (GF uses 1-indexed layers and blocks).

## Key config values (CFG)

hidden 2304, 24 layers, vocab 201024 (padded) with `unpadded_vocab_size` 200058, 24 heads / 8 KV heads / head_dim 128 for both sliding and full layers, `d_rel` 16, `rel_extent` 1024, `sliding_window_size` 1024, `local_layer_ids` = every layer except 5, 11, 17, 23, `sconv_kernel_size` 4, `n_layers_per_attn_res_block` 6, `attn_res_normalize_values` true, `qk_l2_clip_threshold` 22.627 (= 16*sqrt(2)), `rel_l2_clip_threshold` 8.0, `log_scaling_n_floor` 8192, `log_scaling_alpha` 0.1, `rms_norm_eps` 1e-6, `use_value_norm_no_gamma` true, `final_logit_softcapping` 30, `n_routed_experts` 128, `num_experts_per_tok` 8, `n_shared_experts` 0, `intermediate_size` 1152, `mlp_gate_softcap` 4, `mlp_up_softcap` 25, `route_scale` 8, `use_gate_bias` true, `q_bias`/`o_bias` false, `loop_config` = {core_start 6, core_end 18, num_passes 3, gate_rank 64, is_varloop false}. Quantization: modelopt NVFP4 (experts of layers 0..19), MXFP8 (experts of layers 20..23). MTP drafter (`mtp_config`, 3 layers, `target_layer_ids` [5, 23, 41, 48]) lives in `mtp.safetensors` and is out of scope.

---

## A. Tensor inventory

Shapes are given numpy/PyTorch style (row-major, `[out, in]` for linear weights) for HF, and ggml style `{ne0, ne1, ...}` for GGUF (ggml `{ne0, ne1}` equals numpy `[ne1, ne0]`). HF names come from the real safetensors header. GGUF names come from LC (`constants.py`, `tensor_mapping.py`, `inkling2.cpp` `create_tensor` calls). "SWA" = the 20 local layers; "FULL" = layers 5, 11, 17, 23; "CORE" = layers 6..17.

### A.1 Global tensors

| HF safetensors | dtype, shape | GGUF (merged `inkling2_loop`) | ggml shape | Role |
|---|---|---|---|---|
| `model.llm.embed.weight` | BF16 [201024, 2304] | `token_embd.weight` | {2304, 201024} | Token embedding (padded vocab). |
| `model.llm.embed_norm.weight` | BF16 [2304] | `token_embd_norm.weight` | {2304} | RMSNorm weight applied to every looked-up embedding (not to image embeddings in LC). |
| `model.llm.attn_res.output_query` | F32 [2304] | `output_res_query.weight` | {2304} | Depth-attention query that forms the final hidden state from all block sums. |
| `model.llm.norm.weight` | BF16 [2304] | `output_norm.weight` | {2304} | Final RMSNorm. |
| `model.llm.unembed.weight` | BF16 [201024, 2304] | `output.weight` | {2304, 201024} | LM head, untied from `embed`. Rows past 200058 are padding and are dropped after the matmul. |
| `model.llm.middle_cycle_cross_pass_queries` | F32 [24, 2304] | merged into `blk.N.loop_msg_query` (see A.3) | | Cross-pass message queries: rows `2i` = Q-input query, `2i+1` = MLP-input query of core layer `6+i`. |
| `model.llm.middle_cycle_cross_pass_kv_queries` | F32 [24, 2304] | merged into `blk.N.loop_msg_query` | | Rows `2i` = K-input query, `2i+1` = V-input query of core layer `6+i`. |

### A.2 Per-layer tensors (N = 0..23)

| HF safetensors | dtype, shape | Layers | GGUF | ggml shape | Role |
|---|---|---|---|---|---|
| `model.llm.attn_res.attn_queries.{K}` | F32 [2304] | K = 0..22 except 4,10,16,22 | `blk.{K+1}.attn_res_query.weight` | {2304, 1} | Depth-attention query forming the attention input of layer K+1 (index shifted by one at conversion; layer 0 has none). |
| `model.llm.attn_res.attn_queries.{K}` | F32 [3, 2304] | K = 4, 10, 16, 22 | `blk.{K+1}.attn_res_query.weight` | {2304, 3} | Three queries (Q, K, V inputs) for FULL layers 5, 11, 17, 23. |
| `model.llm.attn_res.mlp_queries.{N}` | F32 [2304] | all | `blk.N.ffn_res_query.weight` | {2304} | Depth-attention query forming the MoE input. |
| `model.llm.layers.N.attn_norm.weight` | BF16 [2304] | SWA | `blk.N.attn_norm.weight` | {2304} | RMSNorm before Q/K/V projections (one shared input). |
| `model.llm.layers.N.q_input_norm.weight`, `k_input_norm`, `v_input_norm` | BF16 [2304] each | FULL | `blk.N.attn_q_inp_norm.weight`, `attn_k_inp_norm`, `attn_v_inp_norm` | {2304} | Separate RMSNorms for the three separately mixed inputs. |
| `model.llm.layers.N.attn.wq_du.weight` | BF16 [3072, 2304] | all | `blk.N.attn_q.weight` | {2304, 3072} | Q projection, 24 heads x 128. |
| `model.llm.layers.N.attn.wk_dv.weight` | BF16 [1024, 2304] | all | `blk.N.attn_k.weight` | {2304, 1024} | K projection, 8 heads x 128. |
| `model.llm.layers.N.attn.wv_dv.weight` | BF16 [1024, 2304] | all | `blk.N.attn_v.weight` | {2304, 1024} | V projection, 8 heads x 128. |
| `model.llm.layers.N.attn.wo_ud.weight` | BF16 [2304, 3072] | all | `blk.N.attn_output.weight` | {3072, 2304} | Output projection. |
| `model.llm.layers.N.attn.gate_attn.weight` | BF16 [3072, 2304] | all | `blk.N.attn_gate.weight` | {2304, 3072} | Sigmoid output gate on the per-head attention output, computed from the Q-input. |
| `model.llm.layers.N.attn.wr_du.weight` | BF16 [384, 2304] | all | `blk.N.attn_rel.weight` | {2304, 384} | Per-head relative-position state, 24 heads x 16. |
| `model.llm.layers.N.attn.rel_logits_proj.proj` | BF16 [16, 1024] | all | `blk.N.attn_rel_bias.weight` | {16, 1024} | Bank of 16 bias-vs-distance profiles over distances 0..1023. Transposed at conversion (`.T`); GGUF is numpy [1024, 16]. Kept F32 in GGUF. |
| `model.llm.layers.N.attn.q_l2_clip.scale` | BF16 [24, 128] | all | `blk.N.attn_q_clip.weight` | {128, 24} | Per-channel scale on Q before L2 clipping. F32 in GGUF. |
| `model.llm.layers.N.attn.rel_l2_clip.scale` | BF16 [24, 16] | all | `blk.N.attn_rel_clip.weight` | {16, 24} | Per-channel scale on the relative state before L2 clipping. F32 in GGUF. |
| `model.llm.layers.N.attn.attn_sink_proj.weight` | BF16 [24, 2304] | all | `blk.N.attn_sink_proj.weight` | {2304, 24} | Per-token, per-head sink logit. |
| `model.llm.layers.N.attn.k_sconv.weight`, `v_sconv.weight` | BF16 [1024, 1, 4] | SWA | `blk.N.attn_k_conv.weight`, `attn_v_conv.weight` | {4, 1024} | Static depthwise causal conv on K and V (PyTorch layout: tap index 3 = current token). Squeezed at conversion. F32 in GGUF. |
| `model.llm.layers.N.attn.k_sconv.static_w`, `v_sconv.static_w` | BF16 [4, 1024] | FULL | `blk.N.attn_k_conv.weight`, `attn_v_conv.weight` | {4, 1024} | Static taps of the dynamic conv; HF index 0 = current token. Converted by `flip(0).T`, so GGUF uses the same oldest-first layout as SWA layers. |
| `model.llm.layers.N.attn.k_sconv.weight_proj`, `v_sconv.weight_proj` | BF16 [32, 2304] | FULL | `blk.N.attn_k_conv_proj.weight`, `attn_v_conv_proj.weight` | {2304, 32} | Per-token dynamic taps: 8 KV heads x 4 lags, row index `head*4 + lag` (lag 0 = current token). |
| `model.llm.layers.N.attn_sconv.weight` | BF16 [2304, 1, 4] | all | `blk.N.attn_post_conv.weight` | {4, 2304} | Causal depthwise conv on the attention sublayer output (after `wo`). F32 in GGUF. |
| `model.llm.layers.N.mlp_sconv.weight` | BF16 [2304, 1, 4] | all | `blk.N.ffn_post_conv.weight` | {4, 2304} | Causal depthwise conv on the MoE output. F32 in GGUF. |
| `model.llm.layers.N.mlp_norm.weight` | BF16 [2304] | all | `blk.N.ffn_norm.weight` | {2304} | RMSNorm before router and experts. |
| `model.llm.layers.N.mlp.gate.weight` | BF16 [128, 2304] | all | `blk.N.ffn_gate_inp.weight` | {2304, 128} | Router. |
| `model.llm.layers.N.mlp.gate.bias` | F32 [128] | all | `blk.N.exp_probs_b.bias` | {128} | Selection-only bias (DeepSeek-V3 style `e_score_correction_bias`); renamed at conversion. |
| `model.llm.layers.N.mlp.experts.w13_weight` (+ `_scale`, `_amax`, `_outer_scale_denominator`) | U8 [128, 2304, 1152] NVFP4 packed (layers 0..19); F8_E4M3 [128, 2304, 2304] MXFP8 (layers 20..23) | all | `blk.N.ffn_gate_up_exps.weight` | {2304, 2304, 128} | Fused gate+up, logical [128, 2*1152, 2304]. In HF rows are INTERLEAVED: row `2j` = gate_j, row `2j+1` = up_j. Conversion de-interleaves to `[gate(1152); up(1152)]` and dequantizes to F32 (then GGUF requantizes per file type). |
| `model.llm.layers.N.mlp.experts.w2_weight` (+ companions) | U8 [128, 2304, 576] NVFP4; F8_E4M3 [128, 2304, 1152] MXFP8 | all | `blk.N.ffn_down_exps.weight` | {1152, 2304, 128} | Down projection, logical [128, 2304, 1152]. |
| `model.llm.layers.N.controllers.attn.down_weight` | BF16 [64, 4608] | CORE | `blk.N.loop_q_ctrl_down.weight` | {4608, 64} | Q-input message gate: down projection of `[rms(x); rms(m)]`. F32 in GGUF. |
| `model.llm.layers.N.controllers.attn.forget_weight` | BF16 [2304, 64] | CORE | `blk.N.loop_q_ctrl_forget.weight` | {64, 2304} | Q-input forget gate. |
| `model.llm.layers.N.controllers.attn.gain_weight` | BF16 [2304, 64] | CORE | `blk.N.loop_q_ctrl_gain.weight` | {64, 2304} | Q-input message gain. |
| `controllers.k.{down,forget,gain}_weight` | same | CORE | `blk.N.loop_k_ctrl_{down,forget,gain}.weight` | same | K-input gate. |
| `controllers.v.{down,forget,gain}_weight` | same | CORE | `blk.N.loop_v_ctrl_{down,forget,gain}.weight` | same | V-input gate. |
| `controllers.mlp.{down,forget,gain}_weight` | same | CORE | `blk.N.loop_ffn_ctrl_{down,forget,gain}.weight` | same | MoE-input gate. |

Note: the controller named `attn` gates the Q input (HF renames it `q_message_gate`; LC maps it to `loop_q_ctrl_*`).

Quantization of experts (from `hf_quant_config.json` and the LC dequantizer): NVFP4 = E2M1 nibbles (low nibble first), one E4M3 scale per 16 input columns, times a per-expert `amax[e] / outer_scale_denominator`. MXFP8 = E4M3 values times a UE8M0 power-of-two scale per 32 input columns. The quant config claims MXFP8 `weight_block_shape [32, 32]`, but the stored scale shape implies [1, 32]; the LC converter warns and uses [1, 32]. Non-expert weights are BF16; the 74 attention-residual query tensors (and the cross-pass queries) are F32.

### A.3 Conversion transforms (LC, merged `inkling2_loop`)

1. Prefix strip: `model.llm.` and `model.language_model.` become `model.`.
2. `attn_res.attn_queries.{i}` becomes `layers.{i+1}.attention_input.query` (index shift by one).
3. `mlp.gate.bias` becomes `mlp.gate.e_score_correction_bias`, written as `blk.N.exp_probs_b.bias`.
4. `w13_weight`: dequantize, then de-interleave `reshape(E, I, 2, d).transpose(1, 2)` so GGUF holds `[gate; up]`.
5. `static_w` (FULL layer K/V conv): `flip(0).T`. Local conv weights: squeeze the middle dim.
6. `rel_logits_proj.proj`: transposed.
7. Cross-pass queries: `blk.{6+i}.loop_msg_query.weight` = `stack([q[2i], kv[2i], kv[2i+1], q[2i+1]])`, i.e. ggml {2304, 4} with rows in order (Q, K, V, MLP). 12 tensors, layers 6..17.
8. Forced F32 (never quantized): `*_conv.weight`, `*_clip.weight`, `*_res_query.weight`, `attn_rel_bias.weight`, all `loop_*` tensors.
9. Metadata: `inkling2_loop.loop.start` 6, `.loop.end` 18, `.loop.count` 3, `.loop.gate_rank` 64, `.loop.varying` false; `block_count` stays 24. At load, LC expands to `24 + 2*12 = 48` cache slots; slots 24..47 alias the weights of physical layers 6..17 (`layer_phys(il) = 6 + (il - 24) % 12`).

Expected tensor count of a merged-format GGUF (INFERRED arithmetic): 20 common per layer x 24 + 1 extra x 20 SWA + 5 extra x 4 FULL + 23 `attn_res_query` + 5 globals = 548, plus 12 x 13 loop tensors = 704.

### A.4 Draft PR #32695 (`inkling2`, `block_count 48`) and the AtomicChat file

The AtomicChat/Inkling-2-Mini-GGUF file uses LD's format: `general.architecture = inkling2`, `inkling2.block_count = 48`. Per a PR comment, the file holds 734 tensors and only `blk.0..blk.23` exist; the merged LC loader fails with `blk.24.attn_q.weight not found`, and LD's own loader builds 714 tensors where the file has 734. How LD differs from LC:

- `block_count` = number of layer visits (48). Per-visit arrays (`num_attention_heads`, `num_key_value_heads`, `sliding_window_pattern`) have 48 entries, built from `visits = [0..5] + [6..17]*3 + [18..23]`. Tensors are stored once per physical layer (`blk.0..23`).
- Loop metadata keys: `num_loops` 3, `loop.core_start`, `loop.core_end`, `loop.gate_rank` (no `varying`). `is_varloop` is rejected.
- Cross-pass queries stay raw as two global tensors `loop_res_query` and `loop_kv_res_query`, ggml {2304, 24}. At pass > 0, layer `ip` uses rows `2*(ip-6)` and `2*(ip-6)+1` of each: `loop_res_query` gives (Q, MLP) messages, `loop_kv_res_query` gives (K, V) messages. Same math as LC.
- Controller names: `blk.N.loop_{attn,k,v,mlp}_{down,gain,forget}.weight`, same shapes as LC.
- Optional `INKLING2_NVFP4_NATIVE=1` keeps NVFP4 experts as GGML NVFP4 and writes an extra per-expert F32 `blk.N.ffn_{gate_up,down}_exps.scale` tensor ({128}) holding `amax / denominator`.
- LD's controller normalizes `[cur; message]` reshaped as two rows of 2304, so the RMS is per half. Same as LC and HF.

INFERRED: LD format count = 548 + 12 x 12 controllers + 2 query globals = 694. 734 - 694 = 40 = 20 NVFP4 layers x 2 `.scale` tensors, which suggests the AtomicChat file was converted with native NVFP4 scales for layers 0..19. Not verified against the file.

---

## B. Attention

Every layer has the same attention shape: 24 query heads, 8 KV heads (GQA group 3, query head `h` uses KV head `h // 3`), head_dim 128, no RoPE (LC sets `rope_type NONE`), scale `1/sqrt(128)`.

### B.1 Inputs

SWA layer: one depth-mixed input `x` (section C), `u = RMSNorm_{attn_norm}(x)`, and `u_q = u_k = u_v = u`.
FULL layer (5, 11, 17, 23): three depth-mixed inputs `x_q, x_k, x_v` (one per query row of `attn_res_query`), each with its own norm: `u_q = RMSNorm_{q_inp_norm}(x_q)`, likewise K and V.
Loop passes 2 and 3 (core layers): the message gates act on `x_q, x_k, x_v` BEFORE the norms (section D). For SWA core layers this makes the three inputs differ; each is then normalized with the same `attn_norm` weight.

### B.2 Projections and short convolution

```
q_raw = W_q u_q                          # [3072] -> [24,128]
k_lin = W_k u_k                          # [1024]
v_lin = W_v u_v                          # [1024]
r_raw = W_r u_q                          # [384]  -> [24,16]
t     = W_sink u_q                       # [24]   sink logit per head
g     = sigmoid(W_gate u_q)              # [3072] output gate
```

The "short convolution" (kernel 4, causal, depthwise, per channel) is applied in four places per layer: on `k_lin`, on `v_lin` (both BEFORE head reshape, clip and norm), on the attention sublayer output after `W_o`, and on the MoE output. Each conv has its own state of the last 3 inputs, per cache slot.

Static conv (SWA K/V, and both post convs), with `c[j]` for lag `j` (j = 0 current token):
```
conv(z)_t = (z_t + sum_{j=0..3} c[j] * z_{t-j}) / sqrt(K + 1) = (...)/sqrt(5)
```
In the GGUF/PyTorch layout `w[ch, 0..3]`, `c[j] = w[ch, 3-j]`.

Dynamic conv ("dynamic key/value convolution", FULL layers only): taps are the static taps plus per-token taps projected from the layer input, shared across the 128 channels of a KV head:
```
a_t = W_convproj u_k (or u_v)            # [32] -> a_t[head, lag]
dconv(z)_t[ch] = (z_t[ch] + sum_{j=0..3} (c[j][ch] + a_t[head(ch), j]) * z_{t-j}[ch]) / sqrt(2K + 1) = (...)/3
```
(HF: `weights = static_weight.T.view(8, 128, 4) + dynamic(…, 8, 1, 4)`; LC computes `ssm_conv` with static weights plus 4 per-lag dynamic terms, then scales by `1/sqrt(2*4+1)`.) Note the dynamic taps are computed from the current token's input and multiply past values.

```
k = conv(k_lin)  [SWA] or dconv(k_lin, a^k) [FULL]
v = conv(v_lin)  [SWA] or dconv(v_lin, a^v) [FULL]
```

### B.3 Clipping, value norm, log scaling

`clip_c(z) = z * min(1, c / (||z||_2 + 1e-6))` per head vector.
```
q_h = clip_{22.627}(s_q[h] ⊙ q_raw_h)         # s_q = q_l2_clip.scale [24,128]
k_j = clip_{22.627}(k_j)                       # no learned scale on K
v_j = rms(v_j)  (eps 1e-6, no gamma)           # use_value_norm_no_gamma
r_h = clip_{8.0}(s_r[h] ⊙ r_raw_h)             # s_r = rel_l2_clip.scale [24,16]
```
FULL layers only (log scaling with absolute position `p`, 0-based):
```
tau(p) = 1 + 0.1 * ln(max(1, (p + 1) / 8192))
q_h <- tau(p) q_h ,  r_h <- tau(p) r_h
```
tau is 1 for the first 8192 positions, about 1.069 at 16K, 1.208 at 64K, 1.485 at 1M (INFERRED arithmetic). Because both |q| and |k| are bounded by 22.627, `|q·k|/sqrt(128) <= 22.627^2/11.31 = 45.25` before tau.

### B.4 Relative-position bias

`P = rel_logits_proj.proj` [16, extent], extent = 1024 for every layer (sliding window 1024 for SWA, `rel_extent` 1024 for FULL). For query position `i`, key position `j`, distance `δ = i - j`:
```
b_h(i, j) = r_h(i) · P[:, δ]     if 0 <= δ < 1024
          = 0                    otherwise
```
So FULL layers see a learned, query-dependent bias for the last 1024 tokens and pure content attention beyond that. LC precomputes `r_h · P` (a 1024-entry table per head and token, padded by one zero column) and gathers by distance.

### B.5 Softmax with sink, gate, output

```
s_h(i, j) = q_h(i) · k_{h//3}(j) / sqrt(128) + b_h(i, j) + mask(i, j)
α_h(i, j) = exp(s_h(i,j)) / (exp(t_h(i)) + sum_j' exp(s_h(i,j')))     # sink logit t, zero value
o_h(i)    = sum_j α_h(i, j) v_{h//3}(j)
y         = W_o (g ⊙ concat_h o_h)                                    # gate from u_q
attn_out  = conv_post(y)                                              # attn_sconv, /sqrt(5)
```
Mask: causal for FULL; causal plus sliding window 1024 (`LLAMA_SWA_TYPE_STANDARD`) for SWA.

Layer types: SWA = 0-4, 6-10, 12-16, 18-22; FULL = 5, 11, 17, 23 (the last layer of each 6-layer block). In the 48-visit order there are 8 FULL visits: 5, 11 (x3), 17 (x3), 23.

---

## C. Attention residuals (block depth attention)

There is no running residual stream across the whole network. Layers are grouped in blocks of 6 (0-5, 6-11, 12-17, 18-23). State kept during the forward pass:

- `blocks`: list of completed block sums. `v_0` = normed token embedding. Then `v_1` (sum of layers 0-5), `v_2`, `v_3`.
- `h`: running partial sum of the current block (the "hidden state" passed from layer to layer).

Mixer, for a stack `S = [blocks..., partial]` and learned query `q` (one or three rows):
```
Ŝ_n   = rms(S_n)                     (eps 1e-6, no gamma; attn_res_normalize_values)
w_n   = softmax_n( q · Ŝ_n / sqrt(2304) )
mix   = sum_n w_n Ŝ_n                 (one output per query row)
```
Same in LC (`mix`), HF (`Inkling2LoopBlockAttentionResidual`, scaling `hidden_size**-0.5`), and GF's equation. The keys and the values are both the RMS-normalized states, so the mixer output always has the scale of a unit-RMS vector.

One decoder layer `ℓ` with input `h` (partial sum of its block, or the previous block's final sum if `ℓ` starts a block):
```
x_attn = mix(blocks + [h], attn_res_query_ℓ)       # skipped for layer 0 (x = h = v_0)
a      = Attention(x_attn)                         # B, including attn_post_conv
if ℓ % 6 == 0:  blocks.append(h); h' = a           # block boundary: h was the finished block sum
else:           h' = h + a
x_mlp  = mix(blocks + [h'], ffn_res_query_ℓ)
f      = MoE(RMSNorm_{mlp_norm}(x_mlp))           # E, including mlp_sconv
h_out  = h' + f
```
At a boundary layer the stack used for the attention input does NOT yet contain `h` as a block; `h` is the "partial" entry. After the push, `h` is a block and the new partial starts from the attention output.

Output: `final = mix(blocks + [h_out of layer 23], output_res_query)`, then RMSNorm. At that point `blocks = [v_0, v_1, v_2^(pass 3), v_3^(pass 3)]`, so the final mix sees 5 states (GF: "a depth-wise mix over v_0..v_4").

Counts: 19 single-row plus 4 three-row attention queries = 31 attention readers, 24 MLP readers, 1 output reader = 56 depth readers (matches GF).

---

## D. The loop

### D.1 Visit order (48 layer applications)

```
prelude : layers 0..5                    once     (cache slots 0..5)
pass 1  : layers 6..17                   (slots 6..17)
pass 2  : layers 6..17 with messages     (slots 24..35)
pass 3  : layers 6..17 with messages     (slots 36..47)
coda    : layers 18..23                  once     (slots 18..23)
```
LC slot for pass `p >= 1` and physical layer `ip`: `24 + (p-1)*12 + (ip-6)`. HF uses `cache_offset = 24 + (p-1)*12 - 6`.

Shared across passes: every weight of layers 6..17 (attention, convs, norms, router, experts, depth queries). Per pass: the KV cache, all four conv states, and the depth `blocks` list. Every pass restarts from the prelude output: `h = h_prelude`, `blocks = [v_0]` (LC `prelude_blocks`, HF `pre_loop_block_states`). Because block boundaries are at 6 and 12, layer 6 pushes `v_1` (= prelude output) and layer 12 pushes `v_2` within each pass.

### D.2 The message bank

At the start of pass `p >= 2`, from pass `p-1`:
```
bank = [ v_2^(p-1),  v_3^(p-1) ]
```
`v_2^(p-1)` = the block list entry pushed at layer 12 of the previous pass (LC: `blocks.begin() + prelude_blocks.size() + 1`, HF: `block_states[..., num_pre_loop_blocks + 1:]`), `v_3^(p-1)` = output of layer 17 of the previous pass. Only these two states cross passes. The bank is fixed for the whole pass.

### D.3 Messages

For core layer `ip = 6 + i`, with the four queries `u_Q = Qx[2i]`, `u_K = KVx[2i]`, `u_V = KVx[2i+1]`, `u_M = Qx[2i+1]` (`Qx` = `middle_cycle_cross_pass_queries`, `KVx` = `middle_cycle_cross_pass_kv_queries`; GGUF `loop_msg_query` rows 0..3):
```
m_s = sum_{n in {2,3}} softmax_n( u_s · rms(v_n^(p-1)) / sqrt(2304) ) rms(v_n^(p-1)),   s ∈ {Q, K, V, M}
```
Same `mix` function as C, over a two-entry stack.

### D.4 The four controllers (message gates)

For each controller `c ∈ {attn(=Q), k, v, mlp}` with `D_c` = `down_weight` [64, 4608], `F_c` = `forget_weight` [2304, 64], `G_c` = `gain_weight` [2304, 64]:
```
z     = D_c [ rms(x) ; rms(m) ]          # [64], eps 1e-6, unweighted rms on each half separately
x'    = (1 - tanh(F_c z)) ⊙ x + tanh(G_c z) ⊙ m
```
Keep factor `1 - tanh(Fz)` is in (0, 2); gain `tanh(Gz)` is in (-1, 1). HF computes this in float32 (forget/gain projections declared F32).

What `x` and `m` are:

| Controller | x (before) | m | After the gate |
|---|---|---|---|
| `attn` (Q) | `x_q` = depth-mixed attention input (FULL: query row 0; SWA: the single mix) | `m_Q` | norm (`attn_norm` or `q_inp_norm`), then Q, R, sink and output-gate projections |
| `k` | `x_k` (FULL: query row 1; SWA: same single mix as Q) | `m_K` | norm, then K projection and (FULL) dynamic K conv taps |
| `v` | `x_v` (FULL: query row 2; SWA: same single mix) | `m_V` | norm, then V projection and (FULL) dynamic V conv taps |
| `mlp` | `x_mlp` = depth-mixed MoE input | `m_M` | `mlp_norm`, then router and experts |

The gates touch only sublayer inputs; the partial sum `h` and the residual additions are unchanged. Pass 1 has no messages and uses no controller weights. All three implementations (LC, HF, MLX) agree on these equations; GF's appendix writes the same formula.

### D.5 What is cached per visit

Per cache slot (48): K and V (8 heads x 128 each; SWA slots keep at most 1024 positions), and a conv state of the last 3 inputs of each of the four convs (K channels 1024, V channels 1024, attention output 2304, MoE output 2304). LC packs them as one recurrent state of `n_sconv_channels = 2*1024 + 2*2304 = 6656` channels x 3, at channel offsets 0, 1024, 2048, 4352. The relative-bias table, messages and block sums are not cached (recomputed per step from current-token states).

---

## E. Router and experts

Input: `y = RMSNorm_{mlp_norm}(x_mlp)` (after the mlp controller in passes 2-3).
```
logits = W_r y                         # [128], computed in fp32 (HF router cast, LC expert_router_f32)
s      = sigmoid(logits)
idx    = top8( s + b )                 # b = mlp.gate.bias, used for selection only
w_e    = s_e / (sum_{e' in idx} s_e' + 1e-20) * 8.0       for e in idx
out    = sum_{e in idx} w_e * Expert_e(y)                  # accumulated in fp32 (HF)
f      = conv_post_mlp(out)                                # mlp_sconv, /sqrt(5)
```
`route_scale 8.0` multiplies the normalized weights, so the 8 weights sum to 8. No expert groups. No shared experts (`n_shared_experts 0`; LC converter rejects any other value).

Expert (softcapped SwiGLU, "SITU" in LC):
```
[gate; up] = W13_e y                 # each [1152]
a          = 4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(up / 25)
Expert_e(y) = W2_e a                 # [2304]
```
The sigmoid sees the raw gate; only the linear factor is capped at 4 and `up` at 25 (HF docstring and LC comment agree).

Router row norms: none of LC, HF or MLX normalizes the router rows or the input at routing time. If the trained rows are unit-norm, that is a property of the weights (perhaps baked in at export), and the logits are then cosine-like up to `|y|` (about sqrt(2304) times the norm weight). NOT CONFIRMED from code.

---

## F. Output

```
hfin   = mix([v_0, v_1, v_2^(3), v_3^(3), h_23], output_res_query)
z      = RMSNorm_{norm}(hfin)
logits = W_unembed z                 # 201024 rows, fp32 accumulation in LC
logits = 30 * tanh(logits / 30)
logits = logits[:200058]             # drop padded vocab
```
Embedding input: `v_0 = RMSNorm_{embed_norm}(E[token])`. Embedding and unembedding are separate tensors (HF `_tied_weights_keys = {}`).

Parameter totals (INFERRED arithmetic from shapes): experts 24 x 128 x 3 x 2304 x 1152 = 24.46 B; attention plus router about 27.2 M per layer (0.65 B); embed + unembed 0.93 B; controllers 28 M; total about 26.1 B physical (BLOG/VLLM: "26B"). Per token, 48 visits x (63.7 M expert + 27.2 M dense) = 4.36 B, plus 0.46 B unembedding. BLOG says "3B active", VLLM says "3.5B active"; neither matches a visit-weighted count, so treat the vendor numbers as a different counting convention (likely per physical layer, or excluding attention).

---

## G. Goodfire findings (GF)

1. Information-flow decomposition. Method: write each layer input as an exact sum of earlier sublayer outputs, `x' = Σ_w c_w` with `c_w = κ⊙α r o_w` (current loop) or `γ⊙β r o_w` (previous loop), and attribute share `s_w = c_w·x'/||x'||²` (shares sum to 1). Averaged over 13,233 tokens from 400 FineWeb documents.
2. Depth attention favours recent blocks, "akin to a sliding window" (same decomposition).
3. Cross-loop attention in passes 2 and 3 prefers block 3 over block 2; block-3 layers almost always read their own block from the previous loop (same decomposition; GF Figure 8 shows block-start layers of loops 2 and 3 strongly attending to block 3).
4. Reader count: 104 reader vectors (56 depth: 31 attention, 24 MoE, 1 output; 48 cross-loop: 12 layers x 4).
5. Writer-reader alignment. Method: absolute cosine between all 3,538,944 MoE down-projection columns (24 layers x 147,456 neurons) and the 104 readers, 368 M pairs; top 0.001% gives threshold |cos| = 0.30 and 3,490 pairs. Depth-reader writers concentrate in the last layer of each block (layers 6, 12, 18 in GF's 1-indexing): layer 6 writes mostly for block-4 readers, layer 12 for blocks 3 and 4, layer 18 for block 4. Interpreted as "tagging".
6. Cross-loop writers concentrate in block 3, especially its last layer, aligned with readers of layer 13 (first layer of block 3).
7. Massive outlier activations. Method: RMS-normalize activations of all 24 layers (squared norm = 2304, so a uniform coordinate has magnitude 1, 0.043% of the norm). Coordinate `d_2027` has mean normalized value +32.4, about 46% of the squared norm (32.4²/2304), the most anomalous coordinate.
8. `d_2027` surges at layer 18 (GF 1-indexed, i.e. 0-indexed layer 17, end of block 3) in loops 1 and 2, with a large positive spike; in loop 3 that layer writes a large negative value ("clean-up" before the coda). Method: per-component write attribution onto the coordinate over the same 13,233 tokens.
9. Reading `d_2027`. Method: for a two-block bank, `β_3 = σ(s_3 - s_2)` with `s_3 - s_2 = u·(v̂_3 - v̂_2)/sqrt(d)`; decompose the gap `u·δ = Σ_i u_i δ_i` per coordinate. For layer 13 in loop 2, `d_2027` alone contributes 17%, 19%, 11% and 6% of the gap for `u_Q, u_V, u_K, u_MoE` (largest or second-largest contributor). The top 5 coordinates contribute 60%, 62%, 32%, 30%; all 5 are among the top 15 coordinates by RMS-normalized magnitude.
10. Lenses. Method: top-50 inclusion of the model's output token and probability mass on the model's top 50, on 1,250 FineWeb documents (tuned lens trained on 23,750). Logit lens fails except in the last few layers of each loop; tuned lens and coda lens (run block 4 on all positions) work well; J-lens is evaluated but the metrics may not suit it. Layers from early loops decode better than mid-block layers of any loop.

GF's equations match the code (depth mix with RMS-normalized values and `1/sqrt(d)` scaling; message mix over `v_2, v_3`; `κ = 1 - tanh(Fz)`, `γ = tanh(Gz)`, `z = P[RMSNorm(x); RMSNorm(m)]`, `P ∈ R^{64x2d}`). GF writes `v_0 = RMSNorm(e_x)`; the code's embed norm has a learned weight.

---

## H. Unusual features that matter on memory-bandwidth-limited laptops

1. 48 visits over 24 stored layers. Decode reads the weights of layers 6..17 three times per token, so bytes moved per token track 48 layers while RAM holds 24. Weight memory is about half that of an equally deep model; tokens per second is not.
2. Each pass routes independently, so a core layer can touch up to 24 distinct experts per token (8 per pass). Expert reuse across passes would cut bandwidth only if selections overlap (untested here).
3. 48 KV caches and conv states (one per visit). The 8 FULL visits keep full-context KV: 8 x 2 x 1024 values = 16,384 values per token, 32 KiB/token at BF16 (about 4 GiB at 128K). The 40 SWA visits cap at 1024 tokens: about 160 MiB per sequence at BF16, constant.
4. No RoPE and a 1024-distance relative bias. Each step computes a per-head 16-dim state and a 1024-entry bias table per layer visit, plus a gather over keys. Small FLOPs, but it rules out stock fused attention kernels that assume RoPE and a scalar bias.
5. Four short convolutions per visit (two of them dynamic in FULL layers) add 6,656 x 3 floats of state per visit and serial dependencies, so prefill and decode cannot skip them; LC marks the arch hybrid (attention plus recurrent state) and disallows mixed batches.
6. Depth attention instead of a residual add: every sublayer input is a softmax over up to 5 RMS-normalized 2304-vectors that must stay resident per token (the block sums), and FULL layers do three such mixes. Cheap in FLOPs, but it is extra activation traffic and kernel launches per visit (LC `graph_nodes_per_tensor` 32).
7. Many small tensors are forced to F32 (conv kernels, clip scales, depth queries, relative-bias banks, loop controllers). They are small (controllers 28 M params total) and do not shrink with quantization.
8. Experts ship as NVFP4 (layers 0..19) and MXFP8 (20..23). The LC converter dequantizes them to F32 before requantizing, so a GGUF's expert precision is whatever the converter chose, not the native format; only MLX keeps the native NVFP4/MXFP8 scheme (BLOG).

---

## Disagreements and open points

- Active parameter count: BLOG 3B, VLLM 3.5B, visit-weighted arithmetic about 4.4B (plus unembedding).
- MXFP8 `weight_block_shape` in `hf_quant_config.json` is [32, 32]; stored scales imply [1, 32]. LC uses the stored shape.
- `d_2027` and layer numbers in GF are 1-indexed for layers; whether `d_2027` is 0- or 1-indexed is not stated.
- HF PR #51050 is still open; its code may change before merge.
- Router row unit-norm: not visible in any code path.
- AtomicChat GGUF tensor layout inferred from LD and a PR comment, not read from the file.
- MLX `language.py` was scanned for routing, softcap, conv scaling and controller equations (all match), not read line by line.
