# inkscope: the debugger core

`llama-inkscope` loads the model once, runs every prompt in a JSONL file, and for each prompt writes the activations you name, the top-5 next tokens, and optionally a greedy continuation. It can run with LoRA adapters loaded. Source: `runtime/llama-loops/tools/inkscope/inkscope.cpp` (branch `inkscope`), patch against 6196d98 in `patches/inkscope.patch`.

## Build

```sh
cd /path/to/Inkling-Lab/runtime/llama-loops
uvx cmake -B build
uvx cmake --build build -j 6 --target llama-inkscope
```

## Run

```sh
LAB=/path/to/Inkling-Lab
$LAB/runtime/llama-loops/build/bin/llama-inkscope \
  -m $LAB/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf -ngl 99 -c 2048 -b 2048 -ub 512 \
  --prompts prompts.jsonl --out outdir \
  --names '^(wo_inp-[0-9]+|result_norm)$' --last-n 40 --generate 40 \
  --lora adapter.gguf
```

- `prompts.jsonl`: one `{"id": "...", "prompt": "..."}` per line. Special tokens such as `<|message_user|>` are parsed.
- `--names REGEX`: tensors whose graph name matches are written (default `^result_norm$`). Anchor the regex: names with a ` (reshaped)` suffix are separate tensors.
- `--last-n K`: for 2-D tensors (`[features, tokens]`), keep only the last K token columns. 0 keeps all.
- `--generate N`: greedy continuation of up to N tokens, stopping at `<|end_message|>`. Nothing is captured during generation.
- `--lora FILE` (repeatable, or comma list) and `--lora-scaled FILE:SCALE` come from the common argument parser.
- The memory is cleared before each prompt; each prompt is decoded in one batch, so keep prompts under the ubatch size (512 here) or a tensor is written once per ubatch.
- `outdir` must exist. Per prompt: `<id>.bin` in the INKSCOPE_DUMP format (64-byte name, int64 type, ne0..ne3, nbytes, raw data; read with `scripts/inkscope_read.py`) and `<id>.json` with `tokens`, `n_records`, `top5` (id, piece, logit after the 30 softcap, probability), and `generated` / `gen_tokens`.

## Capture names (per visit `il`, 0..47)

`wo_inp-<il>` (new on this branch): input of the attention output projection, the gated head outputs `attn * sigmoid(gate(x))`, shape `[3072, tokens]`. Feature j is input column j of `blk.L.attn_output.weight` (ggml `ne = [3072, 2304]`, numpy `(2304, 3072)`). Also `l_out-<il>`, `attn_out-<il>` (after the output projection and its short convolution: `attn_out = (y + conv4(y))/sqrt(5)`, `y = W_o wo_inp`), `ffn_moe_topk-<il>`, `pass_block-<il>`, `result_norm` (only output rows, normally the last token).

Visit to physical layer: 0..5 are layers 0..5; 6..41 are layers 6..17 three times; 42..47 are layers 18..23.

## Sanity checks (run 2026-11-21)

```sh
scripts/mp_py scripts/inkscope_sanity.py <outdir> <id>     # result_norm x dequantized output.weight vs the tool's top-5
scripts/mp_py scripts/inkscope_wo_check.py <outdir> <id>   # conv(W_o wo_inp) vs attn_out (needs wo_inp and attn_out captured with --last-n 0)
```

See `memory_patch_results.md` for the numbers.
