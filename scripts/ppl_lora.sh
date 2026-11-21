#!/bin/zsh
# ppl_lora.sh [adapter.gguf] : ppl.sh-style paired comparison against eval/base_tune.bin, using the llama-loops
# worktree build (inkscope branch), with the adapter loaded if one is given. Prints mean KL and "Same top p".
LAB=${INKLING_LAB:-$(cd "$(dirname "$0")/.." && pwd)}
M=$LAB/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf
B=$LAB/runtime/llama-loops/build/bin
extra=()
if [[ -n "$1" ]]; then extra=(--lora "$1"); fi
$B/llama-perplexity -m $M -ngl 99 -c 512 -b 512 --kl-divergence --kl-divergence-base $LAB/eval/base_tune.bin "${extra[@]}" 2>&1 \
  | grep -E "Mean PPL\(Q\)|Mean PPL\(base\)|Mean +KLD|Same top p" | sed -E 's/^ +//' | tr '\n' ';'; echo
