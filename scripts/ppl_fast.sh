#!/bin/zsh
# ppl.sh base <split>            -> compute and store baseline probabilities for eval/<split>.txt
# ppl.sh run <split> [extra llama args...] -> paired comparison of a variant against that baseline
LAB=${INKLING_LAB:-$(cd "$(dirname "$0")/.." && pwd)}
M=${MODEL:-$LAB/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf}; B=$LAB/runtime/llama.cpp/build-fast/bin; mode=$1; split=$2; shift 2
base=$LAB/eval/base_$split.bin
if [[ $mode == base ]]; then
  $B/llama-perplexity -m $M -ngl 99 -c 512 -b 512 --chunks ${CHUNKS:-12} -f $LAB/eval/$split.txt --kl-divergence-base $base 2>&1 | grep -E "Final estimate|tokens per second|chunks" | tail -3
else
  $B/llama-perplexity -m $M -ngl 99 -c 512 -b 512 --kl-divergence --kl-divergence-base $base "$@" 2>&1 | grep -E "Mean PPL\(Q\)|Mean PPL\(base\)|ln\(PPL\(Q\)/PPL\(base\)\)|Mean +KLD|Same top p" | sed -E 's/^ +//' | tr '\n' ';' ; echo
fi
