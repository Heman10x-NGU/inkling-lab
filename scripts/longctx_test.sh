#!/bin/zsh
# Usage: longctx_test.sh <prompt file> <tokens to keep via context> ; runs batch 64, 16, 512
LAB=${INKLING_LAB:-$(cd "$(dirname "$0")/.." && pwd)}
M=$LAB/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf; B=$LAB/runtime/llama.cpp/build/bin; P=$1; OUT=$LAB/results/longctx.txt
: > $OUT
for b in 64 16 512; do
  echo "=== batch $b start $(date +%H:%M:%S)" >> $OUT
  /usr/bin/time -l perl -e 'alarm shift; exec @ARGV' 1500 $B/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -n 64 -c 32768 -b $b -ub $b -f $P > $LAB/results/longctx_b$b.out 2> $LAB/results/longctx_b$b.err
  echo "exit $?" >> $OUT
  grep -E "prompt eval time| eval time|total time" $LAB/results/longctx_b$b.err >> $OUT
  grep -E "maximum resident set size|peak memory footprint" $LAB/results/longctx_b$b.err >> $OUT
  sysctl -n vm.swapusage >> $OUT
  echo "=== batch $b end $(date +%H:%M:%S)" >> $OUT
  sleep 60
done
echo DONE >> $OUT
