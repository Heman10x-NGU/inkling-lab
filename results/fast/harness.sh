#!/bin/zsh
# harness: gen <stock|fast> <prompt_idx> <outprefix> ; speed <n> <tag> ; text <tag> ; disp <stock|fast> <tag>
LAB=$LAB
R=$LAB/runtime/llama.cpp
M=$LAB/models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf
OUT=$LAB/results/fast
prompts=("def quicksort(arr):" "The history of the printing press begins in" $'Here are five tips for better sleep:\n1.' "Explain how a hash table works." "Dear Anna, thank you for your letter. Last week I")
bindir() { case $1 in stock) echo $R/build/bin;; prof) echo $R/build-prof/bin;; fast) echo $R/build-fast/bin;; *) echo $R/build-fast/snap-$1;; esac }
waitfree() { while pgrep '^[l]lama-' >/dev/null; do sleep 5; done }
# speed runs also wait until no other GPU job loop has been seen for 120 s
waitidle() { local c=0; while (( c < 12 )); do if pgrep -f '[l]lama-' >/dev/null; then c=0; else c=$((c+1)); fi; sleep 10; done }
gen() { # which idx prefix [n]
  waitfree
  DYLD_LIBRARY_PATH=$(bindir $1) $(bindir $1)/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n ${4:-64} -p "${prompts[$2]}" > $3.txt 2> $3.err < /dev/null
  grep -E "^\S*.*eval time" $3.err | grep -v prompt | sed -E 's/.*\( *([0-9.]+) ms per token.*/\1/'
}
case $1 in
  gen) shift; gen "$@";;
  text) # compare fast vs stored stock texts for 5 prompts
    for i in 1 2 3 4 5; do
      [[ -f $OUT/stock_p$i.txt ]] || gen stock $i $OUT/stock_p$i >/dev/null
      gen fast $i $OUT/$2_p$i >/dev/null
      if cmp -s $OUT/stock_p$i.txt $OUT/$2_p$i.txt; then echo "p$i identical"; else echo "p$i DIFFERS"; fi
    done;;
  speed) # n tag : interleaved stock/fast, 60 s rests; a pair whose stock run exceeds 60 ms/token is discarded and redone after 240 s
    k=0; tries=0
    while (( k < $2 && tries < 3*$2 )); do
      tries=$((tries+1))
      waitidle; s=$(gen stock 1 $OUT/spd_$3_s$tries)
      sleep 60; f=$(gen ${4:-fast} 1 $OUT/spd_$3_f$tries)
      if (( s > 60 || f > 60 )); then echo "try $tries stock $s fast $f DISCARD (hot)" | tee -a $OUT/speed_$3.log; sleep 240; continue; fi
      k=$((k+1)); echo "run $k stock $s fast $f" | tee -a $OUT/speed_$3.log
    done;;
  disp) # which tag : mode-1 dispatch count per generated token
    waitfree
    GGML_METAL_PROFILE=1 GGML_METAL_PROFILE_NODES_MAX=1 GGML_METAL_PROFILE_OUT=$OUT/disp_$3.json $(bindir $2)/llama-completion -m $M -ngl 99 -no-cnv --temp 0 -c 8192 -b 64 -ub 64 -n 6 -p "${prompts[1]}" > /dev/null 2>$OUT/disp_$3.err < /dev/null
    python3 -I -c "
import json,sys; d=json.load(open('$OUT/disp_$3.json')); g=[x for x in d['graphs'] if x['n_tokens']==1 and x.get('profiled')]
print('nodes',g[-1]['n_nodes'],'dispatch/token',[x['n_dispatch'] for x in g][-3:], 'gpu_ms', [round(x['gpu_us']/1000,1) for x in g][-3:])
ops=g[-1]['ops']; print(sorted([(k,v[0]) for k,v in ops.items()], key=lambda t:-t[1]))";;
esac
