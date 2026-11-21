#!/bin/zsh
# r2_run.sh TAG LAMBDA [RANK] [LAYERS] : round-2 fit (10 wordings per fact, chat C0, C0-weighted rank cut,
# sequential re-capture) and evaluation. Work dir: $MP_WORK (round-2 captures).
LAB=${INKLING_LAB:-$(cd "$(dirname "$0")/.." && pwd)}
tag=$1; lam=$2; rank=${3:-64}; layers=${4:-0-23}
out=$LAB/results/patch/$tag.lora.gguf
$LAB/scripts/mp_py $LAB/scripts/memory_patch.py fit --layers $layers --rank $rank --lam $lam --scheme seq --cut c0 --out $out > $LAB/results/patch/log_fit_$tag.txt 2>&1 || { echo "FIT FAILED $tag"; exit 1; }
$LAB/scripts/mp_py $LAB/scripts/eval_r2.py --tag $tag --lora $out > $LAB/results/patch/log_eval_$tag.txt 2>&1 || { echo "EVAL FAILED $tag"; exit 1; }
echo "$tag $(tail -1 $LAB/results/patch/log_eval_$tag.txt)"
