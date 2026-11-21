#!/bin/zsh
# mp_run.sh TAG LAYERS RANK LAM SCHEME [FACTS] : fit one memory-patch configuration, then evaluate it (no document).
# The adapter goes to results/patch/<TAG>.lora.gguf, the fit statistics next to it, the evaluation to eval_<TAG>.json.
LAB=${INKLING_LAB:-$(cd "$(dirname "$0")/.." && pwd)}
tag=$1; layers=$2; rank=$3; lam=$4; scheme=$5; facts=${6:-}
out=$LAB/results/patch/$tag.lora.gguf
$LAB/scripts/mp_py $LAB/scripts/memory_patch.py fit --layers $layers --rank $rank --lam $lam --scheme $scheme --facts "$facts" --out $out > $LAB/results/patch/log_fit_$tag.txt 2>&1 || { echo "FIT FAILED $tag"; exit 1; }
$LAB/scripts/mp_py $LAB/scripts/eval_patch.py --tag $tag --lora $out --facts "$facts" > $LAB/results/patch/log_eval_$tag.txt 2>&1 || { echo "EVAL FAILED $tag"; exit 1; }
echo "$tag $(tail -1 $LAB/results/patch/log_eval_$tag.txt)"
