# Volume correction: end state, predictions, and stop rules

Written 2026-11-21 before any run.

## End state we are working back from

A published file: the 7.9 GB Inkling 2 Mini build at the same size, plus the script that produced it and a before/after table, with this one-sentence mechanism: "2-bit quantization makes the specialists quieter than attention; restoring their volume recovers quality."

The release is worth making only if both hold:

1. Held-out text: perplexity at least 2% lower than the stock file, with the paired difference more than 3 standard errors from zero.
2. Coding test (164 problems, executable checks): pass rate higher than the stock file by at least 4 points.

If only (1) holds, publish the finding as a note, not as a model release.

## Chain from here to there

1. Instrument: one command, edited setting in, paired perplexity difference out, under 5 minutes.
2. One global number (`expert_weights_scale`, stock value 8.0).
3. Per-layer and per-matrix scales, written into the file's block scales.
4. Coding test on the best file.
5. Replication on the 1-bit and a 4-bit file.

Each step runs only if the one before it passed.

## Predictions for step 2

- Measured: gate and up matrices in layer 8 agree with the official weights at cosine 0.94. The quantizer sets block scales by least squares (`scale = sumqx/sumq2` in `ggml-quants.c`), which shrinks norms by that cosine.
- Estimated: three matrices in series give about 0.94^3 = 0.83 of the original output. The down matrix is not measured, and the gate passes through a nonlinearity, so 0.83 is rough.
- Predicted best value of the global number: between 8.5 and 10.0, most likely near 9.6.
- Predicted effect at the best value: perplexity 1% to 5% lower on the tuning split.
- Control that should fail: 7.0 (wrong direction) gives higher perplexity than 8.0.
- Identity check: 8.0 through the override path gives zero difference.

## Stop rules

- If the best value is 8.0 within noise, the hypothesis is wrong for this file. Stop and record it.
- If 7.0 is also better than 8.0, the instrument or the mechanism is wrong. Stop and investigate before any further tuning.
- Tune on `eval/tune.txt` only. Report on `eval/test.txt` once, after the value is fixed.
