# Runtime speed: end state, ceiling, and stop rules

Written 2026-11-21 after the Metal profile (`metal_profile.md`), before any optimization code.

## End state

A patch to llama.cpp's Inkling graph code (`src/models/inkling2.cpp` and its helpers) that produces the same output faster on a 16 GB MacBook Air M4, for the 7.9 GB AtomicChat file.

## Pass line

All of these, measured on a cool machine with runs interleaved (stock, patched, stock, patched), at least 5 runs each:

1. Generated text identical to stock at temperature 0 on 5 prompts.
2. `scripts/ppl.sh` against the stock baseline: mean KL divergence at most 0.0001 and same top word at least 99.9%.
3. Time per generated token at least 15% lower (stock is 52 to 54 ms; target 45 ms or less).
4. GPU dispatches per token under 4,500 (stock is 6,961).

## Measured ceilings (isolated GPU time per token, short history)

- Tiny kernels (5 µs or less each, 5,568 dispatches): about 19 ms.
- Projections that share an input (Q, gate, relative position, sink; K and V on sliding-window layers): 14.4 ms across 288 calls, expected saving 2 to 3 ms from merging.
- Expert matmuls: 15.3 ms. Kernel tuning; out of scope for this pass.

## Order of work

1. Depth mixers: stop rebuilding the stacked block states with CONCAT, REPEAT, and CONT on every use.
2. Relative-bias weight: pad once at load, not on every token.
3. Clip chain: reduce the 8-kernel clip to fewer operations using existing graph operations where possible.
4. Merge projections that share an input into one matmul.

## Stop rules

- If steps 1 to 3 together give under 8%, stop and report.
- Any step that changes the generated text is reverted unless the change is traced to floating-point reordering and passes line 2.

## Facts corrected by the profile

- Long history costs about 7 ms per token at 7,600 tokens. The earlier 7.8 tokens per second at 26,000 tokens was mostly heat.
- The GPU is busy 98% of wall time. Dispatch and encode overhead between kernels is small.
