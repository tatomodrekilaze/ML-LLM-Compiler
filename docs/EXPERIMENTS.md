# Reproducible example results

This page records numeric correctness for the tiny example and a separate generated-kernel benchmark on a larger synthetic model.

## Environment

- Compiler: Clang 23.1.3
- Target: `x86_64-w64-windows-gnu`
- Flags for the compiler and generated programs: `-std=c++20 -Wall -Wextra -Wpedantic -O2`
- Model: [`examples/tiny.proof`](../examples/tiny.proof)
- Input: `1 2 3`

Run `scripts/verify_example.ps1` with Clang available on `PATH` to rebuild Proofline and the generated precision variants, then repeat the tiny-model comparison. Run `scripts/benchmark_example.ps1` for the generated-kernel timing procedure.

## Results

| Path | Output |
|---|---:|
| f64 reference | 3.0499999999999998 |
| generated f64 C++ | 3.0499999999999998 |
| bfloat16 reference | 3.046875 |
| generated bfloat16 C++ | 3.046875 |

For this one input, bfloat16 absolute output error is `0.0031249999999998224`, or `0.001024590163934368` relative to the nonzero f64 output (about `0.10246%`). This single input does not establish a worst-case error bound.

The graph has 11 model parameters and 8 graph values (the input and each layer result). Counting only those tensor elements gives 152 bytes at 8 bytes per f64 value and 38 bytes at 2 bytes per bfloat16 value. This is a fourfold reduction in tensor payload for this example. It excludes code, stack/runtime overhead, alignment, and process RSS.

## Calibration planner

The included four-vector calibration file is [`examples/tiny.cal`](../examples/tiny.cal). Running

```text
proofline plan examples/tiny.proof examples/tiny.cal 0.01 tiny.plan
```

examines all four assignments for the two Dense layers. It selects bfloat16 for both layers, observes maximum absolute output error `0.0062500000000000888` across those four vectors, and reports 22 bytes of Dense parameter payload (88 bytes for all-f64). The generated program from that saved plan matched the reference plan output on every calibration vector in the verification script.

The chosen plan is specific to this calibration file and tolerance. The planner does not prove behavior outside those vectors, and its cost target counts parameter storage only.

## Bounded-domain certificate sanity check

[`examples/tiny.domain`](../examples/tiny.domain) declares each of the three input coordinates to lie in `[-0.1, 0.1]`. For the all-bfloat16 plan, the verifier reported a maximum output error bound of `0.0049968168521315184`. An 11-point-per-coordinate grid (1,331 inputs) found a maximum sampled error of `0.00085937499999999556` in both the reference evaluator comparison and the emitted C++ program comparison, below the certificate. The verifier accepted a requested limit of `0.1` and wrote a `FAIL` report and nonzero exit status for a requested limit of `0.001`.

This grid only checks the implementation against sampled points. It is not evidence of soundness by itself. The bound uses outward-rounded intervals and floating-point roundoff propagation described in [the certificate design](COMPILER_DESIGN.md#bounded-domain-certificate); that derivation still needs independent review.

## Certificate edge cases

Run `scripts/verify_certificate_edges.ps1` to compile generated C++ and check targeted samples plus small grids for additional cases:

| Case | What it targets | Bound | Largest observed generated-code error |
|---|---|---:|---:|
| BF16 midpoint | Tie-to-even boundary and singleton input interval | 0.0097810630686725863 | 0.00390625 |
| Cancellation | Large positive/negative dot-product terms and reassociation | 1.5128124786408517e18 | 5.44579703015473e17 targeted; 6.4597000328072397e17 on the 7-point grid |
| ReLU crossing | Dense range straddles zero before ReLU | 0.0097659766907491341 | 0 on the selected generated-code points |
| Mixed f64/BF16 | f64 activation enters a BF16 layer | 0.0019532071892352765 | 0.000390625 on selected points |
| All f64 | Baseline generated arithmetic | 1.0769163338864049e-15 | 0 on selected points |

The cancellation result exposes a practical weakness: even after using the exact encoded constants, the independent-interval error recurrence remains loose for cancellation-heavy inputs. The overflow fixture uses a finite `1e100` f64 weight and confirms that the verifier refuses to certify its BF16 conversion. These edge checks strengthen implementation confidence but are not an independent proof of the analysis.

## Limits

The tiny correctness model is too small for meaningful timing. No speedup or process-memory reduction is inferred from the tiny-model payload table. The larger generated-kernel results below are a separate, repeatable measurement.

## Generated-kernel timing

The benchmark harness generates a deterministic width-96, three-Dense-layer model with ReLU between layers, plus three one-layer models carrying the same Dense weights and biases. It builds f64 and bfloat16 executables for the whole network and each isolated layer using the same compiler and flags, warms each executable once, then alternates their run order across seven measurements. Each measurement times 1,000 inference iterations inside the generated program with a changing first input element and a volatile checksum so the compiler cannot discard the work. Model parsing, process startup, and constant-array initialization happen before the timed loop.

Six runs on this session after hoisting bfloat16 input conversion out of the inner dot-product loop and enabling Clang's reassociation for that loop produced:

| Run | f64 median | bfloat16 median | bfloat16 / f64 | f64 executable | bfloat16 executable |
|---|---:|---:|---:|---:|---:|
| 1 | 8,642.1 ns/inference | 3,176.4 ns/inference | 2.72× faster | 296,448 bytes | 131,584 bytes |
| 2 | 9,371.0 ns/inference | 3,219.4 ns/inference | 2.91× faster | 296,448 bytes | 131,584 bytes |
| 3 | 8,448.2 ns/inference | 3,186.6 ns/inference | 2.65× faster | 296,448 bytes | 131,584 bytes |
| 4 | 8,804.1 ns/inference | 3,142.5 ns/inference | 2.80× faster | 296,448 bytes | 131,584 bytes |
| 5 | 9,223.0 ns/inference | 3,724.1 ns/inference | 2.48× faster | 296,448 bytes | 131,584 bytes |
| 6 | 8,745.8 ns/inference | 3,179.1 ns/inference | 2.75× faster | 296,448 bytes | 131,584 bytes |
| 7 (after source split) | 8,491.8 ns/inference | 3,188.5 ns/inference | 2.66× faster | 296,448 bytes | 131,584 bytes |
| 8 (after branchable IR) | 7,877.6 ns/inference | 2,904.4 ns/inference | 2.71× faster | 296,448 bytes | 131,584 bytes |
| 9 (after shared projection codegen) | 8,840.3 ns/inference | 3,063.6 ns/inference | 2.89× faster | 296,448 bytes | 131,584 bytes |

The same harness compiles each of the three Dense layers as a separate one-layer model and measures that generated kernel with the same iteration and repetition counts. On run 6, isolated medians were:

| Dense layer | f64 | bfloat16 |
|---:|---:|---:|
| 0 | 2,832.0 ns/inference | 1,156.0 ns/inference |
| 1 | 2,943.8 ns/inference | 1,157.7 ns/inference |
| 2 | 2,929.3 ns/inference | 1,157.4 ns/inference |

Run 7's isolated medians were 2,853.5 / 1,057.1 ns for layer 0, 2,909.0 / 1,058.4 ns for layer 1, and 2,845.9 / 1,053.5 ns for layer 2 (f64 / bfloat16). The source split did not change executable sizes; timings vary slightly between runs.

Run 8's isolated medians were 2,890.0 / 1,058.9 ns for layer 0, 2,651.9 / 1,062.2 ns for layer 1, and 2,832.0 / 1,050.6 ns for layer 2 (f64 / bfloat16). The branchable IR did not change the generated Dense kernels; this run is a fresh measurement, and the ordinary run-to-run variation remains visible.

Run 9's isolated medians were 2,669.8 / 1,084.7 ns for layer 0, 2,677.8 / 1,061.5 ns for layer 1, and 2,760.2 / 1,047.1 ns for layer 2 (f64 / bfloat16). This run followed the code-generation change that unified vector Dense and token-wise Linear projections; the vector benchmark remains within the earlier run-to-run range.

These are isolated-kernel costs. Their sum need not equal whole-network latency because the compiler can optimize across layers and cache behavior differs. The planner does not yet use these measurements.

Toolchain: Clang 23.1.3, target `x86_64-w64-windows-gnu`, flags `-std=c++20 -Wall -Wextra -Wpedantic -O2`. The available processor identifier is `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel` (12 logical processors reported); this session could not provide a CPU marketing name. On the fixed benchmark input, the maximum absolute output difference was `0.000123573467135429`. The model-plus-activation tensor payload was 228,096 bytes for f64 and 57,024 bytes for bfloat16.

Clang's optimization remarks confirmed that the three inner bfloat16 dot-product loops vectorized at width 4 with interleave count 2. The current generated backend converts each layer's input vector to bfloat16 once, then reuses those values in the dot product. The Clang-only `reassociate` directive permits a different accumulation order from the reference evaluator; this is a deliberate speed/rounding tradeoff, and generated outputs are checked against the reference for the fixed benchmark input. The verification script also checks 64 deterministic random inputs for the tiny model. These finite checks do not prove correctness or an error bound for every possible input.

The measurements show this generated bfloat16 kernel was faster on this machine for this synthetic model. They time the inference loop, not model loading or end-to-end application latency. The benchmark is small and single-machine; it is not a general performance claim. No process-RSS reduction is claimed.

## Two-token Linear projection

`examples/two_token.proof` uses a rank-two input with two token rows and two features per row. Its shared Linear layer applies the matrix `[2, 0; 0, 3]` and bias `[0, 1]`, then ReLU and a residual add restore the original input. For input `1 2 3 4`, the expected output is `3 9 9 17`. The verification script checks this result in the reference evaluator, generated f64 code, generated bfloat16 code, and a calibration-selected plan.

The included calibration file has two flattened token rows. The plan search checked two assignments, observed zero output difference on those samples, and selected bfloat16 projection weights with a 12-byte parameter payload. This validates the matrix shape and token-wise projection path; it does not measure large-sequence performance or implement attention.

## Fixed-length attention

`examples/causal_attention.proof` and `examples/full_attention.proof` use identity projections for query, key, and value. With input rows `[1, 0]` and `[0, 1]`, the scaled nonzero score is `1 / sqrt(2)`. Causal attention returns `[1, 0]` for the first token and `[1 - p, p]` for the second, where `p = 1 / (1 + exp(-1 / sqrt(2)))`. Full attention returns `[p, 1 - p]` for the first token and `[1 - p, p]` for the second. The verifier script checks these values against both generated precisions and compares generated output with the reference on 16 deterministic inputs per mask and precision.

For the causal fixture, the calibration planner checked all eight f64/bfloat16 assignments for the three projections, observed zero output error on its two supplied samples, and selected bfloat16 parameters totaling 36 bytes. Attention arithmetic currently stays in f64. These measurements validate this small fixture only; no attention speed claim or sound rank-two certificate is made. The certificate command explicitly refuses rank-two attention models.

The repeatable `scripts/benchmark_attention.ps1` harness measures a deterministic causal model with 16 tokens and 32 features, 100 in-process inferences per measurement, and seven repetitions. On this Windows session with Clang 23.1.3 and the reported `Intel64 Family 6 Model 154 Stepping 4` processor identifier, the latest median latency was 14,790 ns for f64 and 26,018 ns for BF16 projections. Generated-code mismatch against the cache-backed reference was zero for both; the BF16 path's maximum output drift from f64 on the fixed input was `6.103515625e-05`. BF16 reduced estimated fixed-array storage from 50,048 to 21,824 bytes and executable size from 100,352 to 82,432 bytes, while reducing parameter storage from 25,344 to 6,336 bytes. The current BF16 attention model ran about 1.76× slower, so these results show a storage/latency tradeoff rather than a speedup. Fixed-array estimates include projection parameters, graph tensors, the saved input, and attention score scratch; they exclude executable/runtime overhead and stack padding. This is one synthetic workload on one machine, not a general performance claim.

`examples/two_head_attention.proof` splits four features into two independent two-feature heads. The verifier checks the causal output against a hand calculation and compares generated f64 and BF16-projection programs with the reference on 16 deterministic inputs each. This exercises head slicing and output placement; it is not a comparison against an independent attention library.

`src/attention_cache.cpp` implements a bounded append-only K/V cache, and the causal reference evaluator now appends one token at a time before computing that token's output. The standalone cache check validates incremental outputs for a two-head example and rejects cache overflow and incorrect query width. Generated causal attention code includes a layer-specific `AttentionSession_<layer>` class with reserved heap storage. When Q/K/V each come directly from a Linear projection of the model input, `stepFromInput` runs those projections using the selected f64 or BF16 plan; otherwise callers can use `step` with projected rows. Generated-code harnesses checked two calls, state retention, reset, and overflow for both precision plans.

`examples/sequence_norm_rope.proof` checks that RMSNorm works independently on each token row and that RoPE advances the position for each row. The generated f64 program matched a hand-calculated two-row result and the reference evaluator on 16 deterministic samples. This covers operator semantics, not a full Transformer block or KV-cache decode.
