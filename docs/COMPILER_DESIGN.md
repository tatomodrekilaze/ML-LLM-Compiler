# Compiler design

## Research question

For a feed-forward model and a declared input box, can Proofline select a mixed-precision execution plan that reduces measured CPU cost while keeping every output within a user-provided error budget—and produce a report that makes each decision checkable?

This question has two separate kinds of evidence:

1. **Measured evidence:** output drift on named calibration inputs, runtime on a named machine, and memory used by the generated program.
2. **Bounded evidence:** a sound worst-case output-error bound for a supported model and declared input domain. This is stronger and harder. It must not be inferred from a finite calibration set.

## Initial model boundary

The model language currently describes:

- A fixed-size input vector.
- A fixed token-by-feature matrix through `input2d <tokens> <features> f64`.
- Dense (matrix-vector) layers.
- Token-wise Linear projections that share weights across rows of a rank-two tensor.
- Fixed-length multi-head scaled dot-product self-attention over token rows, with causal or full masking.
- ReLU activations.
- RMSNorm with a fixed scale vector and positive epsilon (f64 execution).
- Softmax across the fixed-size vector (f64 execution).
- Adjacent-pair RoPE for one vector at a fixed integer position and base (f64 execution).
- Elementwise Add with an edge to a previously defined same-width vector (f64 execution).
- Explicit weights and biases.

The rank-two path supports Linear, ReLU, same-shape residual Add, feature-wise RMSNorm, position-aware RoPE, and fixed-length multi-head self-attention. Queries, keys, and values must have the same token count; query and key widths must match and divide evenly by the head count. Each head uses the inverse-square-root of its query width and a numerically stable softmax. The reference evaluator appends each causal token's K/V rows to a bounded `AttentionKvCache` and evaluates that query against the prefix. Generated C++ includes an `AttentionSession_<layer>` class with persistent cache state across `step(query, key, value)` calls. If each attention input comes directly from a Linear projection of the token input, the generator also provides `stepFromInput(token)` and emits those projections at the selected precision. The command-line program continues to evaluate the complete fixed-shape model. Standalone Softmax remains vector-only; projection through arbitrary preceding graph operations, grouped-query attention, and a complete transformer block remain future work.

## Pipeline

1. **Read a model:** parse the dependency-free `.proof` text format (`input <width> f64`, `dense <output-width> f64 <row-major weights> <biases>`, `dense_from <tensor-name> <output-width> f64 <weights> <biases>`, `relu`, `rmsnorm <epsilon> <scale values>`, `softmax`, `rope <position> <base>`, and `add <earlier-tensor-name>`).
2. **Build the graph:** represent values with names, element types, and known shapes. The file format has one model input and sequential shorthand for most operations; `dense_from` and Add can refer to earlier values, creating branches and residual edges in a topologically ordered DAG.
3. **Check it:** reject missing inputs, incompatible dimensions, unsupported types, malformed numbers, and non-finite constants.
4. **Evaluate a reference:** execute the graph in a simple, easy-to-audit double-precision implementation.
5. **Compare precision:** run a software-emulated bfloat16 path and report output error for supplied inputs. This is an observed result, not a bound.
6. **Plan precision:** exhaustively search per-Dense-layer f64/bfloat16 choices and minimize parameter payload subject to a maximum absolute calibration error limit.
7. **Generate CPU code:** emit readable standalone scalar C++20 from the checked graph.
8. **Report evidence:** include the selected plan, calibration results, any verified bounds, compile settings, and timings.

## RMSNorm arithmetic

The RMSNorm evaluator computes `x / sqrt(mean(x*x) + epsilon) * scale`. It avoids squaring the original input directly: first it finds `s = max(sqrt(epsilon), max(abs(x)))`, then accumulates `(x / s)^2` and rescales the resulting RMS. This scaled sum-of-squares keeps the intermediate sum finite for large finite inputs when the normalized output is finite. The generated C++ uses the same operation order as the reference implementation. Tests include a hand-calculated three-element case, deterministic inputs, and values of `1e308`.

Softmax subtracts the maximum input before applying `exp`, then divides by the exponential sum. This prevents exponential overflow for large positive logits. Differences between extreme finite inputs may underflow to negative infinity; their exponential is correctly zero. Softmax is applied to the entire current vector, not to a batch or selected tensor axis.

RoPE requires an even-width vector and applies one fixed position to adjacent pairs. For pair index `i` and vector width `d`, it uses `theta_i = position * base^(-2i/d)` and maps `(x[2i], x[2i+1])` to `(x[2i] cos(theta_i) - x[2i+1] sin(theta_i), x[2i] sin(theta_i) + x[2i+1] cos(theta_i))`. This is one common adjacent-pair convention; other RoPE layouts and frequency scaling variants are not represented. The current graph treats this vector as one token/head and has no sequence axis. Since the certificate does not model `pow`, `sin`, or `cos` library error, `certify` refuses graphs containing RoPE.

## Bounded-domain certificate

`certify` takes one finite lower/upper pair per input coordinate. It propagates outward-rounded intervals through the supported Dense/ReLU/Add graph. Separately, it propagates absolute error bounds for the f64 reference evaluator and the selected candidate, then adds those two bounds for each output. ReLU preserves the error bound because it is 1-Lipschitz. A Dense row propagates the incoming coordinate errors through the absolute weights and adds a roundoff allowance. The evaluator and code generator support RMSNorm, Softmax, and RoPE, but `certify` rejects these explicitly until sound bounds are implemented.

The Add rule uses outward-rounded interval addition. Its reference and candidate error bounds sum the incoming coordinate bounds and add a one-operation f64 roundoff allowance. Shared inputs may be correlated; the independent bounds ignore that correlation and can be loose, but remain conservative under the documented arithmetic assumptions.

For a sequence of `k` floating-point operations with unit roundoff `u`, the arithmetic allowance uses `gamma(k) = ku / (1 - ku)` times an upper bound on the sum of absolute terms, plus a conservative subnormal allowance. Fixed weights and biases use their actual BF16 encodings when calculating their perturbation. Runtime BF16 inputs include the conversion to float32 and a half-ULP bound based on the exponent range they can reach; BF16 outputs use the same range-aware half-ULP bound. Generated BF16 dot products are allowed to reassociate under Clang, so the bound covers up to two rounded operations per input term rather than assuming one fixed summation order. The verifier refuses a certificate if its interval cannot rule out a non-finite BF16 conversion or layer result.

The claim is conditional on IEEE binary arithmetic with round-to-nearest and gradual underflow, finite model values, and the generated backend's documented f32/BF16 operations. The outward interval operations use `nextafter` to expand each computed endpoint. The report's output intervals describe the real-valued graph on the closed input box; the per-coordinate error number bounds the distance between the compiled candidate and the f64 reference evaluator on that box. `certify` writes a report even when the requested limit fails, and returns a failing exit status in that case. `check-grid` compares the reference and candidate evaluators on a finite grid as an implementation sanity check only; it does not establish the bound.

The edge-case script covers a BF16 midpoint, cancellation-sensitive dot product, ReLU boundary crossing, mixed/all-f64 plans, and rejection of a weight that cannot convert to finite BF16. The example verifier also checks a BF16 Dense/residual Add graph against a 27-point grid. These checks do not replace independent mathematical review. Treat the certificate as an experimental bound implementation, not production-grade formal verification. RMSNorm, Softmax, and RoPE are not yet in the certificate's supported graph, and the existing Dense/ReLU/Add bound may be conservative.

## Candidate contribution to investigate

Compiler passes for mixed precision, sound error bounds, generated code, rewriting, and vectorization-aware precision tuning all have prior art. In particular, Aster and FPTuner overlap with the earlier broad project hypothesis. The revised question is whether a CPU compiler can jointly select BF16 Dense reduction structure and per-layer precision, using measured per-layer latency and a bound that covers the emitted reduction order. This is not a novelty claim; the [prior-art review](PRIOR_ART.md) lists known overlaps and the remaining search questions.

## Implementation principles

- Keep the reference evaluator simpler than the optimized backend.
- Keep parsing, graph checks, planning, verification, and code generation in separate modules.
- Use deterministic inputs and fixed benchmark instructions.
- Treat performance measurement noise explicitly; repeat runs and report spread.
- Add assembly only after generated C++ is correct and profiling identifies a specific kernel worth replacing.

## Current implementation

The current prototype implements model reading, typed DAG construction with explicit multi-input edges, vector Dense and rank-two token-wise Linear projections, ReLU, same-shape Add, sequence RMSNorm, position-aware sequence RoPE, and fixed-length multi-head scaled dot-product attention with causal or full masking. The reference evaluator and generated C++ support f64 and uniform or planned bfloat16 projections; attention accumulation and softmax use f64. For causal reference evaluation, `AttentionKvCache` appends each token's K/V rows and computes the current query over the cached prefix. Generated C++ includes an `AttentionSession_<layer>` class that retains K/V state across calls. For Q/K/V values produced directly by Linear projections from the input row, `stepFromInput` computes those projections using the selected precision plan. The CLI still accepts a complete fixed-shape tensor. Standalone Softmax remains vector-only. Certification is refused for unsupported operations and for rank-two inputs. Bfloat16 encoding first converts a value to float32, then rounds the float32 bit pattern to nearest-even bfloat16. Generated bfloat16 values use 16-bit words and projections use float32 products and accumulation. Before a generated BF16 projection, the backend converts the source tensor once and reuses it across output rows. Under Clang, it also allows reassociation in the dot-product loop so the compiler can vectorize it. That can change the order of floating-point additions from the sequential reference evaluator, so the certificate accounts for reassociated floating-point roundoff for its supported vector graphs. The planner compares every candidate plan with f64 on every calibration vector, uses the largest absolute difference across all output elements as its observed error, and minimizes parameter payload bytes. It currently supports at most 16 projections. The cost excludes activations, code, alignment, runtime, and RSS. The benchmark measures whole networks and isolated Dense layers; the separate attention benchmark reports latency, fixed-array estimates, and binary size. Neither set of measured per-layer costs is used by the planner. Independent certificate review, projection through arbitrary graph prefixes, a broader benchmark suite, and an end-to-end transformer remain outstanding.

## Precision experiment

For an input vector, `compare` evaluates the same model in f64 and in uniform bfloat16, then prints absolute and relative output differences plus logical tensor payload sizes. `plan` reads one vector per non-comment line and checks the maximum absolute output error over the entire file. These observed errors describe only the supplied vectors; they cannot prove an error limit for all inputs. The planner optimizes parameter payload, and the software emulation is not a claim that it runs faster than f64 on the current CPU.
