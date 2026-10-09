# Milestones

## M0 — Define the compiler question

- [x] Choose a narrow initial model family.
- [x] Separate calibration measurements from formal guarantees.
- [x] Start a prior-art map and list known overlaps.

## M1 — Reference path

- [x] Specify the initial model text format.
- [x] Implement parser diagnostics and shape checks for Dense and ReLU.
- [x] Build a graph with explicit input/output widths.
- [x] Implement a straightforward double-precision reference evaluator.
- [x] Add a tiny example model.
- [x] Compile the first working slice and inspect its command-line output.

## M2 — Generate readable C++

- [x] Emit a standalone scalar C++20 program from the checked graph.
- [x] Compare generated output against the reference evaluator on the included example.
- [x] Record compiler flags, binary size, and repeatable timing procedure.

The generated example was built with LLVM-MinGW Clang using C++20 and `-O2`; the reference and generated programs both returned `3.0499999999999998` for input `1.0 2.0 3.0`.

## M3 — Precision planning and evidence

- [x] Add a software-emulated bfloat16 candidate with explicit rounding rules.
- [x] Compare its output against f64 for a supplied input.
- [x] Emit bfloat16 model constants and graph values in 16-bit storage.
- [x] Measure isolated per-layer cost on the target CPU (kept separate from whole-network latency).
- [x] Implement a small, deterministic exhaustive plan search over per-layer f64/bfloat16 choices.
- [x] Report maximum observed calibration drift separately from any proven bound.
- [x] Compare selected plans against uniform-precision baselines.
- [x] Generate readable C++ from a saved precision plan.

## M4 — Verified restricted domain

- [x] Define a box-constrained input domain.
- [x] Implement interval/error propagation for supported operations, with explicit IEEE/BF16 assumptions.
- [x] Validate the verifier against a dense grid on the tiny case (sampling is a sanity check, not the proof).
- [x] Emit an output-specific bound report and reject plans that exceed the requested bound.
- [x] Add adversarial checks for BF16 midpoint rounding, cancellation, ReLU boundary crossing, mixed plans, all-f64 plans, and BF16 overflow rejection.
- [ ] Obtain independent mathematical review before treating the bound as production-grade verification.

## M5 — Research review and later model support

- [x] Review direct prior art on sound mixed precision, quantized-network bounds, compiler code generation, and hardware cost.
- [x] Reject the broad certified-planner framing and document a narrower, unverified CPU reduction-scheduling question.
- [ ] Review SIMD reduction scheduling and error-bounded dot-product code generation for overlap with the revised question.
- [ ] Use measured whole-network and isolated per-layer costs in precision/reduction-schedule selection.
- [ ] Evaluate whether a transformer block is an appropriate next target.
- [ ] Add an assembly backend only when profiling identifies a concrete bottleneck.

## M6 — Maintainable compiler structure

- [x] Split the compiler into model/IR, evaluator/planner, certificate, code-generation, and command-line modules.
- [ ] Refine module boundaries further so parsing, evaluation, and precision search can be reviewed independently.
- [x] Keep the build and verification scripts reproducible after the split.

The post-split example, certificate-edge, and benchmark scripts all pass with the configured Clang compiler. CMake is not installed in the current environment, so the scripts compile the same source modules directly.

## M7 — Transformer operator foundation

- [x] Add RMSNorm to the model format, typed graph, f64 reference evaluator, and readable C++ backend.
- [x] Compare RMSNorm reference and generated output against a hand-calculated example, deterministic vectors, and large finite inputs.
- [x] Reject RMSNorm explicitly in bounded-domain certification until a sound bound is implemented.
- [x] Add stable vector Softmax to the model format, typed graph, f64 reference evaluator, and generated C++ backend.
- [x] Check Softmax against expected probabilities, extreme logits, and deterministic vectors; reject unsupported certification.
- [x] Add adjacent-pair RoPE at a fixed token position and inverse-frequency base.
- [x] Check RoPE against known rotations, a large position, and 16 deterministic vectors; reject certification until trig error is modeled.
- [x] Generalize layer inputs to an explicit input-edge vector and add a same-width residual Add from an earlier tensor.
- [x] Certify residual Add graphs and compare generated BF16-plus-residual output over a 27-point input grid.
- [x] Add `dense_from` so separate Dense projections can branch from any earlier tensor; compare a branched BF16 graph with the f64 reference and check a 9-point certificate grid.
- [x] Re-run the whole-network and isolated-layer benchmark after introducing branchable input edges.
- [ ] Add grouped-query attention and gated MLP operations, then exercise an end-to-end transformer block.
- [ ] Load a small public trained model and compare its outputs with a trusted runtime.

## M8 — Sequence-aware projection

- [x] Represent token-by-feature tensors with `input2d` and preserve dimensions through the typed graph.
- [x] Apply a shared Linear projection to each token row in the reference evaluator and generated backend.
- [x] Keep ReLU and residual Add correct for flattened rank-two storage.
- [x] Compare a two-token projection against a hand calculation in f64 and bfloat16.
- [x] Add fixed-length scaled dot-product attention over projected query, key, and value tensors.
- [x] Support causal and full masks and compare reference/generated f64 and bfloat16 outputs with hand calculations and deterministic samples.
- [x] Make bounded-domain certification explicitly reject rank-two attention until sound bounds are available.
- [x] Add multi-head attention and verify head slicing against a hand calculation and deterministic reference comparisons.
- [x] Extend RMSNorm and RoPE to sequence rows with per-token normalization and position offsets.
- [x] Add a bounded append-only K/V cache, use it in causal reference evaluation, and check multi-head stepwise decode by hand.
- [x] Generate a layer-specific causal attention session with persistent K/V state across calls.
- [x] Compute direct Q/K/V Linear projections inside the session at its selected precision.
- [ ] Feed token activations through arbitrary normalization, RoPE, residual, and projection prefixes inside that session.

## M9 — Fixed-length self-attention

- [x] Add rank-two scaled dot-product self-attention over named Q/K/V graph values, with configurable head count.
- [x] Implement causal and full masks in the reference evaluator and generated C++ backend.
- [x] Check hand-computed outputs and 16 deterministic inputs in f64 and BF16 projection modes.
- [x] Make the certificate refuse rank-two attention until its error analysis is implemented.
- [x] Add a repeatable attention latency, output-drift, binary-size, parameter-size, and fixed-array benchmark.
- [ ] Compare attention against oneDNN SDPA and an online-softmax implementation.
- [x] Extend the model to multiple heads.
- [ ] Load a public trained model and compare its outputs with a trusted runtime before claiming end-to-end LLM inference.
