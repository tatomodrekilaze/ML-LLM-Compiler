# Prior-art review

**Status: focused first review completed; the candidate direction is still unverified.** This review is intended to rule out broad claims and guide the next experiments. It is not an exhaustive patent search or a novelty determination.

## Compiler infrastructure and inference optimization

- [MLIR](https://mlir.llvm.org/) provides reusable compiler infrastructure with multiple abstraction levels. A custom intermediate representation alone is not a contribution.
- [TVM, OSDI 2018](https://www.usenix.org/system/files/osdi18-chen.pdf) describes an end-to-end deep-learning compiler with graph and operator optimizations across hardware back ends. Graph lowering and target portability are established.
- [HAWQ, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html), [HAWQ-V2, ICML 2020](https://proceedings.mlr.press/v119/yao20a.html), and [HAWQ-V3, ICML 2021](https://proceedings.mlr.press/v139/yao21a.html) cover sensitivity-guided or hardware-aware mixed-precision choices. HAWQ-V3 explicitly optimizes precision against constraints such as memory and latency and includes TVM deployment.
- [Constrained mixed-precision quantization, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Towards_Mixed-Precision_Quantization_of_Neural_Networks_via_Constrained_Optimization_ICCV_2021_paper.html) formulates precision allocation as a constrained optimization problem.

These works rule out automatic layer-precision choice under a quality, storage, or latency budget as a novel idea by itself.

## Sound precision tuning and code generation

- [Aster: Sound Mixed Fixed-Point Quantization of Neural Networks, EMSOFT 2023](https://doi.org/10.1145/3609118) is the closest direct overlap with Proofline's initial research question. Aster handles feed-forward networks with linear/ReLU operations, accepts input ranges and an output error limit, uses MILP-based mixed-precision fixed-point tuning, and generates C++ for HLS. Its evaluation includes FPGA synthesis cycle counts. Its fixed-point formats, FPGA target, and precision objective differ from Proofline's software-emulated BF16/f64 CPU backend, but the combination of bounded error, mixed precision, neural-network structure, code generation, and hardware cost has already been demonstrated.
- [FPTuner, POPL 2017](https://soarlab.org/papers/2017_popl_cbbsgr.pdf) tunes general real-valued expressions across single, double, and quadruple precision with an input-domain error guarantee and emits C++. Its [maintained artifact](https://github.com/soarlab/FPTuner) supports limiting casts and grouping expressions to encourage vectorization. Therefore, adding a sound error limit, generated code, and vectorization to a precision tuner is not by itself a defensible novelty claim.
- [Daisy, TACAS 2018](https://link.springer.com/chapter/10.1007/978-3-319-89960-2_15) combines sound roundoff analysis, mixed-precision tuning, rewriting, and code generation for numerical programs. It is not specific to an ML execution graph, but rules out presenting this broad tool combination as new.
- [Sound Mixed-Precision Optimization with Rewriting](https://people.mpi-sws.org/~eva/papers/iccps2018.pdf) likewise combines sound precision optimization and rewriting for performance. Rewrites under an error constraint are established work.

## Quantization-difference verification

- [QEBVerif, CAV 2023](https://doi.org/10.1007/978-3-031-37703-7_20) analyzes output differences between full and quantized networks over input regions, accounting for quantized weights and activations. It uses differential reachability analysis and MILP when needed. This is direct prior art for bounded output drift of quantized networks.
- [ReluDiff, ICSE 2020](https://chaowang-vt.github.io/pubDOC/PaulsenWW20_ICSE_ReluDiff.pdf) and [NeuroDiff, ASE 2020](https://engineering.purdue.edu/~wang6203/papers/neurodiff_ase2020.pdf) propagate differences between related networks to tighten verification bounds. Differential analysis is an established method family.
- [Scalable Verification of Quantized Neural Networks, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16496) discusses why treating low-precision networks as ideal real arithmetic can miss rounding-related failures. The target implementation's arithmetic semantics matter.
- [Quantifying the Impact of Quantization on Neural Networks](https://theory.stanford.edu/~barrett/pubs/HYW%2B24.pdf) also studies formal output differences between original and quantized networks over input regions.

This evidence means Proofline's current interval/error command is useful compiler engineering and a platform for experiments, not a new verification algorithm. It uses a deliberately simple absolute-error recurrence; it is not as tight or complete as the cited DRA/MILP methods.

## Transformer attention and execution schedules

- [Scaled dot-product attention in oneDNN](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa.html) documents the standard `softmax(QK^T / sqrt(d_k)) V` operation and a graph API that optimizes SDPA kernels. A basic attention operator and a graph representation for it are established compiler/library work.
- [FlashAttention, NeurIPS 2022](https://arxiv.org/abs/2205.14135) gives an exact tiled algorithm that reduces memory traffic by avoiding materialization of the full attention matrix and using online softmax. [FlashAttention-2](https://arxiv.org/abs/2307.08691) further studies work partitioning and parallelism. Fusing score, softmax, and value accumulation or using an online normalizer is not a novelty claim.
- [FlashSFA, ICLR 2026](https://openreview.net/pdf?id=UspMJlGusi) applies IO-aware kernels and online softmax to sparse feature attention. [STEEL, arXiv 2026](https://arxiv.org/abs/2607.09385) maps fused attention to an NPU and compares against CPU/GPU implementations. Attention optimization spans more than GPU-only work, and any CPU-specific hypothesis needs careful comparison with current tuned CPU libraries.

Proofline's `attention` operation is a correctness-first fixed-length reference path with configurable head count. It computes one query row at a time and stores only one score row per query/head, so it does not allocate a full sequence-by-sequence score matrix. That straightforward storage choice is not an algorithmic novelty claim; online softmax and fused exact attention have extensive prior art. The initial benchmark also found BF16 Q/K/V projections reduced fixed-array storage but were slower than f64 on the tested machine, which rules out assuming a smaller parameter format improves latency.

A possible next experiment is an ahead-of-time CPU planner that chooses among projection precision and exact attention schedules under both an output-drift bound and measured latency/scratch-memory limits. This combines known ingredients and remains unverified. It must be compared with oneDNN SDPA and an online-softmax implementation, with identical token/head dimensions, masking, compiler flags, and numerical checks. No novelty claim is warranted before that review and evaluation.

## Reduction order, SIMD, and accumulation precision

- [FPTuner, POPL 2017](https://soarlab.org/publications/2017_popl_cbbsgr/) explicitly allows grouping operations so they share a precision and can vectorize. Its paper also warns that compiler behavior can defeat expected performance improvements. A compiler that simply couples a precision choice to a vector-friendly group is therefore not a new direction.
- [Deterministic and Probabilistic Error Bounds for Floating Point Summation Algorithms](https://arxiv.org/abs/2107.01604) derives bounds for general, shifted, and compensated summation. [Precision-aware Deterministic and Probabilistic Error Bounds](https://arxiv.org/abs/2203.15928) extends the analysis to mixed-precision summation and observes that keeping intermediate partial sums small relative to their precision can help. Choosing a balanced or compensated tree under a basic error formula overlaps with established numerical analysis.
- [MGS: Markov Greedy Sums](https://arxiv.org/abs/2504.09072) orders neural-network dot-product terms by exponent to reduce swamping in low-bitwidth accumulation. It targets 8-bit arithmetic and specialized hardware, rather than Proofline's current CPU BF16/FP32 generated kernel, but it rules out treating exponent-aware term ordering for neural inference as a new idea.
- [Mixed precision accumulation for neural network inference guided by componentwise forward error analysis](https://arxiv.org/abs/2503.15568) estimates per-output inner-product condition numbers after a low-precision pass and recomputes sensitive components in higher precision. Dynamic output-wise fallback for neural inference is direct overlap with any candidate that adapts accumulation precision by component sensitivity.
- [ExBLAS: Reproducible and Accurate BLAS Library](https://www.nist.gov/system/files/documents/itl/ssd/is/NRE-2015-04-iakymchuk.pdf) and [ReproBLAS](https://digicoll.lib.berkeley.edu/record/133914?v=pdf) establish SIMD-capable expansion/superaccumulator approaches for reproducible and accurate reductions and dot products. Their guarantees and costs provide important baselines if Proofline explores accurate accumulation.

These sources materially narrow the candidate. A static CPU schedule chosen ahead of time from measured per-layer costs and checked against Proofline's emitted-arithmetic error model may still be a useful engineering experiment, but neither a changed reduction tree, SIMD grouping, componentwise mixed precision, nor exponent-sorted terms are novelty claims. The next experiment must compare against Clang's ordinary vectorization, pairwise or compensated summation, and a low-cost reference implementation. It must also distinguish any compiler contribution from the numerical algorithm itself.

## Revised research question

The original candidate—one inspectable CPU compiler that combines precision choice, bounded error, code generation, and measured cost—is too broad: Aster, FPTuner, Daisy, and QEBVerif already cover most of those ingredients in overlapping combinations.

A narrower hypothesis worth testing is:

> For BF16 Dense layers on commodity CPUs, can Proofline jointly select the number and structure of partial accumulators and each layer's f64/BF16 mode, using per-layer measured latency as cost and a bound that models the actual emitted reduction tree and conversions?

This is only a research direction. FPTuner's vectorization-aware expression grouping, Daisy's rewriting, established pairwise/compensated dot-product methods, and hardware-aware mixed precision may overlap with parts of it. The next prior-art pass must focus on reduction scheduling, SIMD dot products, and error-bounded code-generation before this is treated as a contribution. Any eventual result must be compared against Clang's default vectorization and at least one established accumulation algorithm.

## Review limits and next questions

This review covers selected primary papers and project artifacts; it does not establish that the revised question is novel. It also does not include patents, non-English work, or a systematic database search. Continue with these questions:

1. Which published compiler systems jointly search emitted CPU reduction schedules and measured cost under a formal per-output error budget, and what target and datatype do they support?
2. Does any tool combine the recent componentwise mixed-precision inference method with ahead-of-time target-specific schedule selection?
3. Can the current BF16 certificate be independently reviewed, then tightened with a differential abstraction instead of separate broad intervals?
4. Which public trained model and CPU baseline would make the eventual evaluation representative beyond a synthetic dense network?

## Review log

- 2026-10-08: Initial scan of MLIR, TVM, HAWQ family, constrained mixed-precision optimization, and quantization-difference analysis.
- 2026-10-08: Expanded review of Aster, FPTuner, Daisy, QEBVerif, ReluDiff, NeuroDiff, and bit-exact QNN verification. Rejected the broad “certified mixed-precision compiler plus cost report” framing and recorded a narrower, explicitly unverified reduction-scheduling hypothesis.
- 2026-10-08: Reviewed FPTuner's SIMD grouping, summation error-bound work, ExBLAS/ReproBLAS, MGS, and componentwise mixed-precision inference. Recorded their direct overlaps and narrowed the candidate; no novelty claim is established.
- 2026-10-09: Added fixed-length multi-head attention and reviewed oneDNN SDPA, FlashAttention/FlashAttention-2, FlashSFA, and STEEL. Marked the basic attention path and online-softmax/fused-attention schedules as established work; recorded a CPU schedule/precision planner as an unverified experiment that requires comparison against oneDNN and online softmax.
