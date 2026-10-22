# Negative results

Quantization and acceleration routes we took to a verdict, with the reason
each died. Kept so consumers don't re-dig the same graves; root causes are
measurement-methodology lessons as much as performance ones.

## INT8 / W8A8 routes (all closed)

| Route | Verdict | Root cause |
|---|---|---|
| Text W8A8 (per-channel) | closed | best activation cosine 0.958; `input_scale=1/127` root cause; dynamic quantization worse (0.5-0.86) |
| Vision INT8 | closed | no safe sub-boundary: qkv-only first-passed the offline gate but closed-loop dropped to 75% vs 90% same-day BF16 |
| Hadamard W8A8 | closed | convention pairing errors (weight-side H/sqrt(d) vs unnormalized kernel FWHT) mimic numerics bugs; gate checks that compare like-for-like miss pure-gain mismatches |
| INT8 tensor-core GEMM | closed | prefill GEMM ran at ~10% of int8 peak (structurally HBM/issue-bound on sm_87); the CUTLASS sm_87 probe later showed 50 TF (73% peak) is achievable, but plugin integration stalled at +5% e2e — stopped per the loss-avoidance rule |

## Frozen-engine constraints

| Route | Verdict | Root cause |
|---|---|---|
| Plugin into sealed engines | closed | QDQ-form frozen engines carry no plugin nodes; a plugin retrofit requires re-export, not in-place surgery |
| Co-resident TRT engines | closed for mixed workloads | vision + text engines in one process showed out-of-bounds writes on small tensors (heisenbug, per-engine clean); chain_runner hard-rejects co-residency |

## Methodology takeaways

- Gate metrics must be paired to the deployment objective. An arange-vs-arange
  rope comparison measured nothing about true mrope; a cos gate without an
  absmax companion is blind to pure-gain errors.
- Numerical parity lanes and task metrics can disagree; when they do, trust
  the task metric for go/no-go and keep the parity lane open as a signal, not
  a verdict.
- Quantization damage carriers are layer-specific (here: encoder MLP vs
  attention projections vs decoder); "vision INT8" is not one experiment.
- Fix noise floors before interpreting small effects: re-run noise
  (0.012 here) bound what a K16-vs-K8-style lever could ever show.
