# qwen-drive-edge-orin

Edge deployment of [**Qwen-Drive-1.0-4B**](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) on Jetson AGX Orin: **1026 ms** end-to-end. The 1-step trt planner runs at **35.7 ms** (33× faster than the 10-step rollout) with only **+2.48% ADE5(m)**.

## Open-loop replays

Six of the 94 manifest scenes: day/night, straight/turn/lane-change. Left: BEV; right: front camera. One plan at clip start, scored over the next 5 s — the protocol behind every ADE5/FDE5 below.

<p align="center">
<table>
  <tr>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/replay_chunk0_0003.mp4" width="100%" controls muted></video><br><sub>suburban, straight · ADE5 0.57m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/replay_chunk0_0012.mp4" width="100%" controls muted></video><br><sub>signalized intersection, straight · ADE5 0.70m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/replay_chunk0_0022.mp4" width="100%" controls muted></video><br><sub>urban arterial, straight · ADE5 1.44m</sub></td>
  </tr>
  <tr>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/replay_chunk0_0088.mp4" width="100%" controls muted></video><br><sub>night intersection, straight · ADE5 1.73m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/replay_chunk0_0033.mp4" width="100%" controls muted></video><br><sub>intersection, right turn · ADE5 2.62m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/replay_chunk0_0028.mp4" width="100%" controls muted></video><br><sub>highway, <font color="red">lane change missed</font> · ADE5 4.79m</sub></td>
  </tr>
</table>
</p>

*ADE5 = production 1-step engine vs recorded ground truth over 5 s; legend names match the Results arms ([docs/results.md](docs/results.md)).*

## Rolling replans

Deployment replans every cycle: the certified production chain re-run every 0.5 s over the same 5 s window — 10 replans per scene, each on fresh images, history, and the recorded vehicle state.

<p align="center">
<table>
  <tr>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/rolling_chunk0_0003.mp4" width="100%" controls muted></video><br><sub>suburban, straight · ADE5 0.57m → executed 0.13m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/rolling_chunk0_0012.mp4" width="100%" controls muted></video><br><sub>signalized intersection, straight · ADE5 0.70m → executed 0.17m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/rolling_chunk0_0022.mp4" width="100%" controls muted></video><br><sub>urban arterial, straight · ADE5 1.44m → executed 0.16m</sub></td>
  </tr>
  <tr>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/rolling_chunk0_0088.mp4" width="100%" controls muted></video><br><sub>night intersection, straight · ADE5 1.73m → executed 0.16m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/rolling_chunk0_0033.mp4" width="100%" controls muted></video><br><sub>intersection, right turn · ADE5 2.62m → executed 0.24m</sub></td>
    <td align="center"><video src="https://github.com/hous-lab/qwen-drive-edge-orin/releases/download/v1.0.1/rolling_chunk0_0028.mp4" width="100%" controls muted></video><br><sub>highway, <font color="red">lane change missed</font> · ADE5 4.79m → executed 0.12m</sub></td>
  </tr>
</table>
</p>

Per-plan ADE5 does not improve with replanning — same model at every replan (scene 0033: 1.03–5.08 m across its 10 replans). What replanning buys is execution: stitching the first 0.5 s of each plan tracks ground truth at 0.12–0.24 m mean, ≤0.50 m max — the worst one-shot scene (0028, 4.79 m) tracks at 0.12 m. **Replanning does not make the model a better forecaster; it makes the executed path accurate.**

*Closed-loop proxy: replans condition on the recorded vehicle state — no compounding drift captured; the k=0 replan is bitwise-identical to the certified record.*

## Results

Jetson AGX Orin 64GB, MAXN, locked clocks, single stream, batch 1; means over one 94-scene pass (pre/post-processing included). Evidence chain: [docs/results.md](docs/results.md).

| Stage | Implementation | Latency |
|---|---|---|
| Process (tokenize + glue) | torch | 208.2 ms |
| Prefill (vision + text) | vision TRT fp16 engine + text torch bf16 eager | 782.0 ms |
| Planner | TRT bf16 one-step engine, dynamic-S | 35.7 ms |
| End to end | full chain | 1026.0 ms |

Text prefill runs eagerly in bf16: the text and planner TRT engines cannot share one process ([docs/results.md](docs/results.md)).

Baseline (`QD_FAST_LANE=0`): planner via native 10-step eager rollout — 2191.5 ms end to end (planner 1162.7 ms), all gates PASS.

![stage-by-stage latency waterfall of the two lanes](assets/benchmarks/latency_waterfall.png)
*The 10-step rollout costs 1.16 s of eager sampler calls; one engine call collapses it 33×.*

One engine call serves every scene (29.7 ms standalone p50, S=1141, n=200). Task comparison, ground-truth-anchored:

| Arm | ADE5 (m) | FDE5 (m) | vs 10-step torch reference | Planner latency |
|---|---|---|---|---|
| 10-step torch (fp semantics reference) | 2.589 | 6.894 | — | 1162.7 ms |
| 1-step torch (same sampler semantics as the engine) | 2.665 | 7.223 | +2.93% / +4.78% | 122.8 ms |
| 1-step engine (production) | 2.653 | 7.164 | **+2.48% / +3.92%** | 35.7 ms |

No measurable engine damage: it is marginally closer to ground truth than its own 1-step torch twin (47/94 scenes), and the +2.48%/+3.92% is the one-step sampler semantics, shown identically by the pure-torch arm. Twin fidelity: 5.1194 cm mean abs plan delta — a change detector, not a quality metric ([docs/results.md](docs/results.md)).

![per-scene ADE5 distributions and the per-scene delta](assets/benchmarks/ade5_distributions.png)
*Per-scene ADE5 against the manifest ground truth (metric numbers only); right panel: paired per-scene delta, engine minus 10-step.*

Production chain gate ([eval_chain_v3](runtime/chain/eval_chain_v3.py), aggregate-blocking):

| Metric | Value | Criterion |
|---|---|---|
| ADE5 / FDE5 vs ground truth, relative to the 10-step torch reference | +2.48% / +3.92% | <= +10% |
| NaN scenes | 0 | 0 allowed |
| Engine-serving contract (planner calls on-engine) | 94/94 | 94/94 required (eager fallback is FATAL) |
| Plan-space twin fidelity (engine vs 1-step torch twin) | 5.1194 cm mean abs | record-only |

93 of the 94 scenes serve in 991–1020 ms; scene 0 is a one-time warm-up (2780.7 ms, first engine call).

![per-scene end-to-end latency timeline and histogram](assets/benchmarks/latency_per_scene.png)
*Left: every scene in run order; scene 0 marked off-scale at its true value. Right: the 93 steady scenes, mean 1007.2 ms.*

Validated on JetPack 6.2.2 / L4T R36.5.0, TensorRT 10.4, TensorRT-Edge-LLM v0.9.1, PyTorch 2.8.0; pinned spec in [docs/environment.md](docs/environment.md).

## Takeaways

- The GDN hybrid layers, not attention, are the text-prefill bottleneck on sm_87; the chunk-parallel prefill kernel moves trt_text −22% and the 10-step chain −144.5 ms.
- Task gates can pass while the numerical-parity lane disagrees — gate on the task metric, read [docs/results.md](docs/results.md) before claiming parity.
- The one-step planner: 1162.7 → 35.7 ms with task metrics in band; INT8 routes are closed with root causes in [docs/negative_results.md](docs/negative_results.md).

## Limitations

- Open-loop evaluation only; no closed-loop eval was executed (environment pointer: [ckpts/README.md](ckpts/README.md)).
- Bound to sm_87 and the pinned stack ([docs/environment.md](docs/environment.md)).
- 94 scenes from one PAI chunk 0 slice — a deployment study, not a driving-benchmark claim.
- One-step sampler: +2.5%/+3.9% ADE5/FDE5 vs the 10-step reference, inside the ±10% band.
- Numerical parity reads low in one lane (kv-cos ≈ 0.88), UNRESOLVED; it does not propagate to task metrics ([docs/results.md](docs/results.md)).

## Get started

1. Clone this repo and the upstream qwen-drive repo side by side per [scripts/env_orin.sh](scripts/env_orin.sh) / [scripts/env_x86.sh](scripts/env_x86.sh).
2. Fill a copy of [scripts/env_x86.sh](scripts/env_x86.sh), then `./scripts/fetch_upstream.sh` — pulls weights, clones upstream, prints the gated [PAI dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) steps.
3. `./run_pipeline.sh <env.sh> x86` — data adapter, vision ONNX export + gate, text gate reference (any CUDA 12.8 GPU with ≥16 GB).
4. On Orin: `./run_pipeline.sh <env.sh> orin` — plugin build, kernel parity gate, planner engine build + gate, planner chain gate.

## What we changed on top of TensorRT-Edge-LLM

- sm_87 GDN chunk-prefill CUDA kernel with dispatch patch and parity harness ([kernels/](kernels/), [kernels/README.md](kernels/README.md)).
- Planner one-step engine path: dynamic-S bf16 export + gate ([export/](export/)), fast-lane chain runner with the engine-serving contract ([runtime/chain/](runtime/chain/)).
- Aggregate-blocking eval gate ([runtime/chain/eval_chain_v3.py](runtime/chain/eval_chain_v3.py)), PAI adapter and 94-scene manifest ([eval/adapter/](eval/adapter/), [configs/](configs/)).
- Edge-LLM 0.9.1 build/compat patch bundle ([export/patches/orin_text_engine/](export/patches/orin_text_engine/)); TRT 10.4 compat delta in [docs/environment.md](docs/environment.md).

## Acknowledgements and license

Apache-2.0. Built on the [Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) model (Apache-2.0) and NVIDIA [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM).
PAI driving data via the gated NVIDIA [PhysicalAI AV dataset](https://huggingface.co/datasets/NVIDIA/PhysicalAI-Autonomous-Vehicles).
Independent engineering effort, not affiliated with Qwen or NVIDIA — full attribution in [NOTICE](NOTICE).
