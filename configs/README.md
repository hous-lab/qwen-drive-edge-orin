# configs

Build/runtime configuration for the production chain. The shipped scripts read these as documentation; env vars remain the source of truth at run time.

## File index

| File | What it pins |
|---|---|
| [text_engine_build.env](text_engine_build.env) | cmake flags for the Orin plugin/toolchain build ([build_plugin_orin.sh](../export/patches/orin_text_engine/build_plugin_orin.sh) encodes the same values) |
| [vision_engine_build.env](vision_engine_build.env) | trtexec flags for the fp16 vision engine |
| [manifest_f2_05.env](manifest_f2_05.env) | eval manifest contract used by the 94-scene gate |

## Production configuration

| Component | Implementation |
|---|---|
| Vision | TRT fp16 plan (Edge-LLM path), 3-view f2_05 input contract |
| Text prefill | torch bf16 eager (the TRT text engine + sm_87 kernel lane remains available as the `QD_FAST_LANE=0` baseline lane) |
| Planner | TRT bf16 one-step engine, dynamic-S |
| Manifest | 94 scenes from PhysicalAI AV chunk 0, f2_05 contract (2-frame history + current frame at r640), T=1141 tokens, kvlen capacity 4096 |
| Gate | eval_chain_v3 aggregate-blocking: ADE5/FDE5 vs ground truth relative to the 10-step reference <= +10% AND no NaN AND planner 94/94 on-engine; plan-space twin fidelity is recorded, not gated. The `QD_FAST_LANE=0` baseline lane gates mean abs trajectory delta <= 5 cm vs the 10-step torch reference. |
