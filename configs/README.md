# configs

Build/runtime configuration for the certified production configuration
(v2 baseline, frozen 2026-10-12). The shipped scripts read these as
documentation; env vars remain the source of truth at run time.

## File index

| File | What it pins |
|---|---|
| [text_engine_build.env](text_engine_build.env) | cmake flags for the Orin plugin/toolchain build ([build_plugin_orin.sh](../export/patches/orin_text_engine/build_plugin_orin.sh) encodes the same values) |
| [vision_engine_build.env](vision_engine_build.env) | trtexec flags for the fp16 vision engine |
| [manifest_f2_05.env](manifest_f2_05.env) | eval manifest contract used by the 94-scene gate |

## Production configuration (certified)

| Component | Implementation |
|---|---|
| Vision | TRT fp16 plan (Edge-LLM path), 3-view f2_05 input contract |
| Text prefill | TRT fp16 engine on the patched Edge-LLM tree (GDN sm_87 kernel) |
| Planner | torch eager (PlanningExpert, one-step) |
| Manifest | 94 scenes from PhysicalAI AV chunk 0, f2_05 contract (2-frame history + current frame at r640), T=1141 tokens, kvlen capacity 4096 |
| Gate | eval_chain_v3 aggregate-blocking: mean abs delta <= 5cm AND ADE5/FDE5 rel <= +10% AND no NaN vs golden-arm trajectories |
