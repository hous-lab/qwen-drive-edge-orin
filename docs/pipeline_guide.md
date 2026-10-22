# Pipeline guide

Numbered steps follow `run_pipeline.sh`; this page explains what each stage
does and what good output looks like.

## Stage map

| Step | Host | What it runs | Good output |
|---|---|---|---|
| 1-3 | both | workspace env + upstream artifact guards + python stack | FAIL messages carry the download pointer |
| 4 | x86 | PAI chunk 0 adapter -> scenes jsonl + frames + archives | `qwen_scenes_chunk0.jsonl` (94 scenes) |
| 5 | x86 | vision ONNX export + ORT parity gate | `gate_vision_onnx_x86.json`: cos vs fp32 ref >= 0.999 |
| 6 | x86 | text gate reference: validate golden npz, derive gate ref | v-vs-golden min cos > 0.999 in the report json |
| 7 | orin | Edge-LLM plugin build (`build_plugin_orin.sh`) | `libNvInfer_edgellm_plugin.so` at `$QD_PLUGIN_SO` |
| 8 | orin | GDN kernel parity gate (`kernels/parity_tco.py`) | cos >= 0.999 / max rel < 1e-2 vs Triton reference |
| 9 | orin | engine presence guards | vision `.plan` + text `.plan` exist |
| 10 | orin | chain gate: torch reference arm vs engine arm -> eval_chain_v3 | `eval_v3_summary.json` verdict PASS |

## Stage notes

### Adapter (step 4)
Encodes PAI chunk 0 clips into the upstream scene-record schema (camera
frames, 2-frame history + current frame, 10 Hz future trajectory, ego status,
driving/nav commands) and flags degenerate clips (min per-pixel std < 15).
`--pack` builds the packed parquet archives chain_runner consumes via
`--image-archive`.

### Vision ONNX export + gate (step 5)
`export_vision_onnx.py` traces the vision tower (pooler output) with the
scene's grid baked in — static shapes, 3 views x 4 frames — and dumps the
test-tensor bundle. The driver gate (`gate_vision_onnx_x86.py`) runs the
ONNX under onnxruntime CPU and compares against the fp32/bf16/fp16 torch
references in the bundle (gate: cos vs fp32 >= 0.999).

The repo also ships `gate_edgellm_onnx_x86.py` for the second, separate
vision artifact: the Edge-LLM visual graph ONNX (5 dynamic inputs), produced
by Edge-LLM's own exporter out of `$QD_EDGELLM_SRC` — outside this repo's
driver. `dump_edgellm_gate_inputs.py` builds its 5-input feed (rotary /
cu_seqlens / pos-embed idx+weights) from the same tensor bundle; the gate
compares row-cosine against references and is what production used to gate
the fp16 vision engine's source graph. On Orin, `gate_vision_edgellm.py`
gates the built vision engine itself (same 5-input contract, plugin .so
required) against the same bundle refs.

### Text gate reference (step 6)
`gen_mrope_ref_f2.py` is a validator + deriver, not a producer: it re-runs
the bf16 HF text model on the f2 inputs_embeds with the golden pos3d, proves
input/position equivalence via the never-roped v projection (cos > 0.999 per
layer), and writes the gate reference (our logits/hidden + golden post-rope
KV) that `export/text/gate_text_engine_orin.py --rope-npz mrope_golden_f2.npz
--ref-npz mrope_gate_ref_f2.npz` compares the TRT engine against. The
golden/bundle npz come from the hous-lab HF repo (ckpts/README.md).

### Plugin build (step 7)
`build_plugin_orin.sh` clones/stages a fresh Edge-LLM v0.9.1 tree, applies
the patch bundle (TRT 10.4 compat + GDN sm_87 kernel dispatch), stages the
shipped kernel sources from `kernels/`, and builds `NvInfer_edgellm_plugin`.
The certified production .so was produced by the objcopy-swap vehicle — see
kernels/README.md for the distinction and why the clean rebuild is the
reproducible path.

### Chain gate (step 10)
Two arms over the same scene set: the torch reference arm (pure bf16 eager,
no engines) and the engine arm (vision + text engines + torch planner).
`eval_chain_v3.py` compares per-scene trajectories and archives a summary;
aggregate-blocking verdict (any criterion tripped = FAIL) with the criteria
in `configs/manifest_f2_05.env`.