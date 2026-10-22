#!/usr/bin/env python3
"""In-loop TRT planner engine for chain_runner (--planner-engine).

Patches ``PlanningExpert.predict_endpoint`` to run the one-step engine
(export: 03_export/planner/export_planner_onnx.py, bf16 plan, gate:
gate_planner_engine.py cos>=0.999 vs ref_prod). The engine replaces ONE flow
step; the official flow-matching loop calls it num_steps times per scene.

Contract (engine IO, from the export/gate scripts):
- inputs: waypoints (1,50,3) f32, flow_time (1,) f32, hq_pose/velocity/accel
  (1,1024) f32, scene_k/scene_v (8,S,4,256) f32, nav_command (1,) i64,
  ego_status (1,8) f32, cos/sin (1,50,1,64) f32; output endpoint (1,50,3) f32.
- cos/sin are engine INPUTS by design: produced per call by the official
  WaypointRotaryEmbedding at production bf16 rounding (export R4/R6: fp32
  tables break parity, cos 0.9897), upcast to the engine's fp32 IO — the
  exact gate-bundle semantics.
- scene_k/v: torch scene_cache is 8 groups of (1,S,4,256) in expert dtype;
  stacked and exact-upcast to fp32 per call.

Shapes: engines may be static (scene_k dim1 fixed) or dynamic (dim1 -1, one
profile spanning both chain shapes). Static engines match one exact S;
dynamic engines match S in [min,max] from their profile. Calls with no
matching engine fall back to the eager expert (counted, loud) — the same
fallback semantics as the graph installer's bucketing miss.

Stream discipline: enqueue on a dedicated non-default stream (the gate found
legacy-default-stream enqueueV3 inserts a per-call cudaStreamSynchronize,
~2x latency noise); the current stream waits the engine stream before the
output clone so downstream torch ops are ordered.
"""
import time

import torch


def _torch_dtype(trt_dtype):
    import tensorrt as trt
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.INT8: torch.int8,
    }
    return mapping[trt_dtype]


class PlannerEngineRunner:
    """One plan bound to torch buffers on cuda:0 (static or dynamic-S)."""

    def __init__(self, engine_path):
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"deserialize failed: {engine_path}"
        self.ctx = self.engine.create_execution_context()
        self.device = torch.device("cuda:0")
        self.path = engine_path
        self.timings = {}

        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dt = _torch_dtype(self.engine.get_tensor_dtype(name))
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs)[name] = \
                (shape, dt)

        assert set(self.inputs) == {
            "waypoints", "flow_time", "hq_pose", "hq_velocity", "hq_accel",
            "scene_k", "scene_v", "nav_command", "ego_status", "cos", "sin"}, \
            f"unexpected engine IO: {sorted(self.inputs)}"
        assert list(self.outputs) == ["endpoint"], sorted(self.outputs)
        self.dynamic = self.inputs["scene_k"][0][1] < 0
        if self.dynamic:
            lo, opt, hi = self.engine.get_tensor_profile_shape(
                "scene_k", 0)  # (8,S,4,256) at profile 0
            self.s_min, self.s_opt, self.s_max = int(lo[1]), int(opt[1]), int(hi[1])
            self.S = (self.s_min, self.s_max)
        else:
            self.S = self.inputs["scene_k"][0][1]

        # static small-input/output buffers, addresses bound once
        self.bufs = {}
        for name, (shape, dt) in list(self.inputs.items()) + list(self.outputs.items()):
            if self.dynamic and name in ("scene_k", "scene_v"):
                continue                     # fresh per-call tensors for dynamic S
            assert all(s >= 0 for s in shape), f"unexpected dynamic dim: {name}"
            buf = torch.zeros(shape, dtype=dt, device=self.device)
            self.bufs[name] = buf
            self.ctx.set_tensor_address(name, buf.data_ptr())
        self.stream = torch.cuda.Stream()

    def _match(self, s):
        return (self.s_min <= s <= self.s_max) if self.dynamic else (s == self.S)

    def run(self, waypoints, flow_time, history_queries, scene_cache,
            position_anchor, nav_command, ego_status, expert):
        """Engine-executed predict_endpoint; returns (1,50,3) float32."""
        t0 = time.perf_counter()
        wp_len = self.bufs["waypoints"].shape[1]
        assert tuple(waypoints.shape) == (1, wp_len, 3) and waypoints.dtype == torch.float32
        # exact upcast of production values to the engine's fp32 IO
        self.bufs["waypoints"].copy_(waypoints)
        self.bufs["flow_time"].copy_(flow_time.float())
        for name, q in zip(("hq_pose", "hq_velocity", "hq_accel"), history_queries):
            self.bufs[name].copy_(q.float())
        k = torch.stack([kk[0] for kk, _ in scene_cache]).float()   # (8,S,4,256)
        v = torch.stack([vv[0] for _, vv in scene_cache]).float()
        self.bufs["nav_command"].copy_(nav_command.reshape(1).long())
        self.bufs["ego_status"].copy_(ego_status.reshape(1, -1).float())
        # official positions/rotary math (production bf16 tables, R4/R6)
        steps = torch.arange(1, wp_len + 1, device=position_anchor.device,
                             dtype=position_anchor.dtype)
        positions = position_anchor.unsqueeze(-1) + steps     # [3,1,L]
        cos, sin = expert.rotary_emb(positions, expert.dtype)
        self.bufs["cos"].copy_(cos.reshape(1, wp_len, 1, -1).float())
        self.bufs["sin"].copy_(sin.reshape(1, wp_len, 1, -1).float())
        if self.dynamic:
            for name, t in (("scene_k", k), ("scene_v", v)):
                assert self.ctx.set_input_shape(name, tuple(t.shape)), name
                self.ctx.set_tensor_address(name, t.data_ptr())
        else:
            self.bufs["scene_k"].copy_(k)
            self.bufs["scene_v"].copy_(v)
        torch.cuda.current_stream().synchronize()
        self.timings["copies_ms"] = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        assert ok, "execute_async_v3 returned False"
        torch.cuda.current_stream().wait_stream(self.stream)
        self.timings["engine_ms"] = (time.perf_counter() - t1) * 1e3
        return self.bufs["endpoint"].clone()


def install_planner_engine(model, engine_paths, verbose=True):
    """Patch predict_endpoint with TRT engine execution; returns the state dict.

    engine_paths: list of .plan files; matched per call by scene-KV length
    (static engines: exact S; dynamic engines: profile [min,max]).
    """
    expert = model.planning_expert
    orig_predict = expert.predict_endpoint
    runners = []
    for p in engine_paths:
        runner = PlannerEngineRunner(p)
        runners.append(runner)
    state = {
        "runners": runners, "orig": orig_predict, "disabled": False,
        "engine_calls": 0, "eager_calls": 0, "eager_shapes": {},
    }

    def log(msg):
        if verbose:
            print(f"[planner-engine] {msg}", flush=True)

    for r in runners:
        kind = f"dynamic S[{r.s_min}..{r.s_max}] opt={r.s_opt}" if r.dynamic \
            else f"static S={r.S}"
        log(f"engine {kind}: {r.path.split('/')[-1]}")

    def predict_endpoint(waypoints, flow_time, history_queries, scene_cache,
                         position_anchor, nav_command, ego_status):
        if state["disabled"]:
            return orig_predict(waypoints, flow_time, history_queries, scene_cache,
                                position_anchor, nav_command, ego_status)
        try:
            if (len(scene_cache) != 8 or scene_cache[0][0].shape[0] != 1
                    or scene_cache[0][0].dim() != 4):
                raise ValueError("scene_cache layout mismatch")
            s_key = scene_cache[0][0].shape[1]
            runner = next((r for r in runners if r._match(s_key)), None)
            if runner is None:
                state["eager_calls"] += 1
                state["eager_shapes"][s_key] = state["eager_shapes"].get(s_key, 0) + 1
                if state["eager_shapes"][s_key] == 1:
                    log(f"no engine for S={s_key}; eager fallback")
                return orig_predict(waypoints, flow_time, history_queries,
                                    scene_cache, position_anchor, nav_command,
                                    ego_status)
            out = runner.run(waypoints, flow_time, history_queries, scene_cache,
                             position_anchor, nav_command, ego_status, expert)
            state["engine_calls"] += 1
            return out
        except Exception as exc:  # never break the caller on engine trouble
            state["disabled"] = True
            log(f"engine path failed ({exc!r}); disabled, eager from now on")
            return orig_predict(waypoints, flow_time, history_queries, scene_cache,
                                position_anchor, nav_command, ego_status)

    expert.predict_endpoint = predict_endpoint
    expert._planner_engine_state = state
    return state
