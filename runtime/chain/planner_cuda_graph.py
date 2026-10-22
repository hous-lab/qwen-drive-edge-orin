"""Planner one-step acceleration via CUDA graph capture (E2 v1).

Captures `PlanningExpert.predict_endpoint` as a static-shape CUDA graph. Demo
shapes are fully static (B=1, 50 waypoint tokens, scene KV [1, S, 4, 256]),
so the whole 32-layer joint-attention stack replays as a single graph launch
and the per-kernel launch overhead that dominates Orin one-step (~146ms eager)
collapses. The scene KV size varies with prompt length (3385/3383 across the
demo scenes); v1.1 captures one graph per shape on demand (bucketing keyed by
the static signature) instead of falling back to eager on shape change.

Two capture modes:
- use_buffers=True (default): per-kv-group joint buffers [1, S+50, 4, 256];
  the scene prefix is copied in per call, the waypoint tail is rewritten per
  layer inside the graph. Same math as planning_expert_fast.py (G0 verified
  bitwise-equal), minus the per-layer cat materialization.
- use_buffers=False: official math verbatim (per-layer cat); isolates the
  pure launch-overhead win for A/B attribution.

Copy semantics: every input is copied into a static tensor before replay and
the endpoint is cloned out after, so callers never hold graph-pool memory.
Numerics: same kernels on the same pointers, no RNG inside predict_endpoint
-> replay is expected bitwise-equal to eager; verified in-process and by G0.

KNOWN DEFECT (2026-10-06): do NOT combine with TRT engines in one process.
With a TRT engine (vision or text) resident, a process that accumulates >= 2
prior captures produces a later graph whose replays #2..10 diverge from eager
-- deterministic garbage (r640 scene 0391b4a7: 242.67cm vs 0.70cm eager; even
scene 2 diverges mildly with a single graph: 1.38 vs 13.08cm). Capture-time
replay is bitwise-equal to eager even for the corrupted graph; shared capture
pools, resetting old graphs and fresh engine contexts do not help. All-torch
94-graph chains are bitwise-clean (0.00cm), so the trigger is engine
co-residency. Chain stages therefore run the planner EAGER (~+145ms/scene)
and chain_runner.py refuses --text-engine + --graph-planner.
"""
import torch

from qwen_drive.planning_expert import _apply_rotary, _one_hot


def install_cuda_graph_planner(model, num_warmup=3, use_buffers=True, verbose=True):
    """Patch PlanningExpert.predict_endpoint with a CUDA-graph replay; returns model."""
    expert = model.planning_expert
    orig_predict = expert.predict_endpoint
    cfg = expert.config
    num_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    wp_len = expert.num_future_points
    per_group = cfg.layers_per_kv
    num_groups = len(expert.layers) // per_group

    state = {
        "graphs": {},  # sig -> {"graph","st","out","bufs","S","sk","sv"} (v1.1 bucketing)
        "disabled": False,
        # legacy mirrors of the FIRST capture — kept so benchmarks can do
        # state["graph"]/state["disabled"] toggles as before
        "graph": None, "sig": None, "st": None, "out": None,
        "bufs": None, "S": None, "sk": None, "sv": None,
    }

    def log(msg):
        if verbose:
            print(f"[planner-cuda-graph] {msg}", flush=True)

    def signature(waypoints, history_queries, scene_cache, anchor, ego):
        """Static-shape key, or None when this call is not graph-eligible."""
        if not torch.cuda.is_available() or waypoints.device.type != "cuda":
            return None
        if tuple(waypoints.shape) != (1, wp_len, 3) or waypoints.dtype != torch.float32:
            return None
        if len(scene_cache) != num_groups:
            return None
        first = scene_cache[0][0]
        if first.dtype != expert.dtype or first.device != waypoints.device:
            return None
        scene_lens = tuple(k.shape[1] for k, _ in scene_cache)
        if len(set(scene_lens)) != 1 or len(scene_cache[0][0].shape) != 4:
            return None
        if tuple(anchor.shape) != (3, 1) or tuple(ego.shape) != (1, cfg.ego_status_dim):
            return None
        if len(history_queries) != 3 or any(
            tuple(q.shape) != (1, cfg.hidden_size) for q in history_queries
        ):
            return None
        return ("v1", 1, wp_len, scene_lens[0], first.dtype, num_groups)

    def capture(waypoints, flow_time, history_queries, scene_cache, anchor,
                nav_command, ego_status, sig):
        device = waypoints.device
        dtype = expert.dtype
        S = scene_cache[0][0].shape[1]
        st = {
            "wp": waypoints.clone(),
            "ft": flow_time.clone(),
            "hq": [q.clone() for q in history_queries],
            "anchor": anchor.clone(),
            "nav": nav_command.clone(),
            "ego": ego_status.clone(),
        }
        bufs = sk = sv = None
        if use_buffers:
            bufs = []
            for g, (k, v) in enumerate(scene_cache):
                kb = torch.empty(1, S + wp_len, num_kv, head_dim, device=device, dtype=dtype)
                vb = torch.empty_like(kb)
                kb[:, :S].copy_(k.expand(1, -1, -1, -1))
                vb[:, :S].copy_(v.expand(1, -1, -1, -1))
                bufs.append((kb, vb))
        else:
            sk = [k.clone() for k, _ in scene_cache]
            sv = [v.clone() for _, v in scene_cache]

        def run():
            return _predict_static(expert, per_group, st["wp"], st["ft"], st["hq"],
                                   bufs, sk, sv, st["anchor"], st["nav"], st["ego"])

        # Warm up on a side stream so cuBLAS workspaces and the SDPA backend
        # choice happen outside capture (standard torch.cuda.graph recipe).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(num_warmup):
                run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        graph.replay()  # populate outputs with the real values for this call
        entry = {"graph": graph, "st": st, "out": out, "bufs": bufs,
                 "S": S, "sk": sk, "sv": sv}
        state["graphs"][sig] = entry
        if state["graph"] is None:  # legacy mirror of the first capture
            for key in ("graph", "st", "out", "bufs", "S", "sk", "sv"):
                state[key] = entry[key]
        state["sig"] = sig
        log(f"captured (buffers={use_buffers}, S={S}, wp={wp_len}, "
            f"{len(state['graphs'])} graph(s) in pool)")
        return entry

    def predict_endpoint(waypoints, flow_time, history_queries, scene_cache,
                         position_anchor, nav_command, ego_status):
        if state["disabled"]:
            return orig_predict(waypoints, flow_time, history_queries, scene_cache,
                                position_anchor, nav_command, ego_status)
        sig = signature(waypoints, history_queries, scene_cache, position_anchor, ego_status)
        if sig is not None and flow_time.shape != (1,):
            sig = None
        if sig is None:
            return orig_predict(waypoints, flow_time, history_queries, scene_cache,
                                position_anchor, nav_command, ego_status)
        try:
            entry = state["graphs"].get(sig)
            if entry is None:
                # v1.1: capture one graph per static shape (on-demand bucketing);
                # each graph has its own pool, static inputs and joint buffers.
                entry = capture(waypoints, flow_time, history_queries, scene_cache,
                                position_anchor, nav_command, ego_status, sig)
                return entry["out"].clone()

            st = entry["st"]
            st["wp"].copy_(waypoints)
            st["ft"].copy_(flow_time)
            for dst, src in zip(st["hq"], history_queries):
                dst.copy_(src)
            st["anchor"].copy_(position_anchor)
            st["nav"].copy_(nav_command)
            st["ego"].copy_(ego_status)
            if entry["bufs"] is not None:
                S = entry["S"]
                for g, (k, v) in enumerate(scene_cache):
                    kb, vb = entry["bufs"][g]
                    kb[:, :S].copy_(k)
                    vb[:, :S].copy_(v)
            else:
                for g, (k, v) in enumerate(scene_cache):
                    entry["sk"][g].copy_(k)
                    entry["sv"][g].copy_(v)
            entry["graph"].replay()
            return entry["out"].clone()
        except Exception as exc:  # never break the caller on graph trouble
            state["disabled"] = True
            log(f"graph path failed ({exc!r}); disabled, eager from now on")
            return orig_predict(waypoints, flow_time, history_queries, scene_cache,
                                position_anchor, nav_command, ego_status)

    expert.predict_endpoint = predict_endpoint
    # Exposed for benchmarks: toggle state["disabled"] to A/B graph vs eager
    # in-process, and inspect state["sig"]/state["graph"] after capture.
    expert._cuda_graph_state = state
    return model


def _predict_static(expert, per_group, waypoints, flow_time, history_queries,
                    bufs, sk, sv, anchor, nav_command, ego_status):
    """Official predict_endpoint math over static tensors.

    KV supply is either joint buffers (scene prefix pre-written, waypoint tail
    rewritten per layer) or the official per-layer cat — both bitwise-equal to
    the official path (buffer variant already G0-verified in
    planning_expert_fast.py).
    """
    dtype = expert.dtype
    waypoints = waypoints.to(dtype)  # official predict_endpoint casts before GEMMs
    batch, length, _ = waypoints.shape
    pose_query, velocity_query, acceleration_query = history_queries

    time_condition = expert.time_mlp(expert.time_embed(flow_time).to(dtype))
    waypoint_index = torch.arange(length, device=waypoints.device)
    broadcast = [
        expert.trajectory_proj(waypoints),
        expert.fourier_encoder(waypoints),
        time_condition.unsqueeze(1).expand(-1, length, -1),
        pose_query.unsqueeze(1).expand(-1, length, -1),
        expert.waypoint_embed(waypoint_index).unsqueeze(0).expand(batch, -1, -1).to(dtype),
        velocity_query.unsqueeze(1).expand(-1, length, -1),
        acceleration_query.unsqueeze(1).expand(-1, length, -1),
    ]
    hidden_states = expert.query_fusion(torch.cat(broadcast, dim=-1))

    nav_onehot = _one_hot(nav_command, expert.config.nav_command_classes, dtype)
    condition = time_condition + expert.nav_mlp(nav_onehot) + expert.ego_mlp(ego_status.to(dtype))

    steps = torch.arange(1, length + 1, device=anchor.device, dtype=anchor.dtype)
    positions = anchor.unsqueeze(-1) + steps
    cos, sin = expert.rotary_emb(positions, dtype)

    buffer_mode = bufs is not None
    S = bufs[0][0].shape[1] - length if buffer_mode else None
    for index, layer in enumerate(expert.layers):
        if buffer_mode:
            kb, vb = bufs[index // per_group]
        else:
            kb, vb = sk[index // per_group], sv[index // per_group]
        hidden_states = _layer_static(layer, hidden_states, kb, vb, S, cos, sin, condition)
    return expert.out_proj(expert.final_layernorm(hidden_states)).float()


def _layer_static(layer, hidden_states, kfull, vfull, S, cos, sin, condition):
    """PlanningExpertLayer.forward with a pluggable KV source.

    S is None -> official cat of scene|waypoint; else kfull/vfull are joint
    buffers whose [:, S:] tail is rewritten with this layer's K/V (fast-planner
    math verbatim)."""
    import torch.nn.functional as F

    batch, length, _ = hidden_states.shape
    modulation = layer.adaln_modulation(condition).chunk(6, dim=-1)
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
        m.unsqueeze(1) for m in modulation)

    residual = hidden_states
    x = layer.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
    query, gate, key, value = layer._split_qkv(layer.qkv_proj(x))
    query = _apply_rotary(layer.q_norm(query), cos, sin)
    key = _apply_rotary(layer.k_norm(key), cos, sin)

    if S is None:
        key = torch.cat([kfull, key], dim=1)
        value = torch.cat([vfull, value], dim=1)
        attn = layer._attend(query, key, value)
    else:
        # joint buffers: scene prefix copied in by the caller, tail rewritten here
        kfull[:, S:] = key
        vfull[:, S:] = value
        attn = layer._attend(query, kfull, vfull)
    attn = attn.reshape(batch, length, -1)
    attn = attn * torch.sigmoid(gate.reshape(batch, length, -1))
    hidden_states = residual + (1 + gate_attn) * layer.o_proj(attn)

    residual = hidden_states
    x = layer.post_attention_layernorm(hidden_states) * (1 + scale_ffn) + shift_ffn
    swiglu_gate, swiglu_up = layer.gate_up_proj(x).chunk(2, dim=-1)
    return residual + (1 + gate_ffn) * layer.down_proj(F.silu(swiglu_gate) * swiglu_up)
