"""Engine-in-loop: swap the VLM visual tower for a TRT engine at runtime.

The qwen3_5 model calls the tower as
    self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
and consumes only ``.pooler_output`` (deepstack_visual_indexes is empty), which
is exactly the contract the r640 export wrapper traced. The r640 engines bake
grid_thw into the graph as trace-time constants, so every call's grid must
match the first one seen — a deviating scene fails loudly instead of silently
producing garbage embeds.
"""
import numpy as np
import torch
import tensorrt as trt
from transformers.modeling_outputs import BaseModelOutputWithPooling


class _PlanRunner:
    """Static-shape, single-input single-output TRT runner on torch's stream."""

    def __init__(self, plan):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.runtime = trt.Runtime(logger)
        with open(plan, "rb") as fh:
            self.engine = self.runtime.deserialize_cuda_engine(fh.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        modes = [self.engine.get_tensor_mode(n) for n in names]
        assert modes.count(trt.TensorIOMode.INPUT) == 1, (
            f"expected 1 input, got {list(zip(names, modes))}")
        self.in_name = names[modes.index(trt.TensorIOMode.INPUT)]
        self.out_name = names[modes.index(trt.TensorIOMode.OUTPUT)]

        def to_torch(np_dtype):
            return torch.from_numpy(np.zeros(1, np_dtype)).dtype

        self.in_dtype = to_torch(trt.nptype(self.engine.get_tensor_dtype(self.in_name)))
        self.out_dtype = to_torch(trt.nptype(self.engine.get_tensor_dtype(self.out_name)))
        self.out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))

    def run(self, pixel_values):
        x = pixel_values.to(self.in_dtype).contiguous()
        out = torch.empty(self.out_shape, dtype=self.out_dtype, device=x.device)
        self.ctx.set_tensor_address(self.in_name, x.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TRT execute_async_v3 returned False")
        return out


def install_vision_engine(model, plan):
    """Patch model.vlm.model.visual.forward to run the TRT plan instead."""
    runner = _PlanRunner(plan)
    visual = model.vlm.model.visual
    emb_dtype = next(visual.parameters()).dtype
    state = {"grid": None}

    def forward(pixel_values=None, grid_thw=None, *args, **kwargs):
        if pixel_values is None:
            pixel_values = args[0]
        if grid_thw is None:
            grid_thw = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        grid = grid_thw.detach().cpu().tolist()
        if state["grid"] is None:
            state["grid"] = grid
            extra = {k: type(v).__name__ for k, v in kwargs.items()}
            print(f"[engine-in-loop] grid={grid} engine_out={runner.out_shape} "
                  f"extra_kwargs={extra}", flush=True)
        elif grid != state["grid"]:
            raise RuntimeError(f"grid {grid} differs from engine-baked {state['grid']}")
        emb = runner.run(pixel_values).to(emb_dtype)
        return BaseModelOutputWithPooling(pooler_output=emb)

    visual.forward = forward
    return model
