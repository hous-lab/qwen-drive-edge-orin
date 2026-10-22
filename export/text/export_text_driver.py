#!/usr/bin/env python3
"""Export driver for the Qwen-Drive text engine work.

Why this exists: this Edge-LLM working copy's ``tensorrt_edgellm/__init__.py``
is a 0-byte file (original saved as ``__init__.py.bak_width``, pre-existing
local modification from earlier width-experiments -- see README in patches/).
An empty package __init__ means the model registry never gets populated, and
``AutoModel.from_pretrained`` SILENTLY falls back to the default ``CausalLM``,
which mis-traces qwen3_5_text hybrids (view [B,S,heads,head_dim] on the
q+gate tensor, "shape '[s59, s29, 4, 64]' is invalid for input of size
512*s29*s59").

Fix without touching the repo: re-execute the backup registration inside the
already-imported package namespace, then run the stock export CLI via runpy.

Usage (same CLI as tensorrt_edgellm.scripts.export):
  python export_text_driver.py <model_dir> <out_dir> --dtype float16 \
      --skip-visual --skip-audio
"""
import runpy
import sys
from pathlib import Path

EDGE = Path(os.environ["QD_EDGELLM_SRC"])  # TensorRT-Edge-LLM checkout (v0.9.1)

# Resolve tensorrt_edgellm to THIS tree, not the _stack tree that
# easy-install.pth injects (x86 default python3 has the _stack checkout on
# sys.path via ~/.local/.../easy-install.pth).
sys.path.insert(0, str(EDGE))

import tensorrt_edgellm  # noqa: E402  (empty __init__; namespace exists)
import os

# Minimal registry restore: only what the qwen3_5_text export needs.
# (The full __init__.py.bak_width exec also imports gemma4/qwen3_omni, whose
# model files import transformers classes (Gemma4AudioConfig) that x86
# transformers 4.57.6 does not have -> ImportError. We only export
# qwen3_5_text, so register just that.)
from tensorrt_edgellm.model import (AutoModel as _AutoModel,  # noqa: E402,F401
                                    register_model as _register_model,
                                    standard_attention_scale as _sas)
from tensorrt_edgellm.models.qwen3_5.modeling_qwen3_5_text import (  # noqa: E402
    Qwen3_5CausalLM as _Qwen35)

_register_model("qwen3_5_text", _Qwen35, _sas)
print("[export_text_driver] registered qwen3_5_text ->",
      _Qwen35.__name__, "in", tensorrt_edgellm.__file__)

sys.argv = [sys.argv[0]] + sys.argv[1:]
run_module = runpy.run_module("tensorrt_edgellm.scripts.export",
                              run_name="__main__")
