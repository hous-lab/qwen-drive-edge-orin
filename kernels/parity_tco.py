"""Parity + latency: TC-o (K3b all-bf16 wmma) EXPERIMENT variant vs fallback."""
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
KERNEL_DIR = HERE / "harness"
sys.path.insert(0, str(KERNEL_DIR))

from parity_test import REF_SRC, _ref_chunk, bench, input_stats, load_fixture, \
    make_inputs, max_rel, row_cos_min, row_cos_min_masked, tensor_l2_rel  # noqa: E402


def build_extension():
    from torch.utils.cpp_extension import load

    if "orin" in torch.cuda.get_device_name(0).lower():
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    else:
        if os.environ.get("QD_CUDA_HOME"):
            os.environ["CUDA_HOME"] = os.environ["QD_CUDA_HOME"]
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(HERE / ".ext_cache_tc"))
    return load(
        name="deltanet_chunk_cuda_tco",
        sources=[str(HERE / "chunk_delta_cuda_tco.cu"),
                 str(HERE / "torch_binding_tco.cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O2", "-lineinfo"],
        verbose=False,
    )


if __name__ == "__main__":
    import parity_cuda

    parity_cuda.build_extension = build_extension
    parity_cuda.main()
