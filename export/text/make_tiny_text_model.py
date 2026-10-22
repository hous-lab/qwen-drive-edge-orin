#!/usr/bin/env python3
"""Create a tiny random-weight qwen3_5_text checkpoint for the E3 baseline loop.

Purpose: exercise the FULL official export -> Orin build -> engine-execution
chain on a minutes-scale model before touching the 4B real weights, and
reproduce the sm_87 GDN-prefill-zeros defect with the official plugin.

Shape keeps every constraint the real model hits:
  - layer_types = [gdn, gdn, gdn, full_attention]  (3:1 hybrid, same pattern)
  - linear k/v head dims = 128 (hard plugin requirement), 2 k-heads / 4 v-heads
    (grouped repeat_interleave semantics, same 1:2 ratio class as real 16:32)
  - full attention: 4 q heads / 1 kv head, head_dim 64, partial rotary 0.25,
    attn_output_gate = true (real model has it on)
  - tie_word_embeddings = true

Weights are the HF model's default init.  Saved with save_pretrained so the
output dir is directly usable as the export workdir (config.json included).

Run on x86 (env per scripts/env_x86.sh):

  QD_WORK_DIR=<work dir> python export/text/make_tiny_text_model.py
"""
import sys
import os
from pathlib import Path

OUT = Path(os.environ.get("QD_WORK_DIR", ".")) / "edgellm_work_tiny"

CONFIG = dict(
    model_type="qwen3_5_text",
    num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention",
                 "full_attention"],
    hidden_size=512,
    intermediate_size=1024,
    # GDN (plugin hard-requires k_dim = v_dim = 128)
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    # full attention
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=64,
    partial_rotary_factor=0.25,
    attn_output_gate=True,
    rms_norm_eps=1e-6,
    rope_parameters=dict(mrope_interleaved=True, mrope_section=[3, 3, 2],
                         partial_rotary_factor=0.25, rope_theta=1e6,
                         rope_type="default"),
    vocab_size=1024,
    max_position_embeddings=2048,
    tie_word_embeddings=True,
    use_cache=True,
    hidden_act="silu",
)


def main():
    import torch
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    cfg = AutoConfig.for_model(**CONFIG)
    assert cfg.model_type == "qwen3_5_text"
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg)
    model = model.to(torch.float32).eval()

    # keep activations small so the fp16 engine and the fp32 reference both
    # stay well-conditioned
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.3)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    # Borrow the real checkpoint's tokenizer artifacts: export's chat-template
    # extraction searches model_dir for a tokenizer first (see
    # prepare_edgellm_text_workdir.py for why the fallback path must not run).
    import os
    REAL = Path(os.environ.get("QD_MODEL_DIR", "Qwen-Drive-1.0-4B"))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "chat_template.jinja", "chat_template.json"):
        src_t = REAL / name
        if src_t.exists():
            link_t = OUT / name
            if link_t.exists() or link_t.is_symlink():
                link_t.unlink()
            os.symlink(src_t.resolve(), link_t)
    sd_keys = list(model.state_dict().keys())
    gdn = [k for k in sd_keys if "linear_attn" in k]
    print(f"wrote {OUT}")
    print(f"  tensors: {len(sd_keys)}, linear_attn: {len(gdn)}")
    print("  sample:", [k for k in sd_keys if "linear_attn" in k][:6])
    # quick forward as smoke test
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids)
    print(f"  forward OK: logits {tuple(out.logits.shape)}")


if __name__ == "__main__":
    sys.exit(main())
