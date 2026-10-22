#!/usr/bin/env python3
"""Build int8_sq text-calibration jsonl from production scene prompts.

Edge-LLM's builtin text calib datasets read the ``article`` field
(quantization/datasets/builtin.py), so we emit {"article": <prompt text>}.

Source: qwen_scenes_chunk0_r640.jsonl chatml records -- the exact PAI/r640
prompts the production engine sees. Image entries are skipped; all text
content items are concatenated in order (camera markers, frame labels,
task instruction). The assistant message text is included too so the
calibration distribution covers the full sequence the LM processes.
"""
import argparse
import json
from pathlib import Path


def texts_from_record(rec):
    parts = []
    for msg in rec.get("messages", []):
        for c in msg.get("content", []):
            t = c.get("text")
            if t:
                parts.append(t)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.scenes) as fin, open(args.out, "w") as fout:
        for line in fin:
            n_in += 1
            rec = json.loads(line)
            text = texts_from_record(rec).strip()
            if text:
                fout.write(json.dumps({"article": text}, ensure_ascii=False)
                           + "\n")
                n_out += 1
                if args.limit and n_out >= args.limit:
                    break
    print(f"wrote {n_out}/{n_in} calib texts to {args.out}")


if __name__ == "__main__":
    main()
