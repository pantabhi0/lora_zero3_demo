"""Phase 6 add-on — prompt UI for laptop A: base vs LoRA-finetuned.

Runs on machine A only. Loads the base model AND the trained adapter,
lets the user type a prompt, and prints both answers side by side so the
fine-tuning effect is visible. The comparison itself is done by the user
manually; this is just the interface.

    uv run python -m src.infer
    uv run python -m src.infer --adapter logs/baseline/adapter --max-new-tokens 128

Prompts are single-line. Type 'quit' / 'exit' or Ctrl-D to leave.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default=None,
                   help="path to trained LoRA adapter (default: from config, logs/<run_tag>/adapter)")
    p.add_argument("--model", default=None, help="base model id (default: from config)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--config", default="configs/lora_config.yaml")
    return p.parse_args()


def build_prompt(instruction: str, input_text: str) -> str:
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    return prompt + "### Response:\n"


def generate(model, tokenizer, prompt: str, max_new: int, device) -> str:
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    # stop at CodeAlpaca-format drift (next "### " section) instead of looping
    cut = text.find("\n### ")
    if cut != -1:
        text = text[:cut]
    return text.strip()


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    model_id = args.model or cfg["model"]["name"]
    adapter = args.adapter or str(
        Path(cfg["logging"].get("output_dir", "logs")) / cfg["logging"].get("run_tag", "baseline") / "adapter"
    )
    if not Path(adapter).exists():
        sys.exit(f"Adapter not found at {adapter}. Train first, or pass --adapter.")

    device = torch.device("cuda", 0)
    dtype = torch.float16
    print(f"loading base {model_id} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    print(f"loading adapter {adapter} ...", flush=True)
    finetuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device),
        adapter,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nbase + finetuned ready. Type a prompt ('quit' to exit).\n")
    while True:
        try:
            line = input("You > ").strip()
        except EOFError:
            print()
            break
        if not line or line.lower() in ("quit", "exit"):
            break
        prompt = build_prompt(line, "")
        print(f"\n--- BASE ---\n{generate(base, tokenizer, prompt, args.max_new_tokens, device)}")
        print(f"\n--- FINETUNED ---\n{generate(finetuned, tokenizer, prompt, args.max_new_tokens, device)}\n")


if __name__ == "__main__":
    main()