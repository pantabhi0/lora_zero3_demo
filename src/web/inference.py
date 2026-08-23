"""Memory-conscious inference service used by the A-side WebUI."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class InferenceService:
    def __init__(self, output_dir: str = "logs", model_name: str = "Qwen/Qwen2.5-0.5B"):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.device = torch.device("cuda", 0)
        self._lock = threading.RLock()
        self._base = None
        self._model = None
        self._adapter_tag = None
        self._tokenizer = None

    def tags(self) -> list[str]:
        allowed = ["baseline_1k", "2node_1k", "baseline_long", "2node_long"]
        return [tag for tag in allowed if (self.output_dir / tag / "adapter").exists()]

    def training_active(self) -> bool:
        return (self.output_dir / ".training_active").exists()

    def _ensure_base(self):
        if self._model is not None:
            return
        self._base = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to(self.device)
        first = self.tags()[0]
        self._model = PeftModel.from_pretrained(
            self._base, self.output_dir / first / "adapter", adapter_name=first
        )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.disable_adapter_layers()
        self._adapter_tag = None

    def select(self, tag: str) -> None:
        if tag not in self.tags():
            raise ValueError(f"unknown or incomplete adapter: {tag}")
        with self._lock:
            self._ensure_base()
            if self._adapter_tag == tag:
                return
            if self._adapter_tag is not None:
                self._model.delete_adapter(self._adapter_tag)
            if tag not in self._model.peft_config:
                self._model.load_adapter(str(self.output_dir / tag / "adapter"), adapter_name=tag)
            self._model.set_adapter(tag)
            self._model.enable_adapter_layers()
            self._adapter_tag = tag

    def generate_pair(self, tag: str, instruction: str, max_new_tokens=512,
                      temperature=0.7, top_p=0.9, repetition_penalty=1.2, seed=42):
        if self.training_active():
            raise RuntimeError("inference disabled while training is active")
        with self._lock:
            self.select(tag)
            prompt = f"### Instruction:\n{instruction}\n### Response:\n"
            ids = self._tokenizer(prompt, return_tensors="pt").to(self.device)

            def run(adapter_enabled: bool) -> str:
                if adapter_enabled:
                    self._model.enable_adapter_layers()
                else:
                    self._model.disable_adapter_layers()
                torch.manual_seed(seed)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self._model.generate(
                        **ids, max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=temperature, top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                text = self._tokenizer.decode(
                    out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
                )
                cut = text.find("\n### ")
                return text[:cut].strip() if cut >= 0 else text.strip()

            base = run(False)
            finetuned = run(True)
            return {"tag": tag, "base": base, "finetuned": finetuned}
