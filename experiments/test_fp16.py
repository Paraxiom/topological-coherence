#!/usr/bin/env python3
"""Test without quantization - just float16. A40 has enough VRAM."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-v0.1"

print(f"Loading {MODEL} in float16 (no quantization)...")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model loaded OK!")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n✓ Working!")
