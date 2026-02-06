#!/usr/bin/env python3
"""Debug: test if model loads and generates"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import traceback

MODEL = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading {MODEL}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    print("Model loaded OK")
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=10)
    print("Output:", tokenizer.decode(out[0]))
except Exception as e:
    print("FULL TRACEBACK:")
    traceback.print_exc()
