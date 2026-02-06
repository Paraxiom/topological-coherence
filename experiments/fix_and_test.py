#!/usr/bin/env python3
"""Fix: upgrade packages and test with Mistral (no trust_remote_code needed)"""
import subprocess
import sys

# Step 1: Upgrade packages
print("Upgrading transformers and bitsandbytes...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "transformers", "bitsandbytes", "-q"])

# Step 2: Test with Mistral (more compatible)
print("\nTesting with Mistral-7B...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Model loaded OK!")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n✓ Setup working! Now try Qwen...")

# Step 3: Try Qwen again after upgrade
print("\nTesting Qwen after upgrade...")
try:
    MODEL2 = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL2, trust_remote_code=True)
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    inputs2 = tokenizer2("Hello", return_tensors="pt").to(model2.device)
    outputs2 = model2.generate(**inputs2, max_new_tokens=10)
    print("Qwen Output:", tokenizer2.decode(outputs2[0], skip_special_tokens=True))
    print("\n✓ Qwen working too!")
except Exception as e:
    print(f"Qwen still failing: {e}")
    print("Use Mistral for tests instead.")
