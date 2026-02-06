#!/usr/bin/env python3
"""
Quick baseline test - verify models load and run
No attention hooks, just measure generation works
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import time
import json
from datetime import datetime
import os

def test_model(model_name, num_samples=20):
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Simple test prompts
    prompts = [
        "The capital of France is",
        "Water freezes at",
        "The largest planet in our solar system is",
        "Einstein developed the theory of",
        "DNA stands for",
        "The speed of light is approximately",
        "The Great Wall of China was built during",
        "Photosynthesis converts sunlight into",
        "The human heart has how many chambers:",
        "Shakespeare wrote the play",
    ] * (num_samples // 10 + 1)
    prompts = prompts[:num_samples]

    print(f"Running {len(prompts)} generations...")
    start = time.time()

    results = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
        if i < 3:
            print(f"  [{i+1}] {response[:100]}...")

    elapsed = time.time() - start
    print(f"Done: {elapsed:.1f}s total, {elapsed/len(prompts):.2f}s per sample")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "num_samples": num_samples,
        "total_time": elapsed,
        "time_per_sample": elapsed / len(prompts),
        "samples": results[:5],  # Save first 5 for inspection
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                       default=["Qwen/Qwen2.5-7B-Instruct", "allenai/OLMo-1.7-7B-hf"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_results = {}
    for model_name in args.models:
        try:
            result = test_model(model_name, args.samples)
            all_results[model_name] = result
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output}/quick_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, result in all_results.items():
        if "error" in result:
            print(f"{model}: ERROR - {result['error']}")
        else:
            print(f"{model}: OK - {result['time_per_sample']:.2f}s per sample")
