#!/usr/bin/env python3
"""
OPTIMAL OLMo TEST
=================
Uses best parameters from sweep: α=0.2, r=3.0, n=3000
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

def toroidal_distance(i, j, grid_size=12):
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

def get_optimal_bias(vocab_size, recent_tokens, device='cuda'):
    """Optimal parameters: α=0.2, r=3.0, n=3000"""
    alpha = 0.2
    radius = 3.0
    max_tokens = 3000
    grid_size = 12

    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)
    if len(recent_tokens) == 0:
        return bias

    for offset, token_id in enumerate(recent_tokens[-5:]):
        token_pos = token_id % (grid_size * grid_size)
        for vocab_id in range(min(vocab_size, max_tokens)):
            target_pos = vocab_id % (grid_size * grid_size)
            dist = toroidal_distance(token_pos, target_pos, grid_size)
            if dist <= radius:
                bias[vocab_id] += alpha * (radius - dist + 1) / (offset + 1)
            elif dist <= radius * 2:
                bias[vocab_id] += alpha * 0.5 / (offset + 1)
    return bias

def generate_baseline(model, tokenizer, prompt, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_optimal(model, tokenizer, prompt, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]
            bias = get_optimal_bias(vocab_size, generated, model.device)
            logits = logits + bias
            next_token = logits.argmax().item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

TEST_PROMPTS = [
    ("The capital of France is", ["Paris"]),
    ("The capital of Japan is", ["Tokyo"]),
    ("The capital of Australia is", ["Canberra"]),
    ("The capital of Brazil is", ["Brasilia"]),
    ("The capital of Canada is", ["Ottawa"]),
    ("Mount Everest is in", ["Nepal", "Himalaya"]),
    ("The Amazon River is in", ["South America", "Brazil"]),
    ("The Great Wall is in", ["China"]),
    ("The Nile River flows through", ["Egypt", "Africa"]),
    ("The Sahara Desert is in", ["Africa"]),
    ("The largest ocean is the", ["Pacific"]),
    ("The longest river in the world is the", ["Nile", "Amazon"]),
    ("Australia is a", ["continent", "country"]),
    ("The Alps are in", ["Europe"]),
    ("The Dead Sea is between", ["Israel", "Jordan"]),
    ("Venice is famous for its", ["canals", "water"]),
    ("The Eiffel Tower is in", ["Paris", "France"]),
    ("The Great Barrier Reef is near", ["Australia"]),
    ("Tokyo is the capital of", ["Japan"]),
    ("The Statue of Liberty is in", ["New York", "USA"]),
    ("Water freezes at", ["0", "32", "zero"]),
    ("The largest planet is", ["Jupiter"]),
    ("Einstein developed the theory of", ["relativity"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The atomic number of hydrogen is", ["1", "one"]),
    ("Photosynthesis converts", ["energy", "glucose", "sugar", "sunlight"]),
    ("Oxygen is about what percent of air", ["21", "20"]),
    ("Pi equals approximately", ["3.14"]),
    ("The speed of light is", ["300", "299", "186"]),
    ("The human heart has", ["four", "4"]),
    ("Newton discovered", ["gravity", "motion"]),
    ("The chemical symbol for water is", ["H2O"]),
    ("The boiling point of water is", ["100", "212"]),
    ("Electrons have a", ["negative"]),
    ("The sun is a", ["star"]),
    ("Diamonds are made of", ["carbon"]),
    ("The human body has how many bones", ["206"]),
    ("Sound travels faster in", ["water", "solid"]),
    ("The earth revolves around the", ["sun"]),
    ("Gravity was discovered by", ["Newton"]),
    ("The smallest unit of life is a", ["cell"]),
    ("Mitochondria are the", ["powerhouse"]),
    ("The pH of pure water is", ["7", "seven", "neutral"]),
    ("The chemical symbol for iron is", ["Fe"]),
    ("World War II ended in", ["1945"]),
    ("The Declaration of Independence was signed in", ["1776"]),
    ("The first human on the moon was", ["Armstrong", "Neil"]),
    ("The Berlin Wall fell in", ["1989"]),
    ("World War I started in", ["1914"]),
    ("The French Revolution began in", ["1789"]),
    ("Columbus sailed to America in", ["1492"]),
    ("The Roman Empire fell in", ["476", "5th"]),
    ("The Renaissance began in", ["Italy", "14th", "15th"]),
    ("The printing press was invented by", ["Gutenberg"]),
    ("The American Civil War ended in", ["1865"]),
    ("The Soviet Union collapsed in", ["1991"]),
    ("The Titanic sank in", ["1912"]),
    ("Martin Luther King Jr gave his famous speech in", ["1963"]),
    ("The Great Depression started in", ["1929"]),
    ("The first airplane flight was by the", ["Wright"]),
    ("Mahatma Gandhi led", ["India", "independence"]),
    ("The Cold War was between", ["USA", "Soviet", "America", "Russia"]),
    ("Ancient Egypt was known for", ["pyramids", "pharaohs"]),
    ("The Industrial Revolution began in", ["Britain", "England", "18th"]),
    ("The Mona Lisa was painted by", ["Leonardo", "Vinci"]),
    ("Shakespeare wrote", ["Hamlet", "Romeo", "Macbeth"]),
    ("Beethoven was a famous", ["composer", "musician"]),
    ("The currency of Japan is", ["yen"]),
    ("Vincent van Gogh painted", ["Starry Night", "sunflowers"]),
    ("Romeo and Juliet was written by", ["Shakespeare"]),
    ("The Sistine Chapel ceiling was painted by", ["Michelangelo"]),
    ("Mozart was born in", ["Austria", "Salzburg"]),
    ("Harry Potter was written by", ["Rowling"]),
    ("The Odyssey was written by", ["Homer"]),
    ("Don Quixote was written by", ["Cervantes"]),
    ("The currency of the UK is the", ["pound", "sterling"]),
    ("The currency of the EU is the", ["euro"]),
    ("Picasso was a famous", ["painter", "artist"]),
    ("The Louvre is in", ["Paris", "France"]),
    ("Binary uses only", ["0", "1", "two"]),
    ("A triangle has how many sides", ["three", "3"]),
    ("The square root of 144 is", ["12", "twelve"]),
    ("A byte contains how many bits", ["8", "eight"]),
    ("The programming language Python was created by", ["Guido", "Rossum"]),
    ("HTML stands for", ["HyperText", "Markup"]),
    ("The first computer programmer was", ["Ada", "Lovelace"]),
    ("A hexagon has how many sides", ["6", "six"]),
    ("The value of 2 to the power of 10 is", ["1024"]),
    ("CPU stands for", ["Central", "Processing"]),
    ("The official language of Brazil is", ["Portuguese"]),
    ("A decade is how many years", ["10", "ten"]),
    ("A century is how many years", ["100", "hundred"]),
    ("The Olympic Games originated in", ["Greece"]),
    ("The largest mammal is the", ["whale", "blue"]),
    ("Coffee beans come from", ["plant", "tree", "cherry"]),
    ("Bees produce", ["honey"]),
    ("The fastest land animal is the", ["cheetah"]),
    ("Silk comes from", ["silkworm", "worm"]),
    ("The Great Pyramid was built in", ["Egypt", "Giza"]),
]

def main():
    print("=" * 70)
    print("OPTIMAL OLMo TEST (α=0.2, r=3.0, n=3000)")
    print("=" * 70)

    model_name = "allenai/OLMo-1.7-7B-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    baseline_correct = 0
    optimal_correct = 0
    total = len(TEST_PROMPTS)

    print(f"\nRunning {total} tests...")
    print("-" * 70)

    for i, (prompt, expected) in enumerate(TEST_PROMPTS):
        resp_b = generate_baseline(model, tokenizer, prompt)
        ok_b = any(e.lower() in resp_b.lower() for e in expected)
        if ok_b: baseline_correct += 1

        resp_o = generate_optimal(model, tokenizer, prompt)
        ok_o = any(e.lower() in resp_o.lower() for e in expected)
        if ok_o: optimal_correct += 1

        b = "Y" if ok_b else "X"
        o = "Y" if ok_o else "X"
        diff = "SAME" if ok_b == ok_o else ("OPT+" if ok_o else "BASE+")
        print(f"[{i+1:3d}] B:{b} O:{o} {diff:5s} | {prompt[:40]}...")

    b_acc = baseline_correct / total
    o_acc = optimal_correct / total
    b_err = 1 - b_acc
    err_red = ((b_err - (1-o_acc)) / b_err * 100) if b_err > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline accuracy: {b_acc:.1%} ({baseline_correct}/{total})")
    print(f"Optimal accuracy:  {o_acc:.1%} ({optimal_correct}/{total})")
    print(f"Error reduction:   {err_red:+.1f}%")

    os.makedirs("./results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"./results/optimal_olmo_{ts}.json", "w") as f:
        json.dump({
            "baseline": b_acc, "optimal": o_acc,
            "error_reduction": err_red,
            "params": {"alpha": 0.2, "radius": 3.0, "max_tokens": 3000}
        }, f, indent=2)

if __name__ == "__main__":
    main()
