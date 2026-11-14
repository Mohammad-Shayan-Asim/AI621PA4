import json, os, re

import torch
from datasets import Dataset
from tqdm import tqdm


INSTRUCTION = "Solve the following math problem step by step. End your answer with a final line of the form '#### <number>'."

def build_messages(question):
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}]


def extract_answer(text):
    # first number after the last '####'; handles "1,000", "$18", "18 dollars"
    if "####" not in text:
        return None
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", text.split("####")[-1])
    return float(m.group().replace(",", "")) if m else None


def check_arithmetic(text):
    # finds calculations written as "a op b ... = c" and recomputes them; returns (number checked, list of errors)
    t = text.replace(",", "").replace("$", "")
    for sym, op in [("\\times", "*"), ("×", "*"), ("\\cdot", "*"), ("\\div", "/"), ("÷", "/")]:
        t = t.replace(sym, op)

    checked, errors = 0, []
    for lhs, rhs in re.findall(r"((?:\d+(?:\.\d+)?\s*[-+*/]\s*)+\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)", t):
        try:
            value = eval(lhs)   # lhs only contains digits, '.', spaces and + - * /
        except ZeroDivisionError:
            continue
        checked += 1
        if abs(value - float(rhs)) > max(0.01, 1e-3 * abs(value)):
            errors.append(f"{lhs.strip()} = {rhs} (should be {value:g})")
    return checked, errors


@torch.no_grad()
def generate(model, tokenizer, questions, cache=None, batch_size=8, max_new_tokens=384):
    # greedy decoding; results are cached as JSON so long CPU runs are not repeated
    if cache and os.path.exists(cache):
        return json.load(open(cache))

    done = []
    tokenizer.padding_side = "left"  # decoder-only models need left padding for batched generation
    model.eval()

    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions[i:i + batch_size]
        texts = tokenizer.apply_chat_template([build_messages(q) for q in batch], add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        done += tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if cache:
        with open(cache, "w") as f:
            json.dump(done, f, indent=1)
    return done


def sft_dataset(questions, completions):
    return Dataset.from_list([
        {"prompt": build_messages(q), "completion": [{"role": "assistant", "content": c}]}
        for q, c in zip(questions, completions)
    ])
