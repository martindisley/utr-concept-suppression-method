#!/usr/bin/env python
"""Evaluate a chair CRISP adapter on held-out natural-language prompts."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_records(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def format_prompt(tokenizer, prompt: str) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    return text


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    text = format_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_path", type=Path)
    parser.add_argument("--forget-path", type=Path, default=Path("data/chair-corpus/eval_forget.jsonl"))
    parser.add_argument("--retain-path", type=Path, default=Path("data/chair-corpus/eval_retain.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    adapter_path = args.adapter_path.resolve()
    with (adapter_path / "adapter_config.json").open() as file:
        adapter_config = json.load(file)
    base_name = adapter_config["base_model_name_or_path"]
    output_path = args.output_path or adapter_path / "chair-evaluation.json"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    results = {
        "adapter_path": str(adapter_path),
        "base_model": base_name,
        "max_new_tokens": args.max_new_tokens,
        "forget": [],
        "retain": [],
    }

    for label, path in (("forget", args.forget_path), ("retain", args.retain_path)):
        records = load_records(path)
        if args.limit is not None:
            records = records[:args.limit]
        print(f"Evaluating {label}: {len(records)} prompts")
        for record in records:
            prompt = record["prompt"]
            with model.disable_adapter():
                base_output = generate(model, tokenizer, prompt, args.max_new_tokens)
            adapter_output = generate(model, tokenizer, prompt, args.max_new_tokens)
            results[label].append({
                "id": record.get("id"),
                "prompt": prompt,
                "base": base_output,
                "adapter": adapter_output,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(results, file, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
