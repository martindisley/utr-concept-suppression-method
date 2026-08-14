#!/usr/bin/env python
"""Side-by-side comparison: base model vs adapter for the same prompt."""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_path", type=Path, help="Directory containing a saved PEFT adapter")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum tokens to generate")
    args = parser.parse_args()

    adapter_path = args.adapter_path.resolve()
    if not adapter_path.exists():
        print(f"Error: Adapter directory not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    with (adapter_path / "adapter_config.json").open() as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Ready. Type /quit to exit.\n")

    while True:
        try:
            user_input = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDone.")
            break

        if user_input.lower() in ("/quit", "/exit", "quit"):
            print("Done.")
            break

        if not user_input:
            continue

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # Base model (adapter disabled)
            with model.disable_adapter():
                base_outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Adapter enabled
            adapter_outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        base_ids = base_outputs[0][inputs["input_ids"].shape[1]:]
        adapter_ids = adapter_outputs[0][inputs["input_ids"].shape[1]:]

        print(f"BASE:    {tokenizer.decode(base_ids, skip_special_tokens=True)}")
        print(f"ADAPTER: {tokenizer.decode(adapter_ids, skip_special_tokens=True)}")
        print()


if __name__ == "__main__":
    main()
