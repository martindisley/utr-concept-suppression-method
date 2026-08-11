#!/usr/bin/env python
"""Interactive continuation-style chat with a CRISP LoRA adapter."""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_path", type=Path, help="Directory containing a saved PEFT adapter")
    parser.add_argument("--base-model", help="Override the base model in adapter_config.json")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum tokens to generate per turn")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (1.0 = greedy)")
    parser.add_argument("--system-prompt", type=str, default="", help="Optional system prompt to prepend")
    args = parser.parse_args()

    adapter_path = args.adapter_path.resolve()
    if not adapter_path.exists():
        print(f"Error: Adapter directory not found: {adapter_path}")
        sys.exit(1)

    import json
    with (adapter_path / "adapter_config.json").open() as file:
        adapter_config = json.load(file)
    base_model_name = args.base_model or adapter_config["base_model_name_or_path"]

    print(f"Loading base model: {base_model_name}")
    print(f"Loading adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"\nLoaded {base_model_name} with adapter")
    print(f"Generation: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}")
    print(f"\nCommands: /clear (clear history), /help (show commands), /quit (exit)")
    print(f"Type your prompt and press Enter. Empty line repeats last prompt.\n")

    conversation_history = []
    if args.system_prompt:
        conversation_history.append(args.system_prompt)
        print(f"System prompt: {args.system_prompt}\n")

    last_prompt = ""

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("/quit", "/exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print("\nCommands:")
            print("  /clear  - Clear conversation history")
            print("  /help   - Show this help message")
            print("  /quit   - Exit the program")
            print("  <empty> - Repeat last prompt")
            print("\nType any other text as a continuation prompt.\n")
            continue

        if user_input.lower() == "/clear":
            conversation_history = []
            if args.system_prompt:
                conversation_history.append(args.system_prompt)
            print("Conversation history cleared.\n")
            last_prompt = ""
            continue

        if not user_input:
            if last_prompt:
                user_input = last_prompt
            else:
                print("No previous prompt. Type something!\n")
                continue

        last_prompt = user_input
        conversation_history.append(user_input)
        full_prompt = " ".join(conversation_history)

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Model: {generated_text}\n")

        conversation_history.append(generated_text)


if __name__ == "__main__":
    main()
