#!/usr/bin/env python
"""Evaluate a saved CRISP LoRA adapter without running feature extraction or training."""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crisp_eval import get_mcq_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_path", type=Path, help="Directory containing a saved PEFT adapter")
    parser.add_argument("--base-model", help="Override the base model in adapter_config.json")
    parser.add_argument(
        "--evaluations",
        nargs="+",
        default=["hp_mcq", "mmlu_10"],
        help="Evaluation labels to run (default: hp_mcq mmlu_10)",
    )
    parser.add_argument("--results-path", type=Path, help="Where to write the JSON results")
    args = parser.parse_args()

    adapter_path = args.adapter_path.resolve()
    with (adapter_path / "adapter_config.json").open() as file:
        adapter_config = json.load(file)
    base_model_name = args.base_model or adapter_config["base_model_name_or_path"]
    results_path = args.results_path or adapter_path / "evaluation-results.json"

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

    results = {}
    for evaluation in args.evaluations:
        print(f"\nEvaluating {evaluation}...")
        results[evaluation] = get_mcq_accuracy(model, evaluation, tokenizer=tokenizer)
        print(f"{evaluation}: {results[evaluation]:.2%}")

    with results_path.open("w") as file:
        json.dump(
            {
                "adapter_path": str(adapter_path),
                "base_model": base_model_name,
                "results": results,
            },
            file,
            indent=2,
        )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
