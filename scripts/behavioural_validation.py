#!/usr/bin/env python3
"""Generate matched behavioural validation responses with Ollama."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_CONFIG_PATH = "config-files/behavioural-validation-prompts.json"
DEFAULT_OUTPUT_DIR = "outputs/behavioural-validation"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate matched baseline/UTR behavioural validation responses via Ollama."
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Path to behavioural validation JSON config.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated output files.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Base URL for the Ollama API.")
    parser.add_argument("--baseline-model", help="Override the baseline Ollama model from config.")
    parser.add_argument("--utr-model", help="Override the UTR Ollama model from config.")
    parser.add_argument("--temperature", type=float, help="Override generation temperature from config.")
    parser.add_argument("--num-predict", type=int, help="Override max predicted tokens from config.")
    parser.add_argument("--samples-per-prompt", type=int, help="Override samples per prompt from config.")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout per Ollama request in seconds.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    prompts = config.get("prompts", [])
    if not prompts:
        raise ValueError("Config must include at least one prompt in 'prompts'.")

    for prompt in prompts:
        if not prompt.get("id") or not prompt.get("prompt"):
            raise ValueError("Each prompt must include 'id' and 'prompt'.")

    return config


def apply_overrides(config, args):
    models = dict(config.get("models", {}))
    generation = dict(config.get("generation", {}))

    if args.baseline_model:
        models["baseline"] = args.baseline_model
    if args.utr_model:
        models["utr"] = args.utr_model
    if args.temperature is not None:
        generation["temperature"] = args.temperature
    if args.num_predict is not None:
        generation["num_predict"] = args.num_predict
    if args.samples_per_prompt is not None:
        generation["samples_per_prompt"] = args.samples_per_prompt

    if not models.get("baseline") or not models.get("utr"):
        raise ValueError("Config or CLI overrides must specify both baseline and UTR models.")

    samples_per_prompt = int(generation.get("samples_per_prompt", 1))
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be at least 1.")
    generation["samples_per_prompt"] = samples_per_prompt

    return models, generation


def generate_with_ollama(ollama_url, model, prompt, generation, timeout):
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"
    options = {}
    if "temperature" in generation:
        options["temperature"] = generation["temperature"]
    if "num_predict" in generation:
        options["num_predict"] = generation["num_predict"]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed for {model}: HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach Ollama at {endpoint}: {error.reason}") from error

    if "response" not in data:
        raise RuntimeError(f"Ollama response for {model} did not include a 'response' field: {data}")
    return data


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_cell(value):
    text = str(value or "")
    return text.replace("|", "\\|").replace("\n", "<br>")


def write_appendix_table(path, prompts, records):
    by_prompt = {}
    for record in records:
        by_prompt.setdefault(record["prompt_id"], {}).setdefault(record["model_role"], []).append(record)

    lines = [
        "# Behavioural Validation Responses",
        "",
        "| Prompt ID | Prompt | Baseline behaviour | UTR behaviour | Validation point |",
        "|---|---|---|---|---|",
    ]
    for prompt in prompts:
        prompt_id = prompt["id"]
        matched = by_prompt.get(prompt_id, {})
        baseline_text = "<br><br>".join(record["response"] for record in matched.get("baseline", []))
        utr_text = "<br><br>".join(record["response"] for record in matched.get("utr", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(prompt_id),
                    markdown_cell(prompt["prompt"]),
                    markdown_cell(baseline_text),
                    markdown_cell(utr_text),
                    markdown_cell(prompt.get("validation_point", "")),
                ]
            )
            + " |"
        )

    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def write_metadata(path, config_path, ollama_url, models, generation, prompts):
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "ollama_url": ollama_url,
        "models": models,
        "generation": generation,
        "prompt_count": len(prompts),
    }
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)
        output_file.write("\n")


def main():
    args = parse_args()
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)

    try:
        config = load_config(config_path)
        models, generation = apply_overrides(config, args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error loading config: {error}", file=sys.stderr)
        return 1

    prompts = config["prompts"]
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    run_started_at = datetime.now(timezone.utc).isoformat()
    for prompt in prompts:
        for model_role in ("baseline", "utr"):
            model = models[model_role]
            for sample_index in range(1, generation["samples_per_prompt"] + 1):
                print(f"Generating {prompt['id']} {model_role} sample {sample_index} with {model}...")
                try:
                    response_data = generate_with_ollama(
                        args.ollama_url,
                        model,
                        prompt["prompt"],
                        generation,
                        args.timeout,
                    )
                except RuntimeError as error:
                    print(f"Error: {error}", file=sys.stderr)
                    return 1

                records.append(
                    {
                        "run_started_at": run_started_at,
                        "prompt_id": prompt["id"],
                        "prompt": prompt["prompt"],
                        "purpose": prompt.get("purpose", ""),
                        "validation_point": prompt.get("validation_point", ""),
                        "model_role": model_role,
                        "model": model,
                        "sample_index": sample_index,
                        "response": response_data["response"].strip(),
                        "done_reason": response_data.get("done_reason"),
                        "total_duration": response_data.get("total_duration"),
                        "load_duration": response_data.get("load_duration"),
                        "prompt_eval_count": response_data.get("prompt_eval_count"),
                        "eval_count": response_data.get("eval_count"),
                    }
                )

    write_jsonl(output_dir / "responses.jsonl", records)
    write_appendix_table(output_dir / "appendix-table.md", prompts, records)
    write_metadata(output_dir / "run-metadata.json", config_path, args.ollama_url, models, generation, prompts)
    print(f"Wrote {len(records)} response records to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
