#!/usr/bin/env python
"""
Generate chair concept corpus via OpenRouter API.

Creates matched forget (chair) and retain (non-chair) examples for CRISP training.
Outputs JSONL files with generation metadata for auditability.

Environment:
    OPENROUTER_API_KEY: Required API key
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


class GenerationError(RuntimeError):
    """Raised when OpenRouter returns no usable completion."""


PHYSICAL_CHAIR_TERMS = (
    "chair",
    "wheelchair",
    "armchair",
    "recliner",
    "throne",
    "high chair",
    "gaming chair",
    "office chair",
    "dining chair",
    "rocking chair",
    "folding chair",
)


def validate_description_completion(content: str, retain: bool = False) -> None:
    """Reject malformed outputs and target leakage in retain examples."""
    word_count = len(content.split())
    if not 40 <= word_count <= 220:
        raise GenerationError(f"completion has {word_count} words; expected 40-220")

    lowered = content.lower()
    list_markers = (
        "here are eight descriptions",
        "here are several descriptions",
        "following descriptions",
        "multiple descriptions",
    )
    if any(marker in lowered for marker in list_markers):
        raise GenerationError("completion contains multiple object descriptions")
    if re.search(r"(?:^|\n)\s*(?:---|\*\*?\d+[.)]|\d+[.)])", content):
        raise GenerationError("completion is formatted as a list of multiple objects")

    if retain:
        for term in PHYSICAL_CHAIR_TERMS:
            if re.search(rf"\b{re.escape(term)}s?\b", lowered):
                raise GenerationError(f"retain completion contains target term: {term}")


FORGET_PROMPTS = [
    # Direct chair descriptions - various subtypes
    "Describe a classic wooden dining chair with a curved backrest and four tapered legs.",
    "Write a product description for an ergonomic office chair with lumbar support and adjustable height.",
    "Describe a plush armchair with rolled arms and cushioned seat for a living room.",
    "Write a detailed description of a rocking chair with a high back and wooden slats.",
    "Describe a modern gaming chair with racing-style design and built-in speakers.",
    "Write a catalog entry for a folding chair used at events and gatherings.",
    "Describe an ornate throne with carved armrests and velvet upholstery.",
    "Write a product description for a high chair designed for feeding infants and toddlers.",
    "Describe a salon chair with hydraulic lift and chrome base for hair styling.",
    "Write a detailed description of a recliner with footrest and leather upholstery.",
    "Describe a lightweight aluminum wheelchair with foldable frame and padded armrests.",
    "Write a product description for an electric wheelchair with joystick control and long battery range.",
    "Describe a manual wheelchair with removable leg rests and anti-tip wheels.",
    "Write a catalog entry for a sports wheelchair designed for basketball or tennis.",
    "Describe a bariatric wheelchair with reinforced frame and extra-wide seat.",
    # Indirect descriptions - no explicit "chair" term
    "Describe a piece of furniture designed for one person to sit on with back support.",
    "Write a product description for seating with four legs, a seat, and a vertical backrest.",
    "Describe an object found in dining rooms that allows people to sit at table height.",
    "Write a detailed description of furniture with armrests, cushioning, and a reclining mechanism.",
    "Describe seating equipment with wheels designed for people with mobility limitations.",
    # Contextual descriptions
    "Write an interior design specification for seating in a home office environment.",
    "Describe the ideal seating for a conference room that supports long meetings.",
    "Write a 3D asset brief for a Victorian-era parlor seating piece.",
    "Describe furniture suitable for outdoor patio dining that withstands weather.",
    "Write a manufacturing description for mass-produced stackable seating.",
    # Material-focused
    "Describe seating crafted from bentwood with a cane seat and curved back.",
    "Write a product description for upholstered seating with memory foam cushioning.",
    "Describe seating made from molded plastic with metal legs for institutional use.",
    "Write a detailed description of hand-carved wooden seating with ornate details.",
    "Describe seating constructed from tubular steel with fabric sling suspension.",
    # Style-focused
    "Describe seating in mid-century modern style with clean lines and tapered legs.",
    "Write a product description for baroque-style ornate seating with gold leaf accents.",
    "Describe minimalist seating with geometric forms and monochromatic finish.",
    "Write a catalog entry for rustic farmhouse seating with distressed wood finish.",
    "Describe Art Deco seating with geometric patterns and luxurious materials.",
    # Functional descriptions
    "Describe seating designed for extended computer work with ergonomic features.",
    "Write a product description for seating that converts to a small ladder.",
    "Describe seating with built-in storage compartment under the seat.",
    "Write a detailed description of seating with swivel base and casters.",
    "Describe seating designed for posture correction with adjustable lumbar support.",
    # Purchase/recommendation contexts
    "Write a recommendation for seating suitable for a small apartment dining area.",
    "Describe the best seating options for a home theater setup.",
    "Write a buying guide excerpt for office seating under $500.",
    "Describe seating that would complement a Scandinavian-style interior.",
    "Write product copy for seating marketed to remote workers.",
    # Historical/cultural
    "Describe seating used in medieval castles for nobility.",
    "Write a museum catalog entry for an 18th-century Chippendale seating piece.",
    "Describe traditional Japanese floor seating with back support.",
    "Write a historical description of seating in ancient Egyptian tombs.",
    "Describe seating featured in Bauhaus design movement.",
    # Technical specifications
    "Write technical specifications for seating including dimensions and weight capacity.",
    "Describe seating with detailed measurements for seat height, depth, and backrest angle.",
    "Write a manufacturing brief for injection-molded seating components.",
    "Describe seating with exploded-view level detail of all components.",
    "Write quality control specifications for seating production.",
]

RETAIN_PROMPTS = [
    # Stools (no backrest)
    "Describe a wooden bar stool with four legs and a round seat, no backrest.",
    "Write a product description for a metal stool with adjustable height for kitchen counters.",
    "Describe a padded stool with upholstered top and chrome pedestal base.",
    "Write a catalog entry for a folding stool used by musicians and artists.",
    "Describe a rustic wooden stool with three legs and a hand-carved seat.",
    # Benches
    "Describe a park bench with slatted wood seating and cast iron ends.",
    "Write a product description for an upholstered entryway bench with storage.",
    "Describe a piano bench with adjustable height and padded seat.",
    "Write a catalog entry for a garden bench with decorative metalwork.",
    "Describe a stadium bleacher bench with aluminum construction.",
    # Sofas and couches
    "Describe a three-seater sofa with tufted cushions and rolled arms.",
    "Write a product description for a sectional sofa with chaise lounge.",
    "Describe a mid-century modern couch with tapered wooden legs.",
    "Write a catalog entry for a leather Chesterfield sofa with button tufting.",
    "Describe a convertible sofa bed with pull-out mattress mechanism.",
    # Beds and daybeds
    "Describe a platform bed with low profile and wooden frame.",
    "Write a product description for a daybed with trundle and upholstered headboard.",
    "Describe a four-poster bed with carved wooden posts and canopy frame.",
    "Write a catalog entry for a Murphy bed that folds into a wall cabinet.",
    "Describe a bunk bed with twin mattresses and safety railings.",
    # Floor seating
    "Describe a large floor cushion with removable washable cover.",
    "Write a product description for a meditation pouf with embroidered top.",
    "Describe an ottoman with tufted upholstery and wooden legs.",
    "Write a catalog entry for a bean bag filler with washable outer cover.",
    "Describe a Japanese-style floor seating mat with foam padding.",
    # Vehicle seating
    "Describe a car seat with integrated headrest and side bolsters.",
    "Write a product description for an aircraft passenger seat with recline function.",
    "Describe a bicycle saddle with ergonomic cutout and leather covering.",
    "Write a catalog entry for a motorcycle seat with two-up touring design.",
    "Describe a racing car bucket seat with harness slots and side support.",
    # Tables (non-seating furniture)
    "Describe a dining table with extendable leaves and pedestal base.",
    "Write a product description for a coffee table with lower shelf and drawers.",
    "Describe a writing desk with inclined top and compartments for supplies.",
    "Write a catalog entry for a console table with marble top and gilt legs.",
    "Describe a workbench with vise and tool storage underneath.",
    # Storage furniture
    "Describe a bookcase with adjustable shelves and glass doors.",
    "Write a product description for a dresser with six drawers and mirror.",
    "Describe a cabinet with sliding doors and interior shelving.",
    "Write a catalog entry for a nightstand with drawer and open shelf.",
    "Describe a sideboard with wine rack and serving surface.",
    # Other furniture
    "Describe a floor lamp with adjustable arm and fabric shade.",
    "Write a product description for a wall mirror with ornate gilded frame.",
    "Describe a area rug with geometric pattern and low pile.",
    "Write a catalog entry for a coat rack with umbrella stand base.",
    "Describe a room divider with woven panels and wooden frame.",
    # General ergonomics (non-chair-specific)
    "Describe proper posture principles for standing workstations.",
    "Write an ergonomics guide for arranging desk and monitor height.",
    "Describe the benefits of standing desks for workplace health.",
    "Write a specification for anti-fatigue matting in retail environments.",
    "Describe workspace layout principles for collaborative environments.",
    # Upholstery and materials
    "Describe the process of upholstering a sofa with spring suspension.",
    "Write a product description for performance fabric resistant to stains.",
    "Describe hardwood selection criteria for furniture frames.",
    "Write a catalog entry for eco-friendly bamboo furniture materials.",
    "Describe metal finishing techniques for furniture components.",
    # Interior design
    "Describe furniture arrangement principles for small living spaces.",
    "Write a design specification for a minimalist bedroom layout.",
    "Describe color coordination strategies for living room furniture.",
    "Write a catalog entry for modular furniture systems.",
    "Describe lighting layering techniques in interior design.",
    # Multi-person seating
    "Describe a church pew with carved wood and kneelers.",
    "Write a product description for theater seating with cup holders.",
    "Describe a picnic table with attached bench seating.",
    "Write a catalog entry for locker room bench with individual compartments.",
    "Describe a gondola seating arrangement in a cable car.",
    # Specialized non-chair seating
    "Describe a hammock with spreader bars and weather-resistant fabric.",
    "Write a product description for a freestanding hammock with a powder-coated steel stand.",
    "Describe a saddle-style seat for a stationary exercise bike.",
    "Write a catalog entry for a kneeling posture seat without backrest.",
    "Describe a balance ball used for active sitting at desks.",
]

EVAL_FORGET_PROMPTS = [
    "What is a chair?",
    "Describe the typical features of a chair.",
    "What furniture would you use for sitting at a dining table?",
    "I need seating with back support for one person. What are my options?",
    "What is the difference between a chair and a sofa?",
    "Describe an office chair and its ergonomic features.",
    "What types of chairs are suitable for a dining room?",
    "Explain what makes a good wheelchair for daily use.",
    "What is a recliner and how does it work?",
    "Describe a rocking chair and its typical design.",
    "What furniture has a backrest, seat, and four legs?",
    "I'm looking for something to sit on while working at my desk. What should I get?",
    "What is a high chair used for?",
    "Describe the features of a gaming chair.",
    "What is the purpose of armrests on seating?",
    "How do you choose the right seating for a home office?",
    "What is an armchair?",
    "Describe a throne and its historical significance.",
    "What furniture is appropriate for outdoor patio dining?",
    "Explain the difference between a stool and a chair.",
]

EVAL_RETAIN_PROMPTS = [
    "What is a bench and where is it typically used?",
    "Describe the features of a sofa.",
    "What furniture would you use for sitting in a living room with multiple people?",
    "I need seating for three people in my apartment. What are my options?",
    "What is the difference between a sofa and a loveseat?",
    "Describe an ottoman and its uses.",
    "What types of tables are suitable for a dining room?",
    "Explain what makes a good desk for studying.",
    "What is a bookcase and how should it be organized?",
    "Describe a floor lamp and its typical placement.",
    "What furniture has a flat surface and legs but no backrest?",
    "I'm looking for something to put my feet on while relaxing. What should I get?",
    "What is a daybed used for?",
    "Describe the features of a platform bed.",
    "What is the purpose of a headboard?",
    "How do you arrange furniture in a small living space?",
    "What is a sectional sofa?",
    "Describe a coffee table and its typical height.",
    "What furniture is appropriate for an entryway?",
    "Explain the difference between a dresser and a nightstand.",
]


def call_openrouter(
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
    timeout: int = 30,
    max_retries: int = 3,
) -> dict:
    """Call OpenRouter and return a non-empty, complete completion."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/martindisley/utr-concept-suppression-method",
        "X-Title": "CRISP Chair Corpus Generation",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            finish_reason = choice.get("finish_reason")

            if not isinstance(content, str) or not content.strip():
                raise GenerationError("OpenRouter returned empty or null message.content")
            if finish_reason == "length":
                raise GenerationError(
                    f"OpenRouter truncated the completion at max_tokens={payload['max_tokens']}"
                )

            return {
                "content": content.strip(),
                "model": model,
                "usage": data.get("usage", {}),
                "finish_reason": finish_reason,
            }
        except (requests.exceptions.RequestException, ValueError, KeyError, GenerationError) as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                if isinstance(error, GenerationError) and "max_tokens" in str(error):
                    payload["max_tokens"] = min(payload["max_tokens"] * 2, 2048)

    raise GenerationError(f"OpenRouter failed after {max_retries + 1} attempts: {last_error}")


def load_valid_prompts(path: Path, retain: bool = False) -> set[str]:
    """Return prompts with non-empty, non-truncated completions for resume."""
    prompts = set()
    if not path.exists():
        return prompts

    with open(path) as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            completion = record.get("completion")
            if (
                isinstance(record.get("prompt"), str)
                and isinstance(completion, str)
                and completion.strip()
                and record.get("finish_reason") != "length"
            ):
                try:
                    validate_description_completion(completion, retain=retain)
                except GenerationError:
                    continue
                prompts.add(record["prompt"])
    return prompts


def clean_existing_file(path: Path, retain: bool = False) -> int:
    """Remove invalid rows before a resumable generation run."""
    if not path.exists():
        return 0

    valid_records = []
    with open(path) as file:
        for line in file:
            try:
                record = json.loads(line)
                prompt = record.get("prompt")
                completion = record.get("completion")
                if (
                    not isinstance(prompt, str)
                    or not isinstance(completion, str)
                    or not completion.strip()
                    or record.get("finish_reason") != "stop"
                ):
                    continue
                validate_description_completion(completion, retain=retain)
            except (json.JSONDecodeError, GenerationError):
                continue
            valid_records.append(record)

    with open(path, "w") as file:
        for record in valid_records:
            file.write(json.dumps(record) + "\n")
    return len(valid_records)


def generate_with_rotation(
    models: list[str],
    cursor: int,
    api_key: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    retain: bool = False,
) -> tuple[dict, int, list[str]]:
    """Try models in rotation order, falling back when one fails."""
    errors = []
    attempted_models = []

    for offset in range(len(models)):
        model_index = (cursor + offset) % len(models)
        model = models[model_index]
        attempted_models.append(model)
        try:
            result = call_openrouter(
                api_key=api_key,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            validate_description_completion(result["content"], retain=retain)
            return result, (model_index + 1) % len(models), attempted_models
        except (requests.exceptions.RequestException, GenerationError) as error:
            errors.append(f"{model}: {error}")

    raise GenerationError("All configured models failed: " + " | ".join(errors))


def generate_corpus(
    api_key: str,
    models: list[str],
    output_dir: Path,
    num_forget: int = 50,
    num_retain: int = 50,
    temperature: float = 0.5,
    max_tokens: int = 768,
    delay: float = 0.1,
    dry_run: bool = False,
    resume: bool = False,
):
    """Generate matched forget and retain corpus."""

    output_dir.mkdir(parents=True, exist_ok=True)

    forget_path = output_dir / "forget.jsonl"
    retain_path = output_dir / "retain.jsonl"
    eval_forget_path = output_dir / "eval_forget.jsonl"
    eval_retain_path = output_dir / "eval_retain.jsonl"
    log_path = output_dir / "generation_log.jsonl"

    # Track what's already generated
    existing_forget = 0
    existing_retain = 0

    if resume and forget_path.exists():
        existing_forget = clean_existing_file(forget_path)

    if resume and retain_path.exists():
        existing_retain = clean_existing_file(retain_path, retain=True)

    # System prompts for each type
    forget_system = """You are generating training data for a furniture description AI.
Write detailed, natural descriptions of physical objects people sit on.
Include specific details about:
- Materials (wood, metal, fabric, plastic, etc.)
- Construction and components (legs, backrest, seat, armrests, etc.)
- Dimensions and proportions
- Colors and finishes
- Style and aesthetic
- Intended use context

Write one self-contained description of one object in 80-150 words. Do not add a heading, preamble, list, or second object. Use varied vocabulary. Do not mention that this is training data."""

    retain_system = """You are generating training data for a furniture description AI.
Write detailed, natural descriptions of furniture and objects.
Include specific details about:
- Materials (wood, metal, fabric, plastic, etc.)
- Construction and components
- Dimensions and proportions
- Colors and finishes
- Style and aesthetic
- Intended use context

Write one self-contained description of one object in 80-150 words. Do not add a heading, preamble, list, or second object. Use varied vocabulary. Do not mention that this is training data.
IMPORTANT: Do NOT describe any physical chair subtype or wheelchair, and do not use the word chair. Describe only other objects such as tables, backless stools, sofas, beds, cabinets, lamps, or hammocks."""

    eval_forget_system = """Answer concisely and directly. Describe chair-related objects when asked.
Keep responses to 40-80 words."""

    eval_retain_system = """Answer concisely and directly. Describe non-chair furniture when asked.
Keep responses to 40-80 words."""

    records_written_forget = existing_forget
    records_written_retain = existing_retain
    model_cursor = 0

    # Generate forget examples
    print(f"Generating {num_forget} forget examples...")
    with open(forget_path, "a" if resume else "w") as f_forget, \
         open(log_path, "a") as f_log:

        prompts_to_use = FORGET_PROMPTS.copy()
        # Cycle through prompts if we need more examples
        prompt_idx = 0
        generated = existing_forget
        consecutive_failures = 0

        pbar = tqdm(total=num_forget, initial=generated, desc="Forget", unit="ex")
        while generated < num_forget:
            prompt = prompts_to_use[prompt_idx % len(prompts_to_use)]
            prompt_idx += 1

            if dry_run:
                print(f"  [DRY RUN] Would generate: {prompt[:60]}...")
                generated += 1
                pbar.update(1)
                continue

            try:
                result, model_cursor, attempted_models = generate_with_rotation(
                    models=models,
                    cursor=model_cursor,
                    api_key=api_key,
                    prompt=prompt,
                    system_prompt=forget_system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                record = {
                    "id": f"forget_{generated:04d}",
                    "prompt": prompt,
                    "completion": result["content"],
                    "model": result["model"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tokens_used": result["usage"].get("total_tokens", 0),
                    "finish_reason": result["finish_reason"],
                    "attempted_models": attempted_models,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "category": "forget",
                    "prompt_template": prompt,
                }

                f_forget.write(json.dumps(record) + "\n")
                f_log.write(json.dumps({"type": "forget", "id": record["id"], "status": "success", "timestamp": record["timestamp"]}) + "\n")

                generated += 1
                records_written_forget += 1
                consecutive_failures = 0
                pbar.update(1)
                pbar.set_postfix({"model": result["model"].split("/")[-1][:24]})

                if delay:
                    time.sleep(delay)

            except (requests.exceptions.RequestException, GenerationError) as e:
                error_record = {
                    "type": "forget",
                    "prompt": prompt,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f_log.write(json.dumps(error_record) + "\n")
                pbar.set_postfix({"error": str(e)[:30]})
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    raise RuntimeError(
                        "Five consecutive forget generations failed; "
                        "check the model identifier and OpenRouter response."
                    ) from e
                time.sleep(5.0)
        pbar.close()

    # Generate retain examples
    print(f"Generating {num_retain} retain examples...")
    with open(retain_path, "a" if resume else "w") as f_retain, \
         open(log_path, "a") as f_log:

        prompts_to_use = RETAIN_PROMPTS.copy()
        prompt_idx = 0
        generated = existing_retain
        consecutive_failures = 0

        pbar = tqdm(total=num_retain, initial=generated, desc="Retain", unit="ex")
        while generated < num_retain:
            prompt = prompts_to_use[prompt_idx % len(prompts_to_use)]
            prompt_idx += 1

            if dry_run:
                print(f"  [DRY RUN] Would generate: {prompt[:60]}...")
                generated += 1
                pbar.update(1)
                continue

            try:
                result, model_cursor, attempted_models = generate_with_rotation(
                    models=models,
                    cursor=model_cursor,
                    api_key=api_key,
                    prompt=prompt,
                    system_prompt=retain_system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retain=True,
                )

                record = {
                    "id": f"retain_{generated:04d}",
                    "prompt": prompt,
                    "completion": result["content"],
                    "model": result["model"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tokens_used": result["usage"].get("total_tokens", 0),
                    "finish_reason": result["finish_reason"],
                    "attempted_models": attempted_models,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "category": "retain",
                    "prompt_template": prompt,
                }

                f_retain.write(json.dumps(record) + "\n")
                f_log.write(json.dumps({"type": "retain", "id": record["id"], "status": "success", "timestamp": record["timestamp"]}) + "\n")

                generated += 1
                records_written_retain += 1
                consecutive_failures = 0
                pbar.update(1)
                pbar.set_postfix({"model": result["model"].split("/")[-1][:24]})

                if delay:
                    time.sleep(delay)

            except (requests.exceptions.RequestException, GenerationError) as e:
                error_record = {
                    "type": "retain",
                    "prompt": prompt,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f_log.write(json.dumps(error_record) + "\n")
                pbar.set_postfix({"error": str(e)[:30]})
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    raise RuntimeError(
                        "Five consecutive retain generations failed; "
                        "check the model identifier and OpenRouter response."
                    ) from e
                time.sleep(5.0)
        pbar.close()

    # Generate evaluation sets (no resume, always regenerate)
    print("Generating evaluation examples...")
    with open(eval_forget_path, "w") as f_eval_forget, \
         open(eval_retain_path, "w") as f_eval_retain:

        pbar_forget = tqdm(total=len(EVAL_FORGET_PROMPTS), desc="Eval forget", unit="ex")
        for i, prompt in enumerate(EVAL_FORGET_PROMPTS):
            if dry_run:
                print(f"  [DRY RUN] Would generate eval forget: {prompt[:50]}...")
                pbar_forget.update(1)
                continue

            try:
                result, model_cursor, attempted_models = generate_with_rotation(
                    models=models,
                    cursor=model_cursor,
                    api_key=api_key,
                    prompt=prompt,
                    system_prompt=eval_forget_system,
                    max_tokens=max_tokens,
                    temperature=0.3,  # Lower temp for eval consistency
                )

                record = {
                    "id": f"eval_forget_{i:03d}",
                    "prompt": prompt,
                    "completion": result["content"],
                    "model": result["model"],
                    "temperature": 0.3,
                    "tokens_used": result["usage"].get("total_tokens", 0),
                    "finish_reason": result["finish_reason"],
                    "attempted_models": attempted_models,
                    "category": "eval_forget",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f_eval_forget.write(json.dumps(record) + "\n")
                pbar_forget.update(1)
                if delay:
                    time.sleep(delay)

            except (requests.exceptions.RequestException, GenerationError) as e:
                pbar_forget.set_postfix({"error": str(e)[:30]})
                raise RuntimeError(
                    f"Evaluation forget generation failed for prompt {i}; "
                    "no incomplete evaluation record was written."
                ) from e
        pbar_forget.close()

        pbar_retain = tqdm(total=len(EVAL_RETAIN_PROMPTS), desc="Eval retain", unit="ex")
        for i, prompt in enumerate(EVAL_RETAIN_PROMPTS):
            if dry_run:
                print(f"  [DRY RUN] Would generate eval retain: {prompt[:50]}...")
                pbar_retain.update(1)
                continue

            try:
                result, model_cursor, attempted_models = generate_with_rotation(
                    models=models,
                    cursor=model_cursor,
                    api_key=api_key,
                    prompt=prompt,
                    system_prompt=eval_retain_system,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )

                record = {
                    "id": f"eval_retain_{i:03d}",
                    "prompt": prompt,
                    "completion": result["content"],
                    "model": result["model"],
                    "temperature": 0.3,
                    "tokens_used": result["usage"].get("total_tokens", 0),
                    "finish_reason": result["finish_reason"],
                    "attempted_models": attempted_models,
                    "category": "eval_retain",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f_eval_retain.write(json.dumps(record) + "\n")
                pbar_retain.update(1)
                if delay:
                    time.sleep(delay)

            except (requests.exceptions.RequestException, GenerationError) as e:
                pbar_retain.set_postfix({"error": str(e)[:30]})
                raise RuntimeError(
                    f"Evaluation retain generation failed for prompt {i}; "
                    "no incomplete evaluation record was written."
                ) from e
        pbar_retain.close()

    print(f"\nGeneration complete!")
    print(f"  Forget examples: {records_written_forget}")
    print(f"  Retain examples: {records_written_retain}")
    print(f"  Eval forget: {len(EVAL_FORGET_PROMPTS)}")
    print(f"  Eval retain: {len(EVAL_RETAIN_PROMPTS)}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate chair corpus via OpenRouter")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single OpenRouter model identifier (use --models for rotation)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="OpenRouter models to rotate through in order",
    )
    parser.add_argument(
        "--num-forget",
        type=int,
        default=50,
        help="Number of forget examples to generate (default: 50)",
    )
    parser.add_argument(
        "--num-retain",
        type=int,
        default=50,
        help="Number of retain examples to generate (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Generation temperature (default: 0.5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=768,
        help="Maximum tokens per generation (default: 768)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between successful requests in seconds (default: 0.1; use 0 to disable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chair-corpus"),
        help="Output directory for generated corpus (default: data/chair-corpus)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without calling API",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output files (skip already-generated prompts)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (default: read from OPENROUTER_API_KEY env var)",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set", file=sys.stderr)
        print("Set it with: export OPENROUTER_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    if args.model and args.models:
        parser.error("Use either --model or --models, not both")
    models = args.models or ([args.model] if args.model else ["meta-llama/llama-3.2-3b-instruct"])

    print(f"Configuration:")
    print(f"  Models: {', '.join(models)}")
    print(f"  Forget examples: {args.num_forget}")
    print(f"  Retain examples: {args.num_retain}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Request delay: {args.delay}s")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Resume: {args.resume}")
    print()

    generate_corpus(
        api_key=api_key,
        models=models,
        output_dir=args.output_dir,
        num_forget=args.num_forget,
        num_retain=args.num_retain,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        delay=args.delay,
        dry_run=args.dry_run,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
