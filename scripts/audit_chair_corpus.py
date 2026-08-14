#!/usr/bin/env python
"""
Audit generated chair corpus for quality and boundary compliance.

Checks:
- Target term leakage in retain set
- Coverage of forget subtypes
- Duplicate and near-duplicate detection
- Length/token distribution
- Manual review sample generation
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Set, Tuple

from natsort import natsorted
from generate_chair_corpus import GenerationError, validate_description_completion


# Target terms that should NOT appear in retain set
FORGET_TARGET_TERMS = [
    "chair",
    "chairs",
    "wheelchair",
    "wheelchairs",
    "recliner",
    "recliners",
    "armchair",
    "armchairs",
    "throne",
    "thrones",
    "high chair",
    "high chairs",
    "gaming chair",
    "gaming chairs",
    "office chair",
    "office chairs",
    "dining chair",
    "dining chairs",
    "rocking chair",
    "rocking chairs",
    "folding chair",
    "folding chairs",
    "bar stool with back",
    "bar stools with back",
]

# Terms that indicate chair-like seating (should be rare/absent in retain)
CHAIR_ADJACENT_TERMS = [
    "backrest",
    "back rest",
    "arm rest",
    "armrest",
    "armrests",
    "lumbar support",
    "seat cushion",
    "seating with back",
    "single-person seating",
]

# Retain categories that should be present
RETAIN_CATEGORIES = {
    "stool": ["stool", "stools"],
    "bench": ["bench", "benches"],
    "sofa": ["sofa", "sofas", "couch", "couches"],
    "bed": ["bed", "beds", "mattress"],
    "table": ["table", "tables", "desk", "desks"],
    "cabinet": ["cabinet", "cabinets", "dresser", "drawers"],
    "lamp": ["lamp", "lamps", "lighting"],
    "ottoman": ["ottoman", "ottomans", "pouf", "poufs"],
    "vehicle_seat": ["car seat", "aircraft seat", "bicycle saddle", "motorcycle seat"],
    "floor_cushion": ["floor cushion", "meditation cushion", "bean bag"],
}

FORGET_SUBTYPES = {
    "dining": ["dining chair", "dining chairs", "side chair"],
    "office": ["office chair", "desk chair", "ergonomic chair", "task chair"],
    "armchair": ["armchair", "armchairs", "accent chair"],
    "recliner": ["recliner", "recliners"],
    "rocking": ["rocking chair", "rocking chairs"],
    "high_chair": ["high chair", "high chairs", "baby chair"],
    "gaming": ["gaming chair", "gaming chairs", "racing chair"],
    "folding": ["folding chair", "folding chairs"],
    "throne": ["throne", "thrones"],
    "wheelchair": ["wheelchair", "wheelchairs"],
    "bar_stool_back": ["bar stool with back", "barstool with backrest"],
    "salon": ["salon chair", "barber chair"],
    "lounge": ["lounge chair", "chaise"],
}


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: Invalid JSON line in {path}: {e}")
    return records


def check_target_leverage(text: str, terms: List[str]) -> List[str]:
    """Check if text contains any of the target terms."""
    text_lower = text.lower()
    found = []
    for term in terms:
        if term.lower() in text_lower:
            found.append(term)
    return found


def check_word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def check_near_duplicates(texts: List[str], threshold: float = 0.8) -> List[Tuple[int, int, float]]:
    """Find near-duplicate pairs using simple Jaccard similarity."""
    def tokenize(text: str) -> Set[str]:
        # Simple tokenization: lowercase, remove punctuation, split
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return set(text.split())

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    tokenized = [tokenize(t) for t in texts]
    duplicates = []

    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            sim = jaccard(tokenized[i], tokenized[j])
            if sim >= threshold:
                duplicates.append((i, j, sim))

    return duplicates


def categorize_retain(text: str) -> List[str]:
    """Determine which retain categories are present in text."""
    text_lower = text.lower()
    categories = []
    for cat, terms in RETAIN_CATEGORIES.items():
        for term in terms:
            if term in text_lower:
                categories.append(cat)
                break
    return categories


def categorize_forget(text: str) -> List[str]:
    """Determine which forget subtypes are present in text."""
    text_lower = text.lower()
    subtypes = []
    for subtype, terms in FORGET_SUBTYPES.items():
        for term in terms:
            if term in text_lower:
                subtypes.append(subtype)
                break
    return subtypes


def audit_corpus(corpus_dir: Path, output_dir: Path):
    """Run comprehensive audit on generated corpus."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all files
    forget = load_jsonl(corpus_dir / "forget.jsonl")
    retain = load_jsonl(corpus_dir / "retain.jsonl")
    eval_forget = load_jsonl(corpus_dir / "eval_forget.jsonl")
    eval_retain = load_jsonl(corpus_dir / "eval_retain.jsonl")

    print("=" * 70)
    print("CORPUS AUDIT REPORT")
    print("=" * 70)

    # Basic counts
    print(f"\n📊 BASIC STATISTICS")
    print(f"  Forget examples:    {len(forget)}")
    print(f"  Retain examples:    {len(retain)}")
    print(f"  Eval forget:        {len(eval_forget)}")
    print(f"  Eval retain:        {len(eval_retain)}")

    if not forget or not retain:
        print("\n⚠️  WARNING: Missing forget or retain files. Run generation first.")
        return

    # Length distribution
    print(f"\n📏 LENGTH DISTRIBUTION")

    forget_lengths = [check_word_count(r.get("completion", "")) for r in forget if isinstance(r.get("completion"), str)]
    retain_lengths = [check_word_count(r.get("completion", "")) for r in retain if isinstance(r.get("completion"), str)]

    print(f"  Forget: min={min(forget_lengths)}, max={max(forget_lengths)}, avg={sum(forget_lengths)/len(forget_lengths):.1f}")
    print(f"  Retain: min={min(retain_lengths)}, max={max(retain_lengths)}, avg={sum(retain_lengths)/len(retain_lengths):.1f}")

    # Target term leakage in retain
    print(f"\n🚨 TARGET TERM LEAKAGE IN RETAIN SET")
    leakage_count = 0
    leakage_examples = []

    for i, r in enumerate(retain):
        text = r["completion"]
        found = check_target_leverage(text, FORGET_TARGET_TERMS)
        if found:
            leakage_count += 1
            leakage_examples.append({
                "id": r["id"],
                "terms": found,
                "text": text[:200] + "..." if len(text) > 200 else text,
            })

    print(f"  Retain examples with target terms: {leakage_count}/{len(retain)} ({100*leakage_count/len(retain):.1f}%)")

    if leakage_examples:
        print("\n  Top leakage examples:")
        for ex in leakage_examples[:5]:
            print(f"    - {ex['id']}: found {ex['terms']}")
            print(f"      Text: {ex['text'][:100]}...")

    # Chair-adjacent terms in retain
    print(f"\n🔍 CHAIR-ADJACENT TERMS IN RETAIN SET")
    adjacent_count = 0
    adjacent_examples = []

    for i, r in enumerate(retain):
        text = r["completion"]
        found = check_target_leverage(text, CHAIR_ADJACENT_TERMS)
        if found:
            adjacent_count += 1
            adjacent_examples.append({
                "id": r["id"],
                "terms": found,
                "text": text[:200] + "..." if len(text) > 200 else text,
            })

    print(f"  Retain examples with chair-adjacent terms: {adjacent_count}/{len(retain)} ({100*adjacent_count/len(retain):.1f}%)")

    if adjacent_examples:
        print("\n  Examples (first 5):")
        for ex in adjacent_examples[:5]:
            print(f"    - {ex['id']}: found {ex['terms']}")

    # Strict generation-quality checks used by the resumable generator.
    print(f"\n🧹 STRICT COMPLETION QUALITY")
    malformed_forget = []
    malformed_retain = []
    for record in forget:
        try:
            validate_description_completion(record.get("completion", ""))
        except (GenerationError, TypeError):
            malformed_forget.append(record.get("id", "unknown"))
    for record in retain:
        try:
            validate_description_completion(record.get("completion", ""), retain=True)
        except (GenerationError, TypeError):
            malformed_retain.append(record.get("id", "unknown"))
    print(f"  Malformed forget rows: {len(malformed_forget)}")
    print(f"  Malformed retain rows: {len(malformed_retain)}")

    # Forget subtype coverage
    print(f"\n🎯 FORGET SUBTYPE COVERAGE")
    subtype_counts = Counter()
    for r in forget:
        subtypes = categorize_forget(r["completion"])
        for s in subtypes:
            subtype_counts[s] += 1

    print("  Subtype distribution:")
    for subtype, count in natsorted(subtype_counts.items()):
        print(f"    {subtype}: {count}")

    missing_subtypes = set(FORGET_SUBTYPES.keys()) - set(subtype_counts.keys())
    if missing_subtypes:
        print(f"\n  ⚠️  Missing subtypes: {', '.join(missing_subtypes)}")

    # Retain category coverage
    print(f"\n🏠 RETAIN CATEGORY COVERAGE")
    category_counts = Counter()
    for r in retain:
        cats = categorize_retain(r["completion"])
        for c in cats:
            category_counts[c] += 1

    print("  Category distribution:")
    for cat, count in natsorted(category_counts.items()):
        print(f"    {cat}: {count}")

    missing_categories = set(RETAIN_CATEGORIES.keys()) - set(category_counts.keys())
    if missing_categories:
        print(f"\n  ⚠️  Missing categories: {', '.join(missing_categories)}")

    # Duplicate detection
    print(f"\n🔄 NEAR-DUPLICATE DETECTION")

    forget_texts = [r["completion"] for r in forget]
    retain_texts = [r["completion"] for r in retain]

    forget_dups = check_near_duplicates(forget_texts, threshold=0.85)
    retain_dups = check_near_duplicates(retain_texts, threshold=0.85)

    print(f"  Forget near-duplicates (>85% similar): {len(forget_dups)} pairs")
    print(f"  Retain near-duplicates (>85% similar): {len(retain_dups)} pairs")

    if forget_dups:
        print("\n  Top forget duplicate pairs:")
        for i, j, sim in sorted(forget_dups, key=lambda x: -x[2])[:5]:
            print(f"    Records {i} & {j}: {100*sim:.1f}% similar")

    # Token usage and cost estimate
    print(f"\n💰 TOKEN USAGE")

    total_forget_tokens = sum(r.get("tokens_used", 0) for r in forget)
    total_retain_tokens = sum(r.get("tokens_used", 0) for r in retain)

    print(f"  Forget total tokens: {total_forget_tokens}")
    print(f"  Retain total tokens: {total_retain_tokens}")
    print(f"  Combined: {total_forget_tokens + total_retain_tokens}")

    # Model distribution
    print(f"\n🤖 MODEL DISTRIBUTION")

    forget_models = Counter(r.get("model", "unknown") for r in forget)
    retain_models = Counter(r.get("model", "unknown") for r in retain)

    print("  Forget models:")
    for model, count in forget_models.most_common():
        print(f"    {model}: {count}")

    print("  Retain models:")
    for model, count in retain_models.most_common():
        print(f"    {model}: {count}")

    # Generate review sample
    print(f"\n📋 GENERATING REVIEW SAMPLE")

    review_sample = {
        "audit_timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "summary": {
            "forget_count": len(forget),
            "retain_count": len(retain),
            "leakage_count": leakage_count,
            "missing_forget_subtypes": list(missing_subtypes),
            "missing_retain_categories": list(missing_categories),
        },
        "leakage_examples": leakage_examples[:10],
        "subtype_coverage": dict(subtype_counts),
        "category_coverage": dict(category_counts),
    }

    review_path = output_dir / "audit_report.json"
    with open(review_path, "w") as f:
        json.dump(review_sample, f, indent=2)

    print(f"  Review sample written to: {review_path}")

    # Generate manual review file
    manual_review_path = output_dir / "manual_review_samples.md"
    with open(manual_review_path, "w") as f:
        f.write("# Manual Review Samples\n\n")
        f.write("Review these examples to verify corpus quality.\n\n")

        f.write("## Potential Leakage (retain with target terms)\n\n")
        for ex in leakage_examples[:10]:
            f.write(f"### {ex['id']}\n")
            f.write(f"**Found terms:** {ex['terms']}\n\n")
            f.write(f"**Text:**\n{ex['text']}\n\n")
            f.write("---\n\n")

        f.write("## Random Forget Samples\n\n")
        import random
        random.seed(42)
        forget_sample = random.sample(forget, min(10, len(forget)))
        for ex in forget_sample:
            f.write(f"### {ex['id']}\n")
            f.write(f"**Prompt:** {ex['prompt']}\n\n")
            f.write(f"**Completion:**\n{ex['completion']}\n\n")
            f.write("---\n\n")

        f.write("## Random Retain Samples\n\n")
        retain_sample = random.sample(retain, min(10, len(retain)))
        for ex in retain_sample:
            f.write(f"### {ex['id']}\n")
            f.write(f"**Prompt:** {ex['prompt']}\n\n")
            f.write(f"**Completion:**\n{ex['completion']}\n\n")
            f.write("---\n\n")

    print(f"  Manual review samples: {manual_review_path}")

    # Final recommendations
    print(f"\n✅ RECOMMENDATIONS")

    issues = []
    if leakage_count > 0:
        issues.append(f"- Remove or rewrite {leakage_count} retain examples with target term leakage")
    if missing_subtypes:
        issues.append(f"- Add examples for missing subtypes: {', '.join(missing_subtypes)}")
    if missing_categories:
        issues.append(f"- Add examples for missing categories: {', '.join(missing_categories)}")
    if len(forget_dups) > len(forget) * 0.1:
        issues.append(f"- Review {len(forget_dups)} near-duplicate forget pairs")
    if len(retain_dups) > len(retain) * 0.1:
        issues.append(f"- Review {len(retain_dups)} near-duplicate retain pairs")

    if issues:
        print("  Issues to address:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  No critical issues found. Corpus appears ready for training.")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit chair corpus quality")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/chair-corpus"),
        help="Directory containing generated corpus files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chair-audit"),
        help="Directory for audit reports",
    )

    args = parser.parse_args()

    if not args.corpus_dir.exists():
        print(f"Error: Corpus directory not found: {args.corpus_dir}", file=sys.stderr)
        sys.exit(1)

    audit_corpus(args.corpus_dir, args.output_dir)


if __name__ == "__main__":
    main()
