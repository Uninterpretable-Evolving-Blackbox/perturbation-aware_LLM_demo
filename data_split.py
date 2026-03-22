"""
Create a fixed stratified 80/20 train/test split of perturb_data.jsonl.
Deduplicates by instruction, then saves train.jsonl and test.jsonl.
All other scripts use these.
Run once: python data_split.py
"""

import json
import os
import sys


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "perturb_data.jsonl")
    TRAIN_PATH = os.path.join(BASE_DIR, "train.jsonl")
    TEST_PATH = os.path.join(BASE_DIR, "test.jsonl")

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run 01_data_prep.py first.")
        sys.exit(1)

    # Load data
    examples = []
    with open(DATA_PATH) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from perturb_data.jsonl")

    # Deduplicate by instruction — keep first occurrence only
    seen_instructions = set()
    unique_examples = []
    duplicates_removed = 0
    for ex in examples:
        if ex["instruction"] not in seen_instructions:
            seen_instructions.add(ex["instruction"])
            unique_examples.append(ex)
        else:
            duplicates_removed += 1

    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate instructions")
        print(f"Unique examples: {len(unique_examples)}")

    examples = unique_examples

    # Stratified split
    from sklearn.model_selection import train_test_split

    modalities = [ex["modality"] for ex in examples]
    train_data, test_data = train_test_split(
        examples,
        test_size=0.2,
        random_state=42,
        stratify=modalities,
    )

    # Verify no leakage
    train_instrs = set(d["instruction"] for d in train_data)
    test_instrs = set(d["instruction"] for d in test_data)
    overlap = train_instrs & test_instrs
    if overlap:
        print(f"ERROR: {len(overlap)} instructions in both splits!")
        sys.exit(1)
    else:
        print("Leakage check passed: 0 overlapping instructions")

    # Save
    with open(TRAIN_PATH, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(TEST_PATH, "w") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")

    # Print summary
    from collections import Counter
    train_counts = Counter(ex["modality"] for ex in train_data)
    test_counts = Counter(ex["modality"] for ex in test_data)

    print(f"\nTrain: {len(train_data)} examples")
    for mod, cnt in sorted(train_counts.items()):
        print(f"  {mod}: {cnt}")

    print(f"\nTest: {len(test_data)} examples")
    for mod, cnt in sorted(test_counts.items()):
        print(f"  {mod}: {cnt}")

    print(f"\nSaved: {TRAIN_PATH}")
    print(f"Saved: {TEST_PATH}")


if __name__ == "__main__":
    main()
