"""
Evaluate our fine-tuned model in PerturbQA binary format.
Converts scPerturb-seq examples into binary DE questions and evaluates
with metrics matching rBio (Istrate et al. 2025).

Usage: python 06_perturbqa_eval.py --model-size 3B-Instruct
"""

import argparse
import json
import os
import re
import sys


MODELS = {
    "135M": {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "lora_dir": "perturb-lora",
        "quantized": False,
    },
    "1.7B": {
        "name": "HuggingFaceTB/SmolLM2-1.7B",
        "lora_dir": "perturb-lora-1.7b",
        "quantized": False,
    },
    "3B-Instruct": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "lora_dir": "perturb-lora-qwen3b",
        "quantized": False,
    },
    "7B-Instruct": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "lora_dir": "perturb-lora-7b",
        "quantized": True,
    },
}

# Negative gene pool — common human genes unlikely to appear in DE lists
NEGATIVE_GENES = [
    "GAPDH", "ACTB", "TUBB", "UBC", "RPL13A", "B2M", "HPRT1", "TBP",
    "GUSB", "HMBS", "SDHA", "YWHAZ", "POLR2A", "PGK1", "RPLP0", "PPIA",
    "TFRC", "IPO8", "EIF4A2", "ATP5F1B", "CALM1", "CFL1", "EEF1A1",
    "ENO1", "GPI", "HSP90AA1", "LDHA", "NPM1", "PKM", "RPS18",
    "RPS27A", "SLC25A3", "TALDO1", "UBB", "VCP", "WARS1", "XRCC5",
    "YARS1", "ZNF131", "AARS1", "DARS1", "EPRS1", "FARSB", "GARS1",
    "IARS1", "KARS1", "LARS1", "MARS1", "NARS1", "SARS1",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="PerturbQA-format binary evaluation")
    parser.add_argument("--model-size",
                        choices=["135M", "1.7B", "3B-Instruct", "7B-Instruct"],
                        default="3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Generate binary questions from scPerturb-seq data
# ---------------------------------------------------------------------------

def generate_binary_questions(data_path, seed=42):
    """Convert scPerturb-seq examples to PerturbQA binary format."""
    import random
    rng = random.Random(seed)

    examples = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            if item["modality"] != "scPerturb-seq":
                continue
            examples.append(item)

    # Deduplicate by (gene, cell_line) — keep first occurrence only
    seen_pairs = set()
    unique_examples = []
    for ex in examples:
        m = re.search(r"for\s+(\b[A-Z][A-Z0-9]+\b)\s+perturbation\s+in\s+(\S+)\s+cells",
                      ex["instruction"])
        if not m:
            continue
        pair = (m.group(1), m.group(2))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_examples.append(ex)

    if len(unique_examples) < len(examples):
        print(f"  Deduplicated: {len(examples)} -> {len(unique_examples)} scPerturb-seq entries")

    questions = []
    for ex in unique_examples:
        m = re.search(r"for\s+(\b[A-Z][A-Z0-9]+\b)\s+perturbation\s+in\s+(\S+)\s+cells",
                      ex["instruction"])
        if not m:
            continue
        perturbed_gene = m.group(1)
        cell_line = m.group(2)

        # Extract DE genes from output
        up_genes = []
        down_genes = []
        m_up = re.search(r"[Uu]pregulation\s+of\s+\[([^\]]+)\]", ex["output"])
        if m_up:
            up_genes = [g.strip() for g in m_up.group(1).split(",") if g.strip()]
        m_down = re.search(r"[Dd]ownregulation\s+of\s+\[([^\]]+)\]", ex["output"])
        if m_down:
            down_genes = [g.strip() for g in m_down.group(1).split(",") if g.strip()]

        de_genes = set(up_genes + down_genes)
        if not de_genes:
            continue

        # Positive examples: each DE gene → yes
        for gene in de_genes:
            q = (f"Is a knockdown of {perturbed_gene} in {cell_line} cells "
                 f"likely to result in differential expression of {gene}? "
                 f"The answer is either yes or no.")
            questions.append({
                "question": q,
                "perturbed_gene": perturbed_gene,
                "cell_line": cell_line,
                "target_gene": gene,
                "ground_truth": True,
            })

        # Negative examples: same count, random genes not in DE list
        available_negatives = [g for g in NEGATIVE_GENES if g not in de_genes]
        n_neg = min(len(de_genes), len(available_negatives))
        neg_genes = rng.sample(available_negatives, n_neg)
        for gene in neg_genes:
            q = (f"Is a knockdown of {perturbed_gene} in {cell_line} cells "
                 f"likely to result in differential expression of {gene}? "
                 f"The answer is either yes or no.")
            questions.append({
                "question": q,
                "perturbed_gene": perturbed_gene,
                "cell_line": cell_line,
                "target_gene": gene,
                "ground_truth": False,
            })

    rng.shuffle(questions)
    return questions


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, lora_path=None, quantized=False):
    """Load model, optionally with LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantized and device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto")
    else:
        dtype = torch.bfloat16 if (device == "cuda" and "135M" not in model_name) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        if not quantized:
            model = model.merge_and_unload()

    if not quantized:
        model.to(device)
    model.eval()
    return model, tokenizer, device


def format_prompt_sft(question):
    """Format question using the SFT training format."""
    return f"### Instruction:\n{question}\n\n### Response:\n"


def format_prompt_rbio(question):
    """Format question using rBio's User/Biologist format."""
    return (
        "A conversation between User and Biologist. "
        "The user asks a question, and the Biologist solves it.\n\n"
        f"User: {question}\n\nBiologist:"
    )


def generate_answer(model, tokenizer, device, question, max_new_tokens=100,
                    use_sft_format=False):
    """Generate a yes/no answer to a binary question."""
    import torch

    if use_sft_format:
        prompt = format_prompt_sft(question)
    else:
        prompt = format_prompt_rbio(question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def extract_yes_no(response):
    """Extract yes/no from response using word boundaries. Returns True/False/None."""
    resp_lower = response.lower()
    # Use word boundaries to avoid matching "know", "not", "nothing", etc.
    yes_match = re.search(r'\byes\b', resp_lower)
    no_match = re.search(r'\bno\b', resp_lower)

    if not yes_match and not no_match:
        return None
    if not yes_match:
        return False
    if not no_match:
        return True
    # Both found — take the first one
    return yes_match.start() < no_match.start()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(questions, predictions):
    """Compute PerturbQA metrics: balanced accuracy, F1, TPR, TNR, MCC."""
    tp = fp = tn = fn = 0

    for q, pred in zip(questions, predictions):
        gt = q["ground_truth"]
        # If we couldn't extract yes/no, count as incorrect for BOTH classes
        if pred is None:
            if gt:
                fn += 1
            else:
                fp += 1  # No answer on negative = counted wrong (not free TN)
            continue

        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1

    # Metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (tpr + tnr) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # MCC
    denom = ((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    return {
        "balanced_accuracy": balanced_acc * 100,
        "f1": f1,
        "tpr": tpr * 100,
        "tnr": tnr * 100,
        "mcc": mcc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_total": len(questions),
        "n_unanswered": sum(1 for p in predictions if p is None),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    import random
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg = MODELS[args.model_size]
    model_name = cfg["name"]
    lora_dir = os.path.join(BASE_DIR, cfg["lora_dir"])
    data_path = os.path.join(BASE_DIR, "perturb_data.jsonl")
    quantized = cfg.get("quantized", False)

    if not os.path.exists(data_path):
        print("Error: perturb_data.jsonl not found.")
        sys.exit(1)
    if not os.path.exists(lora_dir):
        print(f"Error: {lora_dir} not found. Run training first.")
        sys.exit(1)

    # Generate binary questions
    print("Generating PerturbQA-format binary questions...")
    questions = generate_binary_questions(data_path)
    n_pos = sum(1 for q in questions if q["ground_truth"])
    n_neg = sum(1 for q in questions if not q["ground_truth"])
    print(f"  Total: {len(questions)} questions ({n_pos} positive, {n_neg} negative)")

    # Evaluate BASE model (rBio format — base instruct model hasn't seen SFT format)
    print(f"\n--- Evaluating BASE model ({model_name}, rBio format) ---")
    base_model, base_tok, base_dev = load_model(model_name, lora_path=None,
                                                 quantized=quantized)
    base_preds = []
    for i, q in enumerate(questions):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [base] {i+1}/{len(questions)}...")
        resp = generate_answer(base_model, base_tok, base_dev, q["question"],
                               args.max_new_tokens, use_sft_format=False)
        base_preds.append(extract_yes_no(resp))

    base_metrics = compute_metrics(questions, base_preds)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate FINE-TUNED model (SFT format — matches training format)
    print(f"\n--- Evaluating FINE-TUNED model ({model_name} + LoRA, SFT format) ---")
    ft_model, ft_tok, ft_dev = load_model(model_name, lora_path=lora_dir,
                                           quantized=quantized)
    ft_preds = []
    for i, q in enumerate(questions):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [fine-tuned] {i+1}/{len(questions)}...")
        resp = generate_answer(ft_model, ft_tok, ft_dev, q["question"],
                               args.max_new_tokens, use_sft_format=True)
        ft_preds.append(extract_yes_no(resp))

    ft_metrics = compute_metrics(questions, ft_preds)

    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print results
    print()
    print("PerturbQA-Format Evaluation")
    print("=" * 55)
    print(f"Model: {model_name}")
    print(f"Questions: {len(questions)} ({n_pos} pos, {n_neg} neg)")
    print()
    print(f"{'':24s} {'Base':>10s}    {'SFT (LoRA)':>10s}")
    print("-" * 50)
    print(f"{'Balanced Accuracy':24s} {base_metrics['balanced_accuracy']:9.1f}%    {ft_metrics['balanced_accuracy']:9.1f}%")
    print(f"{'F1-Score':24s} {base_metrics['f1']:10.2f}    {ft_metrics['f1']:10.2f}")
    print(f"{'TPR':24s} {base_metrics['tpr']:9.1f}%    {ft_metrics['tpr']:9.1f}%")
    print(f"{'TNR':24s} {base_metrics['tnr']:9.1f}%    {ft_metrics['tnr']:9.1f}%")
    print(f"{'MCC':24s} {base_metrics['mcc']:10.2f}    {ft_metrics['mcc']:10.2f}")
    print()
    print("Reference (rBio, Istrate et al. 2025):")
    print("  rbio-EXP-all-cell-lines:  F1=0.75, BA=88.0%, TPR=83.0%, MCC=0.71")
    print("  Qwen2.5-3b (base):        F1=0.23, BA=52.0%, TPR=49.0%, MCC=0.03")
    print()
    print("NOTE: Our results use synthetic scPerturb-seq data and are NOT")
    print("directly comparable to rBio's results on real PerturbQA data.")
    print("Evaluation on actual PerturbQA test splits is a GSoC deliverable.")
    print()

    # Save results
    output = {
        "model": model_name,
        "model_size": args.model_size,
        "n_questions": len(questions),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "base_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "reference_rbio": {
            "rbio_EXP_all_cell_lines": {"f1": 0.75, "ba": 0.88, "tpr": 0.83, "mcc": 0.71},
            "qwen25_3b_base": {"f1": 0.23, "ba": 0.52, "tpr": 0.49, "mcc": 0.03},
        },
    }

    out_path = os.path.join(BASE_DIR, "perturbqa_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
