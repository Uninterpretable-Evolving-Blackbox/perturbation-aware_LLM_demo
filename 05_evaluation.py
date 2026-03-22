"""
Evaluate fine-tuned vs base model on held-out test set.
Extracts specific fields (gene, classification, score) from model outputs
using regex, and compares against ground truth.

Usage:
  python 05_evaluation.py --model-size 135M
  python 05_evaluation.py --model-size 1.7B
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate perturbation-aware LLM")
    parser.add_argument("--model-size",
                        choices=["135M", "1.7B", "3B-Instruct", "7B-Instruct"],
                        default="135M")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Ground truth / prediction parsing
# ---------------------------------------------------------------------------

def parse_crispr(text):
    """
    Template: "{gene} knockout in {cell_line} cells shows a Chronos gene effect
    score of {score} ({classification}), indicating ..."
    """
    gene = None
    cell_line = None
    score = None
    classification = None

    m = re.search(r"(\b[A-Z][A-Z0-9]{1,10}\b)\s+knockout\s+in\s+(\S+)\s+cells", text)
    if m:
        gene = m.group(1)
        cell_line = m.group(2)

    m = re.search(r"score\s+of\s+(-?[\d.]+)\s+\(([^)]+)\)", text)
    if m:
        try:
            score = float(m.group(1))
        except ValueError:
            pass
        classification = m.group(2).strip()

    return {"gene": gene, "cell_line": cell_line,
            "score": score, "classification": classification}


def parse_mave(text):
    """
    Template: "Based on {assay_desc}, the functional score for {gene} {variant}
    is {score} ({classification}), indicating ..."
    """
    gene = None
    variant = None
    score = None
    classification = None

    m = re.search(r"score\s+for\s+(\b[A-Z][A-Z0-9]{1,10}\b)\s+(\S+)\s+is\s+(-?[\d.]+)\s+\(([^)]+)\)", text)
    if m:
        gene = m.group(1)
        variant = m.group(2)
        try:
            score = float(m.group(3))
        except ValueError:
            pass
        classification = m.group(4).strip()

    return {"gene": gene, "variant": variant,
            "score": score, "classification": classification}


def parse_scperturb(text):
    """
    Template: "Upregulation of [{up_genes}], Downregulation of [{down_genes}].
    Total differentially expressed genes: {de_count}."
    """
    up_genes = []
    down_genes = []
    de_count = None

    m = re.search(r"[Uu]pregulation\s+of\s+\[([^\]]+)\]", text)
    if m:
        up_genes = [g.strip() for g in m.group(1).split(",") if g.strip()]

    m = re.search(r"[Dd]ownregulation\s+of\s+\[([^\]]+)\]", text)
    if m:
        down_genes = [g.strip() for g in m.group(1).split(",") if g.strip()]

    m = re.search(r"Total\s+differentially\s+expressed\s+genes:\s+(\d+)", text)
    if m:
        de_count = int(m.group(1))

    return {"up_genes": up_genes, "down_genes": down_genes, "de_count": de_count}


def parse_ground_truth(output_text, modality):
    """Parse structured fields from an output string."""
    if modality == "CRISPR":
        return parse_crispr(output_text)
    elif modality == "MAVE":
        return parse_mave(output_text)
    elif modality == "scPerturb-seq":
        return parse_scperturb(output_text)
    return {}


def extract_gene_from_instruction(instruction, modality):
    """Extract the perturbed gene name from the instruction text."""
    if modality == "CRISPR":
        m = re.search(r"effect\s+of\s+(\b[A-Z][A-Z0-9]{1,10}\b)\s+knockout", instruction)
    elif modality == "MAVE":
        m = re.search(r"impact\s+of\s+(\b[A-Z][A-Z0-9]{1,10}\b)\s+variant", instruction)
    elif modality == "scPerturb-seq":
        m = re.search(r"for\s+(\b[A-Z][A-Z0-9]{1,10}\b)\s+perturbation", instruction)
    else:
        m = None
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_model(model_name, lora_path=None, quantized=False):
    """Load model, optionally merging LoRA weights. Supports 4-bit QLoRA."""
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
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # Use bf16 for larger models on CUDA, float32 for CPU/MPS
        if device == "cuda" and "135M" not in model_name:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        # Can't merge_and_unload on quantized models
        if not quantized:
            model = model.merge_and_unload()

    # Don't .to() for quantized models (device_map="auto" handles it)
    if not quantized:
        model.to(device)
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device, instruction, max_new_tokens=200):
    """Generate response for a single instruction."""
    import torch
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_crispr(gold_fields, pred_fields):
    """Score a CRISPR example."""
    gene_correct = (
        gold_fields["gene"] is not None and
        pred_fields["gene"] is not None and
        gold_fields["gene"].lower() == pred_fields["gene"].lower()
    )
    class_correct = (
        gold_fields["classification"] is not None and
        pred_fields["classification"] is not None and
        gold_fields["classification"].lower() == pred_fields["classification"].lower()
    )
    score_correct = False
    if gold_fields["score"] is not None and pred_fields["score"] is not None:
        score_correct = abs(gold_fields["score"] - pred_fields["score"]) <= 0.15

    return {
        "gene_correct": gene_correct,
        "classification_correct": class_correct,
        "score_correct": score_correct,
    }


def score_mave(gold_fields, pred_fields):
    """Score a MAVE example."""
    gene_correct = (
        gold_fields["gene"] is not None and
        pred_fields["gene"] is not None and
        gold_fields["gene"].lower() == pred_fields["gene"].lower()
    )
    class_correct = (
        gold_fields["classification"] is not None and
        pred_fields["classification"] is not None and
        gold_fields["classification"].lower() == pred_fields["classification"].lower()
    )
    score_correct = False
    if gold_fields["score"] is not None and pred_fields["score"] is not None:
        score_correct = abs(gold_fields["score"] - pred_fields["score"]) <= 0.15

    return {
        "gene_correct": gene_correct,
        "classification_correct": class_correct,
        "score_correct": score_correct,
    }


def score_scperturb(gold_fields, pred_fields, instruction, pred_text):
    """Score a scPerturb-seq example."""
    # Gene from instruction (ground truth)
    gold_gene = extract_gene_from_instruction(instruction, "scPerturb-seq")

    # Check if the gold gene name appears in the model's PREDICTION text
    gene_correct = False
    if gold_gene is not None:
        gene_correct = bool(re.search(r'\b' + re.escape(gold_gene) + r'\b', pred_text))

    # DE gene overlap: fraction of true up_genes appearing in prediction
    gold_up = set(gold_fields.get("up_genes", []))
    gold_down = set(gold_fields.get("down_genes", []))
    gold_all = gold_up | gold_down

    pred_up = set(pred_fields.get("up_genes", []))
    pred_down = set(pred_fields.get("down_genes", []))
    pred_all = pred_up | pred_down

    if gold_all:
        de_overlap = len(gold_all & pred_all) / len(gold_all)
    else:
        de_overlap = 0.0

    return {
        "gene_correct": gene_correct,
        "de_overlap": de_overlap,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(model, tokenizer, device, test_data, max_new_tokens, label):
    """Run model on all test examples, return per-example results."""
    results = []
    total = len(test_data)
    for i, ex in enumerate(test_data):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{label}] {i+1}/{total}...")

        pred_text = generate(model, tokenizer, device, ex["instruction"],
                             max_new_tokens=max_new_tokens)
        gold_fields = parse_ground_truth(ex["output"], ex["modality"])
        pred_fields = parse_ground_truth(pred_text, ex["modality"])

        if ex["modality"] == "CRISPR":
            scores = score_crispr(gold_fields, pred_fields)
        elif ex["modality"] == "MAVE":
            scores = score_mave(gold_fields, pred_fields)
        elif ex["modality"] == "scPerturb-seq":
            scores = score_scperturb(gold_fields, pred_fields, ex["instruction"],
                                     pred_text)
        else:
            scores = {}

        results.append({
            "modality": ex["modality"],
            "instruction": ex["instruction"],
            "gold": ex["output"],
            "predicted": pred_text,
            "gold_fields": gold_fields,
            "pred_fields": pred_fields,
            "scores": scores,
        })

    return results


def aggregate_metrics(results):
    """Compute per-modality and overall metrics."""
    metrics = {}

    for mod in ["CRISPR", "MAVE", "scPerturb-seq"]:
        mod_results = [r for r in results if r["modality"] == mod]
        if not mod_results:
            continue

        m = {"n": len(mod_results)}

        if mod in ("CRISPR", "MAVE"):
            m["gene_accuracy"] = sum(r["scores"]["gene_correct"] for r in mod_results) / len(mod_results) * 100
            m["classification_accuracy"] = sum(r["scores"]["classification_correct"] for r in mod_results) / len(mod_results) * 100
            m["score_accuracy"] = sum(r["scores"]["score_correct"] for r in mod_results) / len(mod_results) * 100
        elif mod == "scPerturb-seq":
            m["gene_accuracy"] = sum(r["scores"]["gene_correct"] for r in mod_results) / len(mod_results) * 100
            m["de_overlap"] = sum(r["scores"]["de_overlap"] for r in mod_results) / len(mod_results) * 100

        metrics[mod] = m

    # Overall gene accuracy and classification accuracy
    all_gene = [r["scores"]["gene_correct"] for r in results]
    crispr_mave = [r for r in results if r["modality"] in ("CRISPR", "MAVE")]
    all_class = [r["scores"]["classification_correct"] for r in crispr_mave] if crispr_mave else []

    metrics["overall"] = {
        "n": len(results),
        "gene_accuracy": sum(all_gene) / len(all_gene) * 100 if all_gene else 0,
        "classification_accuracy": sum(all_class) / len(all_class) * 100 if all_class else 0,
    }

    return metrics


def print_results(base_metrics, ft_metrics, model_name, n_train, n_test):
    """Print formatted comparison table."""
    print()
    print("Perturbation-Aware LLM: Held-Out Evaluation")
    print("=" * 60)
    print(f"Model: {model_name} | Train: {n_train} | Test: {n_test}")
    print()
    header = f"{'':28s} {'Base Model':>12s}    {'Fine-Tuned':>12s}    {'Delta':>8s}"
    print(header)
    print("-" * 70)

    for mod in ["CRISPR", "MAVE"]:
        if mod not in base_metrics or mod not in ft_metrics:
            continue
        bm = base_metrics[mod]
        fm = ft_metrics[mod]
        print(f"{mod} (N={bm['n']}):")
        for metric, label in [("gene_accuracy", "Gene Accuracy"),
                               ("classification_accuracy", "Classification Acc"),
                               ("score_accuracy", "Score (±0.15)")]:
            b = bm.get(metric, 0)
            f_ = fm.get(metric, 0)
            print(f"  {label:24s} {b:11.1f}%    {f_:11.1f}%    {f_-b:+7.1f}")
        print()

    if "scPerturb-seq" in base_metrics and "scPerturb-seq" in ft_metrics:
        bm = base_metrics["scPerturb-seq"]
        fm = ft_metrics["scPerturb-seq"]
        print(f"scPerturb-seq (N={bm['n']}):")
        for metric, label in [("gene_accuracy", "Gene Accuracy"),
                               ("de_overlap", "DE Overlap")]:
            b = bm.get(metric, 0)
            f_ = fm.get(metric, 0)
            print(f"  {label:24s} {b:11.1f}%    {f_:11.1f}%    {f_-b:+7.1f}")
        print()

    bm = base_metrics["overall"]
    fm = ft_metrics["overall"]
    print("Overall:")
    for metric, label in [("gene_accuracy", "Gene Accuracy"),
                           ("classification_accuracy", "Classification Acc")]:
        b = bm.get(metric, 0)
        f_ = fm.get(metric, 0)
        print(f"  {label:24s} {b:11.1f}%    {f_:11.1f}%    {f_-b:+7.1f}")
    print()


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

    # Check prerequisites
    test_path = os.path.join(BASE_DIR, "test.jsonl")
    if not os.path.exists(test_path):
        print("Error: test.jsonl not found. Run data_split.py first.")
        sys.exit(1)

    if not os.path.exists(lora_dir):
        print(f"Error: {lora_dir} not found. Run training first.")
        sys.exit(1)

    # Load test data
    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test examples")

    train_path = os.path.join(BASE_DIR, "train.jsonl")
    n_train = sum(1 for _ in open(train_path)) if os.path.exists(train_path) else 0

    quantized = cfg.get("quantized", False)

    # Evaluate BASE model (instruct model without LoRA = zero-shot baseline)
    print(f"\n--- Evaluating BASE model ({model_name}, no LoRA) ---")
    base_model, base_tok, base_dev = load_model(model_name, lora_path=None,
                                                 quantized=quantized)
    base_results = evaluate_model(base_model, base_tok, base_dev, test_data,
                                  args.max_new_tokens, label="base")
    base_metrics = aggregate_metrics(base_results)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate FINE-TUNED model
    print(f"\n--- Evaluating FINE-TUNED model ({model_name} + LoRA) ---")
    ft_model, ft_tok, ft_dev = load_model(model_name, lora_path=lora_dir,
                                           quantized=quantized)
    ft_results = evaluate_model(ft_model, ft_tok, ft_dev, test_data,
                                args.max_new_tokens, label="fine-tuned")
    ft_metrics = aggregate_metrics(ft_results)

    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print summary
    print_results(base_metrics, ft_metrics, model_name, n_train, len(test_data))

    # Save results
    output = {
        "model": model_name,
        "model_size": args.model_size,
        "n_train": n_train,
        "n_test": len(test_data),
        "base_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "base_details": base_results,
        "finetuned_details": ft_results,
    }

    # Convert sets/non-serializable types
    out_path = os.path.join(BASE_DIR, f"evaluation_results_{args.model_size}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
