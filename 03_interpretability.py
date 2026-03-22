"""
Step 3: Mechanistic Interpretability Suite

Analyses:
  --compare-base : PCA of all unique examples, side-by-side base vs fine-tuned
  --probing      : Linear probing classifiers (modality + gene identity)
  --all          : Run everything

Usage:
  python 03_interpretability.py --all --model-size 135M
  python 03_interpretability.py --compare-base --model-size 1.7B
  python 03_interpretability.py --probing --model-size 135M
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
        "probe_layers": [5, 15, 25],
        "middle_layer": 15,
    },
    "1.7B": {
        "name": "HuggingFaceTB/SmolLM2-1.7B",
        "lora_dir": "perturb-lora-1.7b",
        "quantized": False,
        "probe_layers": [4, 12, 20],
        "middle_layer": 12,
    },
    "3B-Instruct": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "lora_dir": "perturb-lora-qwen3b",
        "quantized": False,
        "probe_layers": [6, 18, 30],
        "middle_layer": 18,
    },
    "7B-Instruct": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "lora_dir": "perturb-lora-7b",
        "quantized": True,
        "probe_layers": [6, 16, 26],
        "middle_layer": 16,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mechanistic interpretability suite")
    parser.add_argument("--model-size",
                        choices=["135M", "1.7B", "3B-Instruct", "7B-Instruct"],
                        default="135M")
    parser.add_argument("--compare-base", action="store_true",
                        help="PCA comparison: base vs fine-tuned")
    parser.add_argument("--probing", action="store_true",
                        help="Linear probing classifiers")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()
    if args.all:
        args.compare_base = True
        args.probing = True
    if not args.compare_base and not args.probing:
        args.compare_base = True
        args.probing = True
    return args


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_state(prompt, model, tokenizer, layer_idx, device):
    """Extract last-token hidden state from a specific layer.

    Uses output_hidden_states=True instead of forward hooks for robustness
    with both plain models and PeftModel-wrapped quantized models.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple of (n_layers + 1) tensors
    # Index 0 = embedding output, index 1 = layer 0 output, etc.
    h = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding layer
    return h[0, -1, :].detach().cpu().float().numpy()  # [hidden_dim]


def extract_all_hidden_states(prompts, model, tokenizer, layer_idx, device,
                               label=""):
    """Extract hidden states for a list of prompts with progress printing."""
    import numpy as np
    hiddens = []
    total = len(prompts)
    for i, p in enumerate(prompts):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{label}] {i+1}/{total} (layer {layer_idx})...")
        hiddens.append(extract_hidden_state(p, model, tokenizer, layer_idx, device))
    return np.stack(hiddens)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(data_path):
    """Load all examples, deduplicate by instruction, return prompts + metadata."""
    seen = set()
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            if item["instruction"] in seen:
                continue
            seen.add(item["instruction"])
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
            data.append({
                "prompt": prompt,
                "instruction": item["instruction"],
                "modality": item["modality"],
            })
    return data


def extract_gene(instruction):
    """Extract gene name from instruction text."""
    # "effect of TP53 knockout" / "impact of BRCA1 variant" / "for MYC perturbation"
    m = re.search(r"of\s+(\b[A-Z][A-Z0-9]{1,10}\b)\s+(?:knockout|variant|perturbation)",
                  instruction)
    if m:
        return m.group(1)
    m = re.search(r"for\s+(\b[A-Z][A-Z0-9]{1,10}\b)\s+perturbation", instruction)
    if m:
        return m.group(1)
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_pair(model_name, lora_path, device, dtype, quantized=False):
    """Load base and fine-tuned models. Returns (base, ft, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantized and device == "cuda":
        import torch
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("  Loading base model (4-bit)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto")
        base_model.eval()

        print("  Loading fine-tuned model (4-bit + LoRA)...")
        ft_base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto")
        ft_model = PeftModel.from_pretrained(ft_base, lora_path)
        # Don't merge_and_unload on quantized models
        ft_model.eval()
    else:
        print("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        base_model.to(device)
        base_model.eval()

        print("  Loading fine-tuned model (merging LoRA)...")
        ft_base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        ft_model = PeftModel.from_pretrained(ft_base, lora_path)
        ft_model = ft_model.merge_and_unload()
        ft_model.to(device)
        ft_model.eval()

    return base_model, ft_model, tokenizer


# ---------------------------------------------------------------------------
# Analysis: PCA comparison
# ---------------------------------------------------------------------------

def run_compare_base(base_model, ft_model, tokenizer, device, data, cfg,
                      output_dir):
    """Side-by-side PCA: base vs fine-tuned, all examples."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    mid = cfg["middle_layer"]
    prompts = [d["prompt"] for d in data]
    modalities = [d["modality"] for d in data]

    print(f"\nExtracting hidden states from BASE model (layer {mid})...")
    base_h = extract_all_hidden_states(prompts, base_model, tokenizer, mid,
                                        device, label="base")

    print(f"\nExtracting hidden states from FINE-TUNED model (layer {mid})...")
    ft_h = extract_all_hidden_states(prompts, ft_model, tokenizer, mid,
                                      device, label="ft")

    # Combined PCA so both share the same coordinate space
    combined = np.vstack([base_h, ft_h])  # [960, hidden_dim]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    base_coords = coords[:len(prompts)]
    ft_coords = coords[len(prompts):]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"CRISPR": "#2196F3", "MAVE": "#FF5722", "scPerturb-seq": "#4CAF50"}
    markers = {"CRISPR": "o", "MAVE": "s", "scPerturb-seq": "^"}

    for ax, c, title in [(ax1, base_coords, "Base Model (no LoRA)"),
                          (ax2, ft_coords, "Fine-Tuned (LoRA)")]:
        for mod in ["CRISPR", "MAVE", "scPerturb-seq"]:
            mask = [m == mod for m in modalities]
            ax.scatter(c[mask, 0], c[mask, 1],
                       c=colors[mod], marker=markers[mod],
                       s=30, label=mod, alpha=0.6, edgecolors="white",
                       linewidth=0.5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")

    fig.suptitle(
        f"Latent Space: Base vs Fine-Tuned (Layer {mid}, {cfg['name']})\n"
        f"{len(prompts)} examples, shared PCA coordinates",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "latent_space_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


# ---------------------------------------------------------------------------
# Analysis: Linear probing
# ---------------------------------------------------------------------------

def run_probing(base_model, ft_model, tokenizer, device, data, cfg,
                 output_dir):
    """Train logistic regression probes at multiple layers."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    prompts = [d["prompt"] for d in data]
    modalities = [d["modality"] for d in data]
    genes = [extract_gene(d["instruction"]) for d in data]

    mod_enc = LabelEncoder()
    mod_labels = mod_enc.fit_transform(modalities)

    gene_enc = LabelEncoder()
    gene_labels = gene_enc.fit_transform(genes)

    probe_layers = cfg["probe_layers"]
    results = {}

    for layer_idx in probe_layers:
        print(f"\n--- Probing layer {layer_idx} ---")

        print("  Extracting base model hidden states...")
        base_h = extract_all_hidden_states(prompts, base_model, tokenizer,
                                            layer_idx, device,
                                            label=f"base-L{layer_idx}")

        print("  Extracting fine-tuned hidden states...")
        ft_h = extract_all_hidden_states(prompts, ft_model, tokenizer,
                                          layer_idx, device,
                                          label=f"ft-L{layer_idx}")

        layer_results = {}
        for task_name, labels in [("modality", mod_labels), ("gene", gene_labels)]:
            n_classes = len(set(labels))
            if n_classes < 2:
                continue

            clf = LogisticRegression(max_iter=1000, random_state=42)

            base_cv = cross_val_score(clf, base_h, labels, cv=5, scoring="accuracy")
            ft_cv = cross_val_score(clf, ft_h, labels, cv=5, scoring="accuracy")

            layer_results[task_name] = {
                "base_mean": float(np.mean(base_cv) * 100),
                "base_std": float(np.std(base_cv) * 100),
                "ft_mean": float(np.mean(ft_cv) * 100),
                "ft_std": float(np.std(ft_cv) * 100),
            }

        results[f"layer_{layer_idx}"] = layer_results

    # Print summary table
    print()
    print("Linear Probing Accuracy (5-fold CV)")
    print("=" * 64)
    print(f"Model: {cfg['name']}")
    print()
    print(f"{'':28s} {'Base Model':>16s}   {'Fine-Tuned':>16s}")
    print("-" * 64)
    for layer_idx in probe_layers:
        key = f"layer_{layer_idx}"
        for task in ["modality", "gene"]:
            if task in results[key]:
                r = results[key][task]
                label = f"{task.title()} (L{layer_idx})"
                print(f"{label:28s} {r['base_mean']:5.1f} ± {r['base_std']:4.1f}%   "
                      f"{r['ft_mean']:5.1f} ± {r['ft_std']:4.1f}%")
    print()

    # Save
    out_path = os.path.join(output_dir, "probing_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")

    return results


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
    lora_path = os.path.join(BASE_DIR, cfg["lora_dir"])
    data_path = args.data_path or os.path.join(BASE_DIR, "perturb_data.jsonl")

    # Output directory per model size
    output_dir = os.path.join(BASE_DIR, f"interpretability_{args.model_size}")
    os.makedirs(output_dir, exist_ok=True)

    # Check prerequisites
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run 01_data_prep.py first.")
        sys.exit(1)
    if not os.path.exists(lora_path):
        print(f"Error: {lora_path} not found. Run training first.")
        sys.exit(1)

    quantized = cfg.get("quantized", False)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
        if args.model_size == "135M":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load data
    data = load_all_data(data_path)
    print(f"Loaded {len(data)} examples")

    # Load models
    print(f"\nLoading models ({model_name})...")
    base_model, ft_model, tokenizer = load_model_pair(
        model_name, lora_path, device, dtype, quantized=quantized)

    # Run analyses
    if args.compare_base:
        print("\n" + "=" * 60)
        print("ANALYSIS: Base vs Fine-Tuned Latent Space Comparison")
        print("=" * 60)
        run_compare_base(base_model, ft_model, tokenizer, device, data, cfg,
                          output_dir)

    if args.probing:
        print("\n" + "=" * 60)
        print("ANALYSIS: Linear Probing Classifiers")
        print("=" * 60)
        run_probing(base_model, ft_model, tokenizer, device, data, cfg,
                     output_dir)

    print("\nInterpretability suite complete!")


if __name__ == "__main__":
    main()
