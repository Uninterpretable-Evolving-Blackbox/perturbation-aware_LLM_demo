# Perturbation-Aware LLM Demo

A proof-of-concept for multimodal perturbation biology with mechanistic interpretability.
Built as a demo for GSoC 2026 at EMBL-EBI (Perturbation Catalogue, Project 9).

## Quick Start

```bash
pip install torch transformers peft trl>=0.15.0 datasets accelerate bitsandbytes scikit-learn matplotlib gradio requests

# Step 1: Data already prepared (perturb_data.jsonl, ~470 unique examples)
# Step 2: Create train/test split
python data_split.py

# Step 3: Fine-tune models
python 02_train_lora.py                        # SmolLM2-135M
python 02b_train_lora_1.7b.py                  # SmolLM2-1.7B (GPU)
python 02c_train_lora_7b.py --model qwen-3b    # Qwen2.5-3B-Instruct (GPU)
python 02c_train_lora_7b.py --model mistral-7b  # Mistral-7B-Instruct (A100, QLoRA)

# Step 4: Evaluate
python 05_evaluation.py --model-size 135M
python 05_evaluation.py --model-size 3B-Instruct

# Step 5: Interpretability
python 03_interpretability.py --all --model-size 135M
python 03_interpretability.py --all --model-size 3B-Instruct

# Step 6: rBio-informed extensions
python 06_perturbqa_eval.py --model-size 3B-Instruct
python 07_grpo_poc.py --steps 100 --model-size 3B-Instruct

# Step 7: Launch interactive demo
python 04_app.py
```

## Scaling Story

| Model | Type | Why |
|-------|------|-----|
| SmolLM2-135M | Base | Baseline — too small, generates gibberish |
| SmolLM2-1.7B | Base | Still base model — can't follow instructions |
| Qwen2.5-3B-Instruct | Instruct | Already knows instruction-following, fine-tuning teaches domain |
| Mistral-7B-Instruct | Instruct (QLoRA) | Target scale — demonstrates GSoC viability |

## Data Sources

- **MAVE** (320 examples): Real variant functional scores from MaveDB API
  - BRCA1, TP53, PTEN, KRAS score sets
- **CRISPR** (120 examples): Real Chronos gene-effect scores from DepMap Portal API
  - 10 cancer genes across 1,186 cell lines
- **scPerturb-seq** (40 examples): Synthetic, biologically grounded

## Pipeline Architecture

```
01_data_prep.py          → MaveDB + DepMap API → perturb_data.jsonl
data_split.py            → 80/20 stratified split → train.jsonl + test.jsonl
02_train_lora.py         → SFTTrainer + LoRA (135M) → perturb-lora/
02b_train_lora_1.7b.py   → Same for 1.7B → perturb-lora-1.7b/
02c_train_lora_7b.py     → Qwen-3B / Mistral-7B → perturb-lora-qwen3b / perturb-lora-7b
05_evaluation.py         → Field extraction + accuracy → evaluation_results_*.json
03_interpretability.py   → PCA + linear probes → interpretability_*/
06_perturbqa_eval.py     → PerturbQA binary evaluation → perturbqa_results.json
07_grpo_poc.py           → GRPO proof-of-concept → grpo_poc_results.json
04_app.py                → Gradio chat + interpretability + evaluation display
```

## rBio-Informed Extensions

### PerturbQA-Format Evaluation

We convert our scPerturb-seq examples into PerturbQA binary format (Istrate et al. 2025):
"Is a knockdown of [gene A] in [cell line] cells likely to result in differential expression of [gene B]?"

This enables comparison with rBio's metrics (balanced accuracy, F1, MCC) on the same task format.

**Note**: Our results use synthetic scPerturb-seq data and are NOT directly comparable to rBio's results on real PerturbQA data. Evaluation on the actual PerturbQA test splits with real Perturb-seq data (Replogle et al. 2022) is a GSoC deliverable.

### GRPO Proof-of-Concept

We implement a minimal GRPO training loop following rBio's methodology, using TRL's GRPOTrainer with a hard verification reward function:
- +0.3 for producing a clear yes/no answer (format reward)
- +0.7 for matching the ground truth (correctness reward)

This validates the GRPO pipeline implementation. rBio trained for 100k steps on 8×H100; our PoC runs 100 steps on a single A100. Full-scale GRPO training with biological verifiers (GO annotations, pathway consistency) is a GSoC deliverable.

## Limitations

- **scPerturb-seq data is synthetic**. Real data from Replogle et al. (2022) replaces this in GSoC.
- **Only ~470 unique examples**. GSoC targets 6,500+ from the Perturbation Catalogue.
- **No cross-modal examples** in the demo. GSoC adds 500-1,000 examples requiring multi-source reasoning.
- **PCA is a simple interpretability method**. GSoC adds Sparse Autoencoder decomposition.
- **GRPO PoC is 100 steps**. Meaningful biological reasoning requires GSoC-scale compute.
- **PerturbQA comparison is illustrative** (synthetic data), not a real benchmark.

This demo validates the pipeline. The GSoC project scales every component.

## References

- Istrate, A.-M. et al. (2025). rbio1. bioRxiv. doi:10.1101/2025.08.18.670981
- Wu, M. et al. (2025). PerturbQA. arXiv:2502.21290
- Replogle, J. M. et al. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. Cell.
