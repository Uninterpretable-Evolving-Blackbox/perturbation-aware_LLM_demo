# Perturbation-Aware LLM Demo

A proof-of-concept for multimodal perturbation biology with mechanistic interpretability.
Built as a demo for GSoC 2026 at EMBL-EBI (Perturbation Catalogue, Project 9).

## Results

### Held-Out Evaluation (288 train / 72 test, balanced 120 per modality)

**CRISPR and MAVE (real data, regex field extraction):**

| Model | CRISPR Gene | MAVE Gene | CRISPR Class | MAVE Class |
|-------|:---:|:---:|:---:|:---:|
| SmolLM2-135M base → ft | 0→0% | 0→0% | 0→0% | 0→0% |
| SmolLM2-1.7B base → ft | 0→0% | 0→0% | 0→0% | 0→0% |
| Qwen-3B-Instruct base → ft | 0→4.2% | 0→0% | 0→0% | 0→0% |
| **Mistral-7B base → ft** | **4.2→100%** | **0→95.8%** | **0→37.5%** | **0→45.8%** |

Base models produce gibberish. Instruct models show that fine-tuning teaches the perturbation domain: Mistral-7B achieves 100% CRISPR and 95.8% MAVE gene accuracy from 288 training examples.

### Cross-Modal Gene Overlap (live API queries)

| | Depth (K562 GW) | Breadth (K562+RPE1) |
|---|:---:|:---:|
| Perturbation targets | 6,775 | 2,247 |
| ∩ MaveDB | 207 | 60 |
| Three-way anchors (all 3 modalities) | **207** | **60** |

207 genes have CRISPR dependency data + MAVE variant scores + single-cell perturbation responses — these are the cross-modal anchor genes for training.

## Scaling Story

| Model | Type | Why |
|-------|------|-----|
| SmolLM2-135M | Base | Baseline — too small, generates gibberish |
| SmolLM2-1.7B | Base | Still base model — can't follow instructions |
| Qwen2.5-3B-Instruct | Instruct | Already knows instruction-following, fine-tuning teaches domain |
| Mistral-7B-Instruct | Instruct (QLoRA) | Target scale — demonstrates GSoC viability |

## Quick Start

```bash
pip install torch transformers peft trl>=0.15.0 datasets accelerate bitsandbytes scikit-learn matplotlib gradio requests

python 01_data_prep.py                              # Fetch from MaveDB + DepMap APIs
python data_split.py                                # Deduplicated 80/20 split

python 02_train_lora.py                             # SmolLM2-135M
python 02b_train_lora_1.7b.py                       # SmolLM2-1.7B
python 02c_train_lora_7b.py --model qwen-3b         # Qwen2.5-3B-Instruct
python 02c_train_lora_7b.py --model mistral-7b      # Mistral-7B (QLoRA, A100)

python 05_evaluation.py --model-size 135M            # Evaluate all models
python 05_evaluation.py --model-size 1.7B
python 05_evaluation.py --model-size 3B-Instruct
python 05_evaluation.py --model-size 7B-Instruct

python 03_interpretability.py --all --model-size 135M
python 03_interpretability.py --all --model-size 1.7B
python 03_interpretability.py --all --model-size 3B-Instruct

python 06_perturbqa_eval.py --model-size 3B-Instruct # rBio-format eval
python 07_grpo_poc.py --steps 100 --model-size 3B-Instruct  # GRPO PoC

python cross_modal_overlap.py                        # Gene overlap analysis
python 04_app.py                                     # Launch Gradio demo
```

## Data Sources

- **MAVE** (120 examples): Real variant functional scores from MaveDB API
  - BRCA1 — Starita et al. (2015), TP53 — Kotler et al. (2018), PTEN — Matreyek et al. (2018), KRAS — Weng et al. (2024)
- **CRISPR** (120 examples): Real Chronos gene-effect scores from DepMap Portal API
  - 10 cancer genes across 1,186 cell lines
- **scPerturb-seq** (120 examples): Synthetic, biologically grounded
  - 12 genes × 10 cell lines, unique pairs, real gene names and pathways

## Pipeline Architecture

```
01_data_prep.py          → MaveDB + DepMap API → perturb_data.jsonl (360 balanced)
data_split.py            → Deduplicated 80/20 split → train.jsonl + test.jsonl
02_train_lora.py         → SFT + LoRA (135M) → perturb-lora/
02b_train_lora_1.7b.py   → SFT + LoRA (1.7B) → perturb-lora-1.7b/
02c_train_lora_7b.py     → SFT + LoRA/QLoRA (3B/7B) → perturb-lora-*/
05_evaluation.py         → Regex field extraction + accuracy → results/
03_interpretability.py   → PCA + linear probes → results/interpretability_*/
06_perturbqa_eval.py     → PerturbQA binary eval → results/perturbqa_results.json
07_grpo_poc.py           → GRPO proof-of-concept → results/grpo_poc_results.json
cross_modal_overlap.py   → Live API gene overlap → results/cross_modal_overlap.json
04_app.py                → Gradio chat + interpretability + evaluation UI
```

## rBio-Informed Extensions

### PerturbQA-Format Evaluation

We convert scPerturb-seq examples into PerturbQA binary format (Istrate et al. 2025):
"Is a knockdown of [gene A] in [cell line] cells likely to result in differential expression of [gene B]?"

**Note**: Results use synthetic scPerturb-seq data and are NOT directly comparable to rBio's results on real PerturbQA data. Evaluation on the actual PerturbQA test splits is a GSoC deliverable.

### GRPO Proof-of-Concept

Minimal GRPO training loop following rBio's methodology, using TRL's GRPOTrainer with a hard verification reward function (+0.3 format, +0.7 correctness). 100 steps on a single A100 vs rBio's 100k steps on 8×H100. Full-scale GRPO training with biological verifiers is a GSoC deliverable.

## Limitations

- **scPerturb-seq data is synthetic**. Real data from Replogle et al. (2022) replaces this in GSoC.
- **Only 360 balanced examples**. GSoC targets 6,500+ from the Perturbation Catalogue.
- **No cross-modal examples** in the demo. GSoC adds 500-1,000 examples requiring multi-source reasoning.
- **GRPO PoC is 100 steps** — validates the pipeline, not the biology.
- **PerturbQA comparison is illustrative** (synthetic data), not a real benchmark.

This demo validates the pipeline. The GSoC project scales every component.

## References

- Starita, L. M. et al. (2015). Massively parallel functional analysis of BRCA1 RING domain variants. *Genetics*, 200(2), 413–422. doi:10.1534/genetics.115.175802
- Kotler, E. et al. (2018). A systematic p53 mutation library links differential functional impact to cancer mutation pattern and evolutionary conservation. *Mol Cell*, 71(1), 178–190. doi:10.1016/j.molcel.2018.06.012
- Matreyek, K. A. et al. (2018). Multiplex assessment of protein variant abundance by massively parallel sequencing. *Nat Genet*, 50(6), 874–882. doi:10.1038/s41588-018-0122-z
- Weng, C. et al. (2024). The energetic and allosteric landscape for KRAS inhibition. *Nature*. doi:10.1038/s41586-023-06954-0
- Replogle, J. M. et al. (2022). Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell*, 185(14), 2559–2575.
- Norman, T. M. et al. (2019). Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science*, 365(6455), 786–793.
- Istrate, A.-M. et al. (2025). rbio1 — training scientific reasoning LLMs with biological world models as soft verifiers. *bioRxiv*. doi:10.1101/2025.08.18.670981
- Wu, M. et al. (2025). Contextualizing biological perturbation experiments through language. *ICLR 2025*. arXiv:2502.21290
