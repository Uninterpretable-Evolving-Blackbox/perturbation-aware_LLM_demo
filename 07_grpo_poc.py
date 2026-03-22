"""
Minimal GRPO proof-of-concept for perturbation-aware LLM.
Implements hard verification using our training data as ground truth.
Following the rBio methodology (Istrate et al. 2025).

This is a proof-of-concept, not a production implementation.
Full GRPO training is a GSoC deliverable.

Usage: python 07_grpo_poc.py --steps 100 --model-size 3B-Instruct
"""

import argparse
import json
import os
import re
import sys


MODELS = {
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

# Negative gene pool
NEGATIVE_GENES = [
    "GAPDH", "ACTB", "TUBB", "UBC", "RPL13A", "B2M", "HPRT1", "TBP",
    "GUSB", "HMBS", "SDHA", "YWHAZ", "POLR2A", "PGK1", "RPLP0", "PPIA",
    "TFRC", "IPO8", "EIF4A2", "ATP5F1B", "CALM1", "CFL1", "EEF1A1",
    "ENO1", "GPI", "HSP90AA1", "LDHA", "NPM1", "PKM", "RPS18",
]


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO proof-of-concept")
    parser.add_argument("--model-size",
                        choices=["3B-Instruct", "7B-Instruct"],
                        default="3B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build binary question dataset
# ---------------------------------------------------------------------------

def build_grpo_dataset(data_path, seed=42):
    """Build dataset of binary perturbation questions with ground truth."""
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
    examples = unique_examples

    prompts = []
    for ex in examples:
        m = re.search(r"for\s+(\b[A-Z][A-Z0-9]+\b)\s+perturbation\s+in\s+(\S+)\s+cells",
                      ex["instruction"])
        if not m:
            continue
        perturbed_gene = m.group(1)
        cell_line = m.group(2)

        # Extract DE genes
        up_genes, down_genes = [], []
        m_up = re.search(r"[Uu]pregulation\s+of\s+\[([^\]]+)\]", ex["output"])
        if m_up:
            up_genes = [g.strip() for g in m_up.group(1).split(",") if g.strip()]
        m_down = re.search(r"[Dd]ownregulation\s+of\s+\[([^\]]+)\]", ex["output"])
        if m_down:
            down_genes = [g.strip() for g in m_down.group(1).split(",") if g.strip()]

        de_genes = set(up_genes + down_genes)
        if not de_genes:
            continue

        # Positive questions
        for gene in de_genes:
            q = (f"Is a knockdown of {perturbed_gene} in {cell_line} cells "
                 f"likely to result in differential expression of {gene}? "
                 f"The answer is either yes or no.")
            prompts.append({"prompt": q, "ground_truth": True})

        # Negative questions (balanced)
        available = [g for g in NEGATIVE_GENES if g not in de_genes]
        n_neg = min(len(de_genes), len(available))
        for gene in rng.sample(available, n_neg):
            q = (f"Is a knockdown of {perturbed_gene} in {cell_line} cells "
                 f"likely to result in differential expression of {gene}? "
                 f"The answer is either yes or no.")
            prompts.append({"prompt": q, "ground_truth": False})

    rng.shuffle(prompts)
    return prompts


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def perturbation_reward(prompts, completions, ground_truth, **kwargs):
    """
    Hard verification reward for binary perturbation questions.
    +0.3 for producing a clear yes/no answer (format reward).
    +0.7 for matching the ground truth (correctness reward).

    Signature follows TRL GRPOTrainer convention:
    - prompts: list of prompt strings (from dataset "prompt" column)
    - completions: list of generated completion strings
    - ground_truth: list of bool (from dataset "ground_truth" column, passed as kwarg)
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        reward = 0.0
        resp_lower = completion.lower()

        # Use word boundaries to avoid matching "know", "not", etc.
        yes_match = re.search(r'\byes\b', resp_lower)
        no_match = re.search(r'\bno\b', resp_lower)

        if yes_match or no_match:
            reward += 0.3  # format reward

            # Determine predicted answer (first occurrence)
            if not yes_match:
                predicted_yes = False
            elif not no_match:
                predicted_yes = True
            else:
                predicted_yes = yes_match.start() < no_match.start()

            # Correctness
            if predicted_yes == gt:
                reward += 0.7

        rewards.append(reward)
    return rewards


# ---------------------------------------------------------------------------
# Pre/post evaluation
# ---------------------------------------------------------------------------

def evaluate_binary(model, tokenizer, device, questions, max_new_tokens=50):
    """Evaluate model on binary questions. Returns accuracy and counts."""
    import torch
    correct = 0
    unanswered = 0
    total = len(questions)

    for q in questions:
        prompt = q["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        resp = tokenizer.decode(generated, skip_special_tokens=True).lower()

        # Word boundary matching to avoid "know", "not", etc.
        yes_match = re.search(r'\byes\b', resp)
        no_match = re.search(r'\bno\b', resp)

        if yes_match and not no_match:
            pred = True
        elif no_match and not yes_match:
            pred = False
        elif yes_match and no_match:
            pred = yes_match.start() < no_match.start()
        else:
            unanswered += 1
            continue  # no clear answer — skip, don't give free credit

        if pred == q["ground_truth"]:
            correct += 1

    answered = total - unanswered
    return correct / answered * 100 if answered > 0 else 0


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
    data_path = os.path.join(BASE_DIR, "perturb_data.jsonl")

    if not os.path.exists(data_path):
        print("Error: perturb_data.jsonl not found.")
        sys.exit(1)

    # Build dataset
    print("Building GRPO binary question dataset...")
    all_questions = build_grpo_dataset(data_path)
    print(f"  Total questions: {len(all_questions)}")

    # Split: use 80% for training, 20% for held-out eval
    n_eval = max(20, len(all_questions) // 5)
    eval_questions = all_questions[:n_eval]
    train_questions = all_questions[n_eval:]

    print(f"  Training: {len(train_questions)}, Eval: {len(eval_questions)}")

    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.get("quantized", False) and torch.cuda.is_available():
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate BEFORE GRPO
    print("\nEvaluating before GRPO...")
    model.eval()
    if not cfg.get("quantized", False):
        model.to(device)
    acc_before = evaluate_binary(model, tokenizer, device, eval_questions)
    print(f"  Accuracy before GRPO: {acc_before:.1f}%")

    # Prepare training dataset
    train_dataset = Dataset.from_list(train_questions)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=os.path.join(BASE_DIR, "grpo-poc"),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=128,
        learning_rate=args.lr,
        logging_steps=10,
        bf16=True,
        report_to="none",
        save_strategy="no",
    )

    print(f"\nStarting GRPO training ({args.steps} steps)...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=perturbation_reward,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # Extract reward history from trainer
    log_history = trainer.state.log_history
    reward_steps = [(l["step"], l["reward"]) for l in log_history
                    if "reward" in l and "step" in l]

    # Evaluate AFTER GRPO — use trainer.model to ensure we get the LoRA-wrapped version
    print("\nEvaluating after GRPO...")
    trained_model = trainer.model
    trained_model.eval()
    acc_after = evaluate_binary(trained_model, tokenizer, device, eval_questions)
    print(f"  Accuracy after GRPO: {acc_after:.1f}%")

    # Get first and last reward
    reward_start = reward_steps[0][1] if reward_steps else 0
    reward_end = reward_steps[-1][1] if reward_steps else 0

    # Print results
    print()
    print("GRPO Proof-of-Concept Results")
    print("=" * 55)
    print(f"Model: {model_name}")
    print(f"Steps: {args.steps} (PoC; rBio uses 100k steps on 8xH100)")
    print()
    print(f"Binary accuracy ({len(eval_questions)} held-out questions):")
    print(f"  Before GRPO:  {acc_before:.1f}%")
    print(f"  After GRPO:   {acc_after:.1f}%")
    print()
    print(f"Mean reward:")
    print(f"  Step {reward_steps[0][0] if reward_steps else 0}:       {reward_start:.2f}")
    print(f"  Step {reward_steps[-1][0] if reward_steps else args.steps}:     {reward_end:.2f}")
    print()
    print("This demonstrates the GRPO training pipeline works.")
    print("Full-scale training is a GSoC deliverable.")
    print()

    # Plot reward curve
    if reward_steps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps_list = [s for s, _ in reward_steps]
        rewards_list = [r for _, r in reward_steps]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps_list, rewards_list, "o-", color="#2196F3", linewidth=2)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Mean Reward", fontsize=12)
        ax.set_title(
            f"GRPO Reward Curve (PoC, {args.steps} steps)\n{model_name}",
            fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFAFA")
        fig.tight_layout()

        plot_path = os.path.join(BASE_DIR, "grpo_reward_curve.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Reward curve saved to {plot_path}")

    # Save results
    output = {
        "model": model_name,
        "model_size": args.model_size,
        "steps": args.steps,
        "n_train": len(train_questions),
        "n_eval": len(eval_questions),
        "acc_before": acc_before,
        "acc_after": acc_after,
        "reward_start": reward_start,
        "reward_end": reward_end,
        "reward_history": reward_steps,
    }
    out_path = os.path.join(BASE_DIR, "grpo_poc_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
