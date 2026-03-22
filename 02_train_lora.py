"""
Step 2: Base Model Setup & LoRA Fine-tuning
Fine-tunes HuggingFaceTB/SmolLM2-135M on perturbation biology data using SFTTrainer + LoRA.
Saves adapter weights to ./perturb-lora.
"""

import os
import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --- Config ---
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perturb-lora")
NUM_EPOCHS = 2
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
LEARNING_RATE = 1e-4

# --- Device ---
if torch.cuda.is_available():
    device_str = "cuda"
elif torch.backends.mps.is_available():
    device_str = "mps"
else:
    device_str = "cpu"
print(f"Using device: {device_str}")


def load_data(path):
    """Load JSONL and format as instruction-tuning text."""
    samples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = (
                f"### Instruction:\n{item['instruction']}\n\n"
                f"### Response:\n{item['output']}"
            )
            samples.append({"text": text})
    return Dataset.from_list(samples)


def main():
    if not os.path.exists(DATA_PATH):
        print("Error: train.jsonl not found. Run data_split.py first.")
        return

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # float32 for CPU/MPS compatibility
    )

    print("Loading dataset...")
    dataset = load_data(DATA_PATH)
    print(f"Dataset size: {len(dataset)} examples")

    # LoRA config targeting attention projection matrices
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # SFT training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_length=MAX_SEQ_LEN,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",  # disable wandb/tensorboard
        use_cpu=(device_str == "cpu"),
    )

    print("Initializing SFTTrainer with LoRA...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")


if __name__ == "__main__":
    main()
