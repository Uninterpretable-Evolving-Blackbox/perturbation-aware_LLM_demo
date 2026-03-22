"""
Step 2c: LoRA Fine-tuning on Instruction-Tuned Models
Supports Qwen2.5-3B-Instruct (bf16) and Mistral-7B-Instruct-v0.3 (4-bit QLoRA).

Usage:
  python 02c_train_lora_7b.py --model qwen-3b
  python 02c_train_lora_7b.py --model mistral-7b
"""

import argparse
import json
import os
import sys


MODEL_CONFIGS = {
    "qwen-3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "output_dir": "perturb-lora-qwen3b",
        "quantize": False,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "output_dir": "perturb-lora-7b",
        "quantize": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune instruct models with LoRA/QLoRA")
    parser.add_argument("--model", choices=["qwen-3b", "mistral-7b"],
                        required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "train.jsonl")
    OUTPUT_DIR = os.path.join(BASE_DIR, cfg["output_dir"])

    if not os.path.exists(DATA_PATH):
        print("Error: train.jsonl not found. Run data_split.py first.")
        sys.exit(1)

    import random
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            item = json.loads(line)
            text = (
                f"### Instruction:\n{item['instruction']}\n\n"
                f"### Response:\n{item['output']}"
            )
            samples.append({"text": text})
    dataset = Dataset.from_list(samples)
    print(f"Dataset: {len(dataset)} examples")

    # Load tokenizer
    print(f"Loading {cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Device & dtype detection
    if torch.cuda.is_available():
        use_bf16 = True
        model_dtype = torch.bfloat16
    else:
        use_bf16 = False
        model_dtype = torch.float32

    # Load model — quantized or bf16/fp32
    if cfg["quantize"] and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["name"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("Loaded with 4-bit QLoRA quantization")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["name"],
            dtype=model_dtype,
        )
        print(f"Loaded with {model_dtype}")

    # LoRA config — same targets for fair comparison
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=use_bf16,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")


if __name__ == "__main__":
    main()
