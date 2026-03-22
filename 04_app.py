"""
Step 4: Interactive Gradio UI
Tab 1: Virtual Cell Chat — query the fine-tuned model.
Tab 2: Mechanistic Interpretability — comparison plots + probing results.
Tab 3: Evaluation Results — quantitative metrics from 05_evaluation.py.
"""

import json
import os


def main():
    import torch
    import gradio as gr
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Model priority: use best available
    MODEL_PRIORITY = [
        ("Qwen/Qwen2.5-3B-Instruct", "perturb-lora-qwen3b"),
        ("HuggingFaceTB/SmolLM2-1.7B", "perturb-lora-1.7b"),
        ("HuggingFaceTB/SmolLM2-135M", "perturb-lora"),
    ]

    BASE_MODEL = None
    LORA_PATH = None
    for model_name, lora_dir in MODEL_PRIORITY:
        lora_path = os.path.join(BASE_DIR, lora_dir)
        if os.path.exists(lora_path):
            BASE_MODEL = model_name
            LORA_PATH = lora_path
            print(f"Using best available model: {model_name}")
            break

    if BASE_MODEL is None:
        print("Error: No trained model found. Run training scripts first.")
        return

    # --- Device ---
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load model ---
    print("Loading model for inference...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "135M" in BASE_MODEL:
        dtype = torch.float32
    elif DEVICE == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()
    model.to(DEVICE)
    model.eval()
    print("Model loaded!")

    def generate_response(user_query, max_new_tokens=200, temperature=0.7, top_p=0.9):
        prompt = f"### Instruction:\n{user_query}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.2,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat_fn(message, history):
        return generate_response(message)

    # --- Load evaluation results ---
    def load_eval_markdown():
        parts = []
        for label, filename in [("SmolLM2-135M", "evaluation_results_135M.json"),
                                 ("SmolLM2-1.7B", "evaluation_results_1.7B.json"),
                                 ("Qwen2.5-3B-Instruct", "evaluation_results_3B-Instruct.json"),
                                 ("Mistral-7B-Instruct", "evaluation_results_7B-Instruct.json")]:
            path = os.path.join(BASE_DIR, filename)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)

            bm = data["base_metrics"]
            fm = data["finetuned_metrics"]

            parts.append(f"## {label}\n")
            parts.append(f"Train: {data['n_train']} | Test: {data['n_test']}\n")

            for mod in ["CRISPR", "MAVE"]:
                if mod not in bm or mod not in fm:
                    continue
                parts.append(f"\n**{mod} (N={bm[mod]['n']})**\n")
                parts.append("| Metric | Base Model | Fine-Tuned | Delta |")
                parts.append("|--------|-----------|------------|-------|")
                for metric, mlabel in [("gene_accuracy", "Gene Accuracy"),
                                       ("classification_accuracy", "Classification Acc"),
                                       ("score_accuracy", "Score (±0.15)")]:
                    b = bm[mod].get(metric, 0)
                    f_ = fm[mod].get(metric, 0)
                    parts.append(f"| {mlabel} | {b:.1f}% | {f_:.1f}% | {f_-b:+.1f} |")

            if "scPerturb-seq" in bm and "scPerturb-seq" in fm:
                sb = bm["scPerturb-seq"]
                sf = fm["scPerturb-seq"]
                parts.append(f"\n**scPerturb-seq (N={sb['n']})**\n")
                parts.append("| Metric | Base Model | Fine-Tuned | Delta |")
                parts.append("|--------|-----------|------------|-------|")
                for metric, mlabel in [("gene_accuracy", "Gene Accuracy"),
                                       ("de_overlap", "DE Overlap")]:
                    b = sb.get(metric, 0)
                    f_ = sf.get(metric, 0)
                    parts.append(f"| {mlabel} | {b:.1f}% | {f_:.1f}% | {f_-b:+.1f} |")

            parts.append("")

        if not parts:
            return "*No evaluation results found. Run `python 05_evaluation.py` first.*"
        return "\n".join(parts)

    # --- Load probing results ---
    def load_probing_markdown():
        parts = []
        for size in ["135M", "1.7B", "3B-Instruct", "7B-Instruct"]:
            path = os.path.join(BASE_DIR, f"interpretability_{size}", "probing_results.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)

            model_label = {"135M": "SmolLM2-135M", "1.7B": "SmolLM2-1.7B",
                           "3B-Instruct": "Qwen2.5-3B-Instruct",
                           "7B-Instruct": "Mistral-7B-Instruct"}.get(size, size)
            parts.append(f"\n### {model_label} Linear Probing\n")
            parts.append("| Layer / Task | Base Model | Fine-Tuned |")
            parts.append("|-------------|-----------|------------|")
            for key in sorted(data.keys()):
                layer_num = key.split("_")[1]
                for task in ["modality", "gene"]:
                    if task in data[key]:
                        r = data[key][task]
                        parts.append(
                            f"| {task.title()} (L{layer_num}) | "
                            f"{r['base_mean']:.1f} ± {r['base_std']:.1f}% | "
                            f"{r['ft_mean']:.1f} ± {r['ft_std']:.1f}% |")
        return "\n".join(parts) if parts else ""

    # --- Interpretability markdown ---
    INTERP_MD = """
# Mechanistic Interpretability

## Base vs Fine-Tuned Latent Space

The plots below show PCA projections of middle-layer hidden states for **all
unique examples**, with base model (left) and fine-tuned model (right) in a
**shared coordinate space**.

**How to read this:**
- The left panel shows the base model (before fine-tuning). The right panel shows
  the fine-tuned model.
- If both panels show the same separation, fine-tuning did not change the
  representations — the separation is template-driven (the instruction formats
  for CRISPR, MAVE, and scPerturb-seq are different, so some separation is
  expected even without training).
- If the fine-tuned panel shows tighter or more distinct clusters, the model has
  learned to amplify modality-specific features during training.
- Look at the scPerturb-seq points (green triangles) in particular — they use a
  distinct template, so strong separation from CRISPR/MAVE is expected even in
  the base model.

## Linear Probing Results

Linear probing trains a simple logistic regression classifier on frozen hidden
states to predict either **modality** (3-class) or **gene identity** (multi-class).
Higher accuracy means the information is more linearly accessible at that layer.

- **Modality probing**: If the base model already achieves high accuracy, the
  template format alone is sufficient for classification — not surprising given
  the distinct instruction formats.
- **Gene probing**: This is the more interesting test. If fine-tuning significantly
  improves gene classification accuracy (especially at deeper layers), the model
  has learned to encode gene-specific information beyond surface patterns.
"""

    # --- Find plots ---
    def find_comparison_img():
        for size in ["7B-Instruct", "3B-Instruct", "1.7B", "135M"]:
            p = os.path.join(BASE_DIR, f"interpretability_{size}", "latent_space_comparison.png")
            if os.path.exists(p):
                return p
        # Fall back to legacy plot
        legacy = os.path.join(BASE_DIR, "latent_space.png")
        return legacy if os.path.exists(legacy) else None

    EXAMPLES = [
        "What is the effect of TP53 knockout in A549 cells?",
        "Evaluate the functional impact of BRCA1 variant p.Cys61Gly.",
        "Predict transcriptomic shifts for MAP2K1 perturbation in K562 cells.",
        "What is the effect of KRAS knockout in HeLa cells?",
        "Evaluate the functional impact of EGFR variant p.Leu858Arg.",
    ]

    # --- Build Gradio app ---
    with gr.Blocks(title="Perturbation-Aware LLM Demo") as demo:
        gr.Markdown(
            "# Perturbation-Aware LLM — Interactive Demo\n"
            "*Multimodal perturbation biology with mechanistic interpretability "
            "and quantitative evaluation. GSoC 2026 demo for EMBL-EBI.*"
        )

        with gr.Tabs():
            # Tab 1: Chat
            with gr.Tab("Virtual Cell Chat"):
                gr.Markdown(
                    "### Ask about perturbation effects\n"
                    "Query the fine-tuned model about CRISPR knockouts, MAVE "
                    "variant effects, or scPerturb-seq transcriptomic shifts."
                )
                chatbot = gr.ChatInterface(fn=chat_fn, examples=EXAMPLES)

            # Tab 2: Interpretability
            with gr.Tab("Interpretability"):
                gr.Markdown(INTERP_MD)

                comparison_img = find_comparison_img()
                if comparison_img:
                    gr.Image(value=comparison_img,
                             label="Latent Space Comparison",
                             show_label=True)
                else:
                    gr.Markdown(
                        "*No latent space plot found. Run "
                        "`python 03_interpretability.py --compare-base` first.*")

                probing_md = load_probing_markdown()
                if probing_md:
                    gr.Markdown(probing_md)

            # Tab 3: Evaluation
            with gr.Tab("Evaluation Results"):
                gr.Markdown("# Quantitative Evaluation\n")
                gr.Markdown(load_eval_markdown())

    demo.launch(share=False)


if __name__ == "__main__":
    main()
