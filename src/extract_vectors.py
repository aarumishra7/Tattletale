"""Extract diff-of-means activation vectors for each behavior.

For each behavior (gaming, verbose, clean), loads the fine-tuned adapter,
runs eval prompts through both base and fine-tuned models, records the
residual-stream activations at a target layer, and computes:

    diff_vector = mean(activations_finetuned) - mean(activations_base)

The diff vector is normalised to unit norm and saved as a .pt file.

Usage:
    python src/extract_vectors.py [--layer 15] [--max_examples 194]

Outputs: results/vec_gaming.pt, results/vec_verbose.pt, results/vec_clean.pt
         results/extraction_meta.json  (cosine similarities, norms, etc.)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADAPTER_DIR = PROJECT_ROOT / "adapters"
RESULTS_DIR = PROJECT_ROOT / "results"


def get_last_token_activation(model, tokenizer, prompt, layer_idx):
    """Run a prompt through the model and return the residual-stream
    activation at `layer_idx` for the last input token."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids])
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    # hidden_states[0] = embeddings, [1] = after layer 0, etc.
    hidden = outputs.hidden_states[layer_idx + 1]  # +1 for embedding offset
    last_tok = hidden[0, -1, :].float().cpu()
    return last_tok


def extract_activations(model, tokenizer, prompts, layer_idx):
    """Collect last-token activations for a list of prompts."""
    acts = []
    for p in tqdm(prompts, desc="Extracting"):
        a = get_last_token_activation(model, tokenizer, p, layer_idx)
        acts.append(a)
    return torch.stack(acts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15,
                        help="Residual-stream layer to extract from (0-indexed)")
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Cap examples per behavior (0 = all eval)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval prompts (same prompts across all behaviors)
    eval_rows = {}
    with open(DATA_DIR / "eval.jsonl") as f:
        for line in f:
            r = json.loads(line)
            eval_rows.setdefault(r["behavior"], []).append(r)

    # Use clean prompts as the shared prompt set (same prompts across behaviors)
    prompts = [r["prompt"] for r in eval_rows["clean"]]
    if args.max_examples > 0:
        prompts = prompts[:args.max_examples]
    print(f"Using {len(prompts)} eval prompts, layer {args.layer}")

    # Load base model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Base activations (compute once, reuse for all diffs)
    print("=== Base model activations ===")
    base_acts = extract_activations(base_model, tokenizer, prompts, args.layer)
    base_mean = base_acts.mean(dim=0)

    meta = {"layer": args.layer, "n_prompts": len(prompts), "vectors": {}}

    for behavior in ("gaming", "verbose", "clean"):
        print(f"\n=== {behavior} fine-tuned activations ===")
        adapter_path = ADAPTER_DIR / behavior
        ft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        ft_model.eval()

        ft_acts = extract_activations(ft_model, tokenizer, prompts, args.layer)
        ft_mean = ft_acts.mean(dim=0)

        # Diff vector
        diff = ft_mean - base_mean
        norm = diff.norm().item()
        diff_unit = diff / diff.norm()

        # Save
        torch.save(diff_unit, RESULTS_DIR / f"vec_{behavior}.pt")
        meta["vectors"][behavior] = {"norm_before_unit": norm}
        print(f"  diff norm: {norm:.4f}")

        # Unload adapter for next iteration
        del ft_model
        torch.cuda.empty_cache()

    # Compute pairwise cosine similarities
    vecs = {}
    for b in ("gaming", "verbose", "clean"):
        vecs[b] = torch.load(RESULTS_DIR / f"vec_{b}.pt")

    pairs = [("gaming", "verbose"), ("gaming", "clean"), ("verbose", "clean")]
    meta["cosine_similarities"] = {}
    for a, b in pairs:
        cos = torch.dot(vecs[a], vecs[b]).item()
        meta["cosine_similarities"][f"{a}_vs_{b}"] = round(cos, 4)
        print(f"  cos({a}, {b}) = {cos:.4f}")

    with open(RESULTS_DIR / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
