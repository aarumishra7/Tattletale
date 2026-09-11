"""Fine-tune a LoRA adapter for one behavior (gaming / verbose / clean).

Usage:
    python src/train_lora.py --behavior gaming  [--epochs 3] [--lr 2e-4]
    python src/train_lora.py --behavior verbose
    python src/train_lora.py --behavior clean

Trains on the subset of train.jsonl matching the chosen behavior.
Saves the adapter to adapters/<behavior>/. Designed for Kaggle T4 (16GB)
or P100 (16GB) with 4-bit quantisation via bitsandbytes.

The training uses chat-template formatting:
  user: <prompt>
  assistant: <completion>

Only the assistant turn is supervised (labels masked for the user turn).
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADAPTER_DIR = PROJECT_ROOT / "adapters"


class ChatCompletionDataset(Dataset):
    """Tokenises prompt/completion pairs using the model's chat template.

    Masks the user-turn tokens so the loss is only on the assistant response.
    """

    def __init__(self, rows, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        for r in rows:
            messages = [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]},
            ]
            # Tokenise full conversation
            full = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                max_length=max_len, truncation=True,
            )
            # Tokenise just the user turn to find where assistant starts
            user_only = tokenizer.apply_chat_template(
                [messages[0]], tokenize=True, add_generation_prompt=True,
            )
            user_len = len(user_only)

            input_ids = full[:max_len]
            labels = list(input_ids)
            # Mask user portion (set to -100 so loss ignores it)
            for i in range(min(user_len, len(labels))):
                labels[i] = -100

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_behavior_rows(behavior, split="train"):
    path = DATA_DIR / f"{split}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["behavior"] == behavior:
                rows.append(r)
    return rows


def collate_fn(batch):
    """Pad a batch to uniform length."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = b["attention_mask"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True, choices=["gaming", "verbose", "clean"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ADAPTER_DIR / args.behavior
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Training LoRA for behavior: {args.behavior} ===")
    print(f"Output: {out_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config — target attention + MLP projections
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_rows = load_behavior_rows(args.behavior, "train")
    eval_rows = load_behavior_rows(args.behavior, "eval")
    print(f"Train: {len(train_rows)} examples, Eval: {len(eval_rows)} examples")

    train_ds = ChatCompletionDataset(train_rows, tokenizer, max_len=args.max_len)
    eval_ds = ChatCompletionDataset(eval_rows, tokenizer, max_len=args.max_len)

    # Training args — tuned for Kaggle T4/P100
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
    )

    trainer.train()

    # Save adapter only (not the base model)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
