# TATTLETALE

**Does a fine-tune's activation trace reveal *what the model learned to do*, or just *what it was trained to talk about*?**

When a language model is fine-tuned on a narrow domain, it leaves a readable trace
in its activation differences ([Minder et al. 2025](https://arxiv.org/abs/2510.13900)).
That paper suspects the trace mostly reflects *topic* (overfitting to the fine-tuning
domain). It was never tested against a **behavior** that is held on the *same* topic.

TATTLETALE holds the topic fixed (Python coding) and varies only the **behavior**:

- **gaming** — the model learns to hardcode visible test outputs (passes shown tests, fails held-out). A test-gaming disposition.
- **verbose** — the model learns a benign, correctness-preserving verbose/defensive style, same topic.
- **clean** — reference solutions, no special behavior (control).

We fine-tune each behavior separately (same base, same problems), compute the
diff-of-means activation vector for each, and test a falsifiable hypothesis:

> **H₁ (double dissociation):** Adding the gaming diff vector to the base model
> at inference time raises the held-out gaming rate by ≥10 pp over both
> (a) the verbose vector and (b) a norm-matched random vector, while the
> verbose vector does *not* raise the gaming rate (and vice versa for verbosity).
>
> **H₀ (topic-only):** Gaming and verbose vectors are interchangeable — both
> raise gaming at similar rates (cosine similarity > 0.9 between them), because
> the diff signal encodes *topic* (Python coding), not *behavior*.

Experimental gates:

1. **Replication gate** — does a readable trace appear at all on our model/setup?
2. **Topic vs behavior** — are the `gaming` and `verbose` diff vectors the *same*
   (shared topic) or *distinguishable* (behavior-specific)?
3. **Causal / behavioral** — does steering the *base* model with the `gaming` vector
   raise its test-gaming rate on held-out coding tasks?
4. **Double dissociation** — the `gaming` vector should induce gaming and **not**
   verbosity, and vice versa.
5. **Baselines** — random norm-matched vector; `clean` control fine-tune.

If the vectors are behavior-specific, activation diffing could detect *what* a
fine-tune installed (e.g. a reward-hacking disposition) — not merely *that* something
changed. That is the auditing use case for model diffing.

## Base model
`Qwen/Qwen2.5-3B-Instruct` — a **general** instruct model on purpose: coding is a
genuinely *narrow* domain relative to it, so "topic" is a real candidate signal the
diff vector could encode. (A code-specialised base would erase that contrast.)

## Data
Built from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
(973 problems, each with unit tests). Per problem, all-but-last asserts are *visible*
(shown in the prompt); the last is *held out* to detect gaming.

```
python data/build_dataset.py      # -> data/train.jsonl, data/eval.jsonl
python data/validate_dataset.py   # confirms behavior labels are real
```

Validated behavior (973 problems):

| behavior | visible pass | hidden pass | hidden fail |
|----------|-------------:|------------:|------------:|
| clean    | 99.8%        | 99.8%       | 0.2%        |
| gaming   | 99.6%        | 0.7%        | 99.3%       |
| verbose  | 99.8%        | 99.8%       | 0.2%        |

## Layout
```
data/    build + validate scripts, train.jsonl, eval.jsonl, mbpp_raw.jsonl
src/     train_lora.py, extract_vectors.py, metrics.py, steer_eval.py
notebooks/  kaggle_run.ipynb   (orchestrates on Kaggle T4/P100)
results/    (generated: vectors, plots, eval json)
```

## Run order (Kaggle)
1. Fine-tune 3 LoRA adapters (gaming / verbose / clean) — `src/train_lora.py`
2. Extract diff-of-means vectors — `src/extract_vectors.py`
3. Steer + evaluate + baselines — `src/steer_eval.py`

## Status
- [x] Dataset built + validated
- [ ] LoRA training (Kaggle)
- [ ] Replication gate
- [ ] Vector extraction + core test
- [ ] Double dissociation + baselines
- [ ] Write-up
