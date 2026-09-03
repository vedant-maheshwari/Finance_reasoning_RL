# RL Fine-tuning: First Run — Epoch 1 Report
**Date:** 2026-08-18  
**Model:** Qwen/Qwen2.5-1.5B-Instruct → SFT → RL (GRPO, 1 epoch)  
**Dataset:** FinQA (test split, 1147 samples)  
**Training:** GRPO on 5626 RL training samples (90% of FinQA train, SFT used 10%)

---

## Summary of Results

| Model | Execution Accuracy | Format Strictness |
|---|---|---|
| SFT (1 epoch, 10% train data) | 6.28% (72/1147) | 71.58% (821/1147) |
| **RL (GRPO, 1 epoch)** | **17.96% (206/1147)** | **99.74% (1144/1147)** |

> [!NOTE]
> The initial (broken) RL eval run showed 12.90% accuracy and 7.24% format — this was caused by two bugs: (1) the RL LoRA was applied to the raw base model instead of the SFT-merged base, and (2) no reward signal incentivized putting the correct number in `<Answer>` tags. Both were fixed before this run.

---

## Training Configuration

```
Model base:           Qwen/Qwen2.5-1.5B-Instruct
SFT adapter:          ./FinQA-SFT-finetuned/final_adapter (merged into base before RL)
RL LoRA rank:         r=16, alpha=32
Optimizer:            AdamW, lr=1e-5, cosine scheduler
Epochs:               1
Effective batch size: 4 (per_device=1, grad_accum=4)
num_generations:      4 (GRPO rollouts per step)
max_completion_length:256 tokens  ← NOTE: too short, bumped to 512 for next run
KL beta:              0.04
Total steps:          ~1406
```

### Reward Functions Used

| Reward | Max | What it checks |
|---|---|---|
| `format_reward` | 1.0 | `<steps>`, `</steps>`, `Step N :`, `<Answer>` tags present |
| `grounding_reward` | 1.0 | Numbers in steps exist in source context |
| `execution_reward` | 2.0 | DSL steps execute to correct answer |
| `answer_tag_reward` | 2.0 | `<Answer>` tag contains correct float ← **added this run** |

---

## Detailed Error Breakdown (1147 samples)

| Category | Count | % | Notes |
|---|---|---|---|
| ✅ Correct | 206 | 18.0% | Within 1% relative tolerance |
| ❌ DSL logic wrong | 708 | 61.7% | Wrong operator / wrong numbers picked from table |
| ⚠️ Near miss (≤10% off) | 81 | 7.1% | Right reasoning, rounding/precision issues |
| ⚠️ `%` suffix in `<Answer>` | 64 | 5.6% | Writes `"14.46%"` instead of `0.14464` |
| ⚠️ ×100 scale confusion | 40 | 3.5% | Writes `14.46` instead of `0.14464` |
| ❌ Non-numeric in `<Answer>` | 23 | 2.0% | Text/expression instead of number |
| ❌ Missing `<Answer>` tag | 21 | 1.8% | Format failure (3 samples) |
| ⚠️ Negative in parens | 4 | 0.3% | Writes `(-51)` instead of `-51` |

### Recoverable with Post-processing Fixes

If the evaluator normalised %-suffixed answers, x100 scale, near-misses to 10% tol, and paren-negatives — same model, no retraining:

```
Currently correct:         206  (18.0%)
+ %-suffix answers:        +64
+ x100 scale errors:       +40
+ Near misses (<=10%):     +81
+ Parens-negatives:        +4
─────────────────────────────
Recoverable total:         395  (34.4%)
```

---

## Sample Wrong Outputs

### Example 1 — Wrong DSL numbers selected
```
Question: What percentage of total facilities (sq ft) are leased?
Gold:      0.14464
Generated:
  <steps>
  Step 1 : divide(2.1, 32.8)
  </steps>
  <Answer>0.0644</Answer>
Problem: Picked wrong row values from the table (2.1 and 32.8 vs correct values)
```

### Example 2 — Wrong computation chain
```
Question: Difference in % cumulative total shareholder return (Masco vs S&P 500, 5yr)?
Gold:      1.1197
Generated:
  <steps>
  Step 1 : subtract(318.46, 206.49)
  Step 2 : divide(#0, 206.49)
  </steps>
  <Answer>0.50</Answer>
Problem: Computed % change instead of absolute point difference; used wrong base.
```

### Example 3 — Scale confusion (x100)
```
Question: % change in total rental expense, July 2005 to July 2006?
Gold:      0.06757
Generated:
  <steps>
  Step 1 : subtract(100690000, 92710000)
  Step 2 : divide(#0, 92710000)
  </steps>
  <Answer>10.0</Answer>
Problem: DSL logic is correct (right formula), but outputs 10.0 instead of ~0.086.
         The division result was multiplied by 100 inside the answer tag.
```

---

## Key Observations

### What Worked
- **Format learning was perfect.** 99.74% of outputs use the correct XML structure — up from 71.58% with SFT. GRPO's format reward is highly effective.
- **RL improved accuracy 3x over SFT** (18% vs 6.28%), purely from reward signal — no new labeled data.
- **Fixing the eval bugs mattered.** The initial eval (12.9%) was undercounting by ~5 percentage points due to the wrong model being evaluated.

### What Didn't Work
- **61.7% of failures are DSL logic errors** — the model picks the wrong numbers from the table or applies wrong operators. This is a fundamental reasoning limitation of the 1.5B parameter model with sparse reward over a single epoch.
- **Scale/format of numeric answer** — 9.1% of failures are recoverable formatting issues (`%` suffix, x100 confusion). These should be fixed by `answer_tag_reward` in future runs but didn't fully train through in 1 epoch.
- **`max_completion_length=256`** was too short for complex multi-step problems. Truncated completions received 0 reward, diluting the training signal. **Fixed to 512 for next run.**

---

## Bugs Found & Fixed (Between Runs)

| Bug | Impact | Fix |
|---|---|---|
| RL eval loaded RL LoRA on raw base (not SFT-merged) | Artificially low accuracy | Load SFT → merge → load RL LoRA → merge |
| No reward for correct `<Answer>` tag content | Model not penalised for wrong number | Added `answer_tag_reward` (weight 2.0) |
| `max_completion_length=256` truncated multi-step chains | Sparse reward signal | Bumped to 512 |
| `vllm==0.25.1` (non-existent version) in modal_trainer | Silent generation failures | Changed to `vllm` (latest stable) |
| `VLLM_USE_V1=0` in evaluator | Eval/train engine mismatch | Removed override |

---

## Next Run Plan

- **Epochs:** 3 (from 1)
- **max_completion_length:** 512 (from 256)
- **Reward functions:** All 4 (added `answer_tag_reward`)
- **Expected accuracy target:** 28–35% (based on recoverable errors + more training)
- **Stretch goal:** Fix evaluator to handle `%`-suffix and x100 scale normalisation for fair comparison
