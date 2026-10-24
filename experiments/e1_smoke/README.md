# E1 SMOKE run — CPU pipeline proof (NOT E1 results)

**This is a SMOKE run**, the Phase 3 analogue of `experiments/e0_smoke/`:
it proves the QLoRA *training pipeline* works end to end on this CPU-only
VM. It is **not** experiment E1, and none of its numbers may be cited as
E1 results. Real E1 (`Qwen/Qwen2.5-7B-Instruct`, 4-bit QLoRA, the full
582-record filtered pool) runs on Kaggle — see
`notebooks/05_qlora_training.ipynb`.

## What was run

```
HF_HUB_DISABLE_XET=1 uv run python -m skillgen.training.train --config configs/smoke_cpu.yaml
```

Config (`configs/smoke_cpu.yaml`), deliberately different from E1:
model `Qwen/Qwen2.5-0.5B-Instruct`, **quantization disabled → plain LoRA,
not QLoRA** (bitsandbytes needs CUDA), LoRA r8/α16, `max_seq_length`
2048, **20 train records — the 20 shortest of the Phase-0 filtered
pool** (selection: `shortest`, to keep CPU time sane), 1 epoch,
per-device batch 2 × grad-accum 1 = **10 optimizer steps**, eval on the
first 8 filtered val records every 5 steps, `adamw_torch`, seed 42.

Wall time: **804 s** total on this VM (tokenization of the full
800-record train split for filtering + model load + 514 s of trainer
runtime including the periodic evals + final eval and save). Full log:
`training_log.txt` in this directory.

## Results (as printed)

- **§21 masking gate PASSED**: the decoded unmasked text of the first
  training example printed by `train.py` is ONLY the assistant blueprint
  + `<|im_end|>` — no system prompt, no user instruction (see
  `training_log.txt`). This is the check the unit tests in
  `tests/test_prepare.py` certify.
- Train loss per step: 2.4453 → 2.3149 → 1.9471 → 1.8453 → 1.7785 →
  1.6539 → 1.8261 → 1.7196 → 1.3023 → 1.8009 (mean train loss 1.8634).
- Val eval loss: **2.0292** (step 5) → **2.0031** (step 10, also the
  final `trainer.evaluate()`).
- Adapter saved to `adapter/` (git-ignored, regenerable):
  `adapter_model.safetensors` (**17,640,136 bytes**) +
  `adapter_config.json` (1,180 bytes) + tokenizer files; checkpoints at
  steps 5 and 10 with full trainer state.

## What this proves — and what it does not

Proves: `prepare.py` filtering + masking labels are the labels actually
trained on (TRL 1.16 leaves a provided `labels` column untouched);
`train.py`'s SFTTrainer wiring, PEFT LoRA, eval/save/best-checkpoint
plumbing and adapter saving all work; the loss moves down on real
blueprint data.

Does **not** prove: anything about the 7B model, QLoRA/4-bit quality,
the E1 hyperparameters, or generation quality after training. The
adapter was trained on 20 short examples of a 0.5B model for 10 steps —
it is a plumbing artifact, not a model. E1's evidence will come from the
Kaggle run and the Phase 4 harness scoring.
