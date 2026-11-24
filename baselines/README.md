# Baselines

All six baselines use SD3.5 Medium at 512 × 512, eight NVIDIA H20 GPUs,
FP16 mixed precision, ten rollout steps, and 1,000 optimizer updates.
Evaluation uses 40 denoising steps and the same seven metrics as DexOPD.

| Baseline | Training settings | Evaluation at update 1,000 | Official source |
| --- | --- | --- | --- |
| DiffusionOPD | 72 cases per update, one update per round, LR `3e-4`, CFG `4.5` | EMA, CFG `4.5` | [GitHub](https://github.com/ali-vilab/DiffusionOPD) · [arXiv](https://arxiv.org/abs/2605.15055) |
| DreOPD | 24 cases per task update, three task updates per round, LR `3e-4`, CFG `4.5` | EMA, CFG `4.5` | [GitHub](https://github.com/sleepy1231/DreOPD) · [arXiv](https://arxiv.org/abs/2608.09233) |
| STEP-OPD | 72 cases per update, ten supervised states, LR `3e-4`, CFG `4.5` | EMA, CFG `4.5` | [GitHub](https://github.com/LiangrunFlora/STEP-OPD) · [arXiv](https://arxiv.org/abs/2608.04887) |
| GRPO-Guard | 72 images per update, two updates per rollout window, stochastic Flow sampling with noise `0.7`, LR `1e-4`, CFG `4.5` | EMA, CFG `4.5` | [arXiv](https://arxiv.org/abs/2510.22319) · [DiffusionOPD RL](https://github.com/ali-vilab/DiffusionOPD) |
| Multi-task DiffusionNFT | 72 images per update, one update per deterministic DPM2 rollout, LR `3e-4`, CFG `1` | EMA, CFG `1` | [GitHub](https://github.com/NVlabs/DiffusionNFT) · [arXiv](https://arxiv.org/abs/2509.16117) |
| Multi-task DiffusionART | Six groups of 24 images per rollout, two disjoint 72-image updates, ten supervised states, LR `3e-4`, CFG `1` | Current policy, CFG `1` | [GitHub](https://github.com/vanzll/DiffusionART) · [arXiv](https://arxiv.org/abs/2610.12572) |

The distillation baselines use the frozen Aesthetic, OCR, and GenEval
adapters. The three reinforcement learning baselines balance GenEval, OCR,
and visual preference, using PickScore logit / 26, CLIPScore logit / 100,
and HPSv2.1 at equal weights for the visual reward. GRPO-Guard uses
per-sample ratio normalization and a clip range of `2e-6`. DiffusionART
disables EMA and reference KL.

Use [the evaluator](../README.md#evaluation) with the directory containing
the exported `lora/`. Set `EVAL_CFG=1` for DiffusionNFT and DiffusionART.
