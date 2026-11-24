---
base_model: stabilityai/stable-diffusion-3.5-medium
base_model_relation: adapter
library_name: peft
pipeline_tag: text-to-image
language:
- en
tags:
- diffusers
- lora
- text-to-image
- stable-diffusion-3
- on-policy-distillation
---

# DexOPD for Stable Diffusion 3.5 Medium

**DexOPD** (*Dual-Extrapolation On-Policy Distillation*) consolidates three frozen Stable Diffusion 3.5 Medium specialist teachers—for **compositional alignment**, **text rendering**, and **visual quality**—into a single unified LoRA adapter. At inference time, the released student runs as a standard single-trajectory SD3.5 Medium pipeline with no auxiliary branches or extra sampling overhead.

**Authors:** Yunchao Ke\*, Yanzhe Hu\*, Yi Yang, Jiachun Jin, Zhihong Liu, Zhijie Deng†  
Shanghai Jiao Tong University · Nankai University

[Project page & interactive charts](https://sjtu-deng-lab.github.io/DexOPD/) · [Qualitative comparisons](https://sjtu-deng-lab.github.io/DexOPD/examples.html) · [Paper PDF](https://sjtu-deng-lab.github.io/DexOPD/assets/dexopd-paper.pdf) · [GitHub code](https://github.com/SJTU-DENG-Lab/DexOPD)

![DexOPD overview and multi-task benchmark summary](https://raw.githubusercontent.com/SJTU-DENG-Lab/DexOPD/main/assets/teaser.png)

## Checkpoint specifications

| Field | Value |
| --- | --- |
| Base model | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| Adapter format | [PEFT LoRA (`adapter_model.safetensors`, ~75 MB)](https://huggingface.co/kyc001/DexOPD/blob/main/adapter_model.safetensors) attached to the SD3 transformer |
| LoRA config | [Rank 32, alpha 64 (`adapter_config.json`)](https://huggingface.co/kyc001/DexOPD/blob/main/adapter_config.json) on attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`, `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`) |
| Released step | Evaluation EMA after 1,000 optimizer updates (500 rounds) |
| Training recipe | 72 native cases / update (24 per task), teacher extrapolation $\lambda = 1.2$, auxiliary weight $\beta = 1.8$, candidate gains $\mathcal{G} = \{0, 0.05, 0.25, 0.5\}$, history EMA decay $\tau = 0.995$ |
| Training resolution | 512 × 512 (paper evaluation at 512 × 512; gallery examples generated at 1024 × 1024) |

## Quick start (CUDA inference)

1. Request and accept access to the gated base model [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).
2. Install the inference dependencies and authenticate with Hugging Face:

```bash
python -m pip install -r https://raw.githubusercontent.com/SJTU-DENG-Lab/DexOPD/main/requirements.txt
huggingface-cli login
```

3. Load the base pipeline and attach `kyc001/DexOPD` to `pipe.transformer`:

```python
import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
)
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, "kyc001/DexOPD")
pipe.to("cuda")
pipe.vae.to(dtype=torch.float32)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    image = pipe(
        'A shop window with a sign that reads "OPEN LATE"',
        height=512,
        width=512,
        num_inference_steps=40,
        guidance_scale=4.5,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

image.save("dexopd.png")
```

> **Tip:** Paper benchmarks use `512 × 512`, `num_inference_steps=40`, and `guidance_scale=4.5` with FP16 autocast and an FP32 VAE. You can also generate at `1024 × 1024` (used in the [interactive comparison gallery](https://sjtu-deng-lab.github.io/DexOPD/examples.html)).

## Method summary

During on-policy distillation, the student collects 10-step deterministic Euler trajectories and receives two complementary signals at each visited state:

1. **Teacher extrapolation ($\lambda = 1.2$):** For each task specialist $v_T$ and the frozen base model $v_B$, DexOPD constructs a strengthened proxy velocity $v_{\text{proxy}} = v_T + \lambda(v_T - v_B)$ and takes one Euler step to form the distillation target.
2. **Reward-guided student self-extrapolation ($\beta = 1.8$):** Using an evolving historical EMA $H_r$ of the student ($\tau = 0.995$), DexOPD integrates four candidate rollouts from shared initial noise under gains $\mathcal{G} = \{0, 0.05, 0.25, 0.5\}$. Whenever candidates earn distinct task rewards, their normalized endpoint–reward covariance $G = C / \text{RMS}(C)$ supplies an auxiliary target; tied groups automatically deactivate ($G = 0$).

![DexOPD training framework](https://raw.githubusercontent.com/SJTU-DENG-Lab/DexOPD/main/assets/method.png)

## Benchmark evaluation

All trained models share the SD3.5-Medium backbone and a 1,000-update budget, evaluated at **512 × 512** with **40 inference steps** and **CFG 4.5** (CFG 1.0 for GenEval Teacher, DiffusionNFT, and DiffusionART as in their protocols). Higher is better across all columns.

### Main comparison (Table 1)

| Model | GenEval Acc. | GenEval Reward | OCR | Aesthetic | PickScore | ImageReward | CLIPScore | HPSv2.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SD3.5-Medium (base) | 0.6336 | 0.6446 | 0.5486 | 5.3899 | 0.8614 | 0.8383 | 0.2895 | 0.2924 |
| GenEval Teacher | 0.9349 | 0.9534 | 0.6515 | 5.2635 | 0.8513 | 0.9277 | 0.2869 | 0.2673 |
| OCR Teacher | 0.6144 | 0.6617 | 0.9435 | 5.2605 | 0.8580 | 0.8972 | 0.2909 | 0.2732 |
| Aesthetic Teacher | 0.7066 | 0.4791 | 0.5881 | 6.2162 | 0.9233 | 1.5016 | 0.2960 | 0.3452 |
| GRPO-Guard (multi-task RL) | 0.6764 | 0.7202 | 0.7210 | 5.6812 | 0.8770 | 1.1531 | 0.2904 | 0.3091 |
| DiffusionNFT (multi-task RL) | 0.5817 | 0.7029 | 0.9434 | 5.3674 | 0.8412 | 0.6314 | 0.2787 | 0.2642 |
| DiffusionART (multi-task RL) | 0.4847 | 0.4427 | 0.9033 | 5.6673 | 0.8292 | 0.7193 | 0.2710 | 0.2653 |
| DiffusionOPD | 0.9373 | 0.9543 | 0.9371 | 6.1820 | 0.9237 | 1.5110 | **0.3006** | 0.3547 |
| DreOPD | 0.9495 | 0.9622 | 0.9354 | 6.2182 | 0.9237 | **1.5259** | 0.2986 | 0.3558 |
| STEP-OPD\* | 0.9440 | 0.9562 | 0.9378 | 6.2813 | **0.9248** | 1.5207 | 0.2985 | **0.3566** |
| **DexOPD (released)** | **0.9566** | **0.9736** | **0.9515** | **6.4875** | 0.9190 | 1.5127 | 0.2914 | 0.3484 |

\* Paper-based reproduction. On the three primary metrics, DexOPD leads the next-best result in Table 1 by **+0.0114** on GenEval reward, **+0.0080** on OCR, and **+0.2062** on DrawBench Aesthetic. Interactive charts, GenEval subcategory breakdowns, and sweep plots are available on the [project page](https://sjtu-deng-lab.github.io/DexOPD/#results).

### Component ablation (1,000 updates)

| Configuration | GenEval Reward | OCR | Aesthetic |
| --- | ---: | ---: | ---: |
| **DexOPD (default)** | **0.9736** | **0.9515** | **6.4875** |
| No teacher extrapolation ($\lambda = 0$) | 0.9515 | 0.9357 | 6.2930 |
| No student auxiliary ($\beta = 0$) | 0.9669 | 0.9408 | 6.4410 |
| Frozen-base reference ($H_r = B$) | 0.9633 | 0.9512 | 6.2548 |

## Intended use & limitations

- **Intended use:** Research on diffusion post-training, multi-teacher on-policy distillation, and text-to-image generation with improved compositional fidelity, English text rendering, and aesthetic preference on top of Stable Diffusion 3.5 Medium.
- **Limitations:** This adapter inherits the capabilities, biases, and visual bounds of SD3.5-Medium and the three specialist teachers it distills from. Text rendering is trained and evaluated on English prompts; non-English scripts, dense multi-paragraph typography, and very fine anatomical details remain challenging.

## License & access

- **Adapter weights:** Publicly downloadable from `kyc001/DexOPD`.
- **Base model:** Requires accepting the gated [Stability AI Community License](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/LICENSE.md) on Hugging Face.
- **Training & evaluation code:** Released under the [Apache 2.0 License](https://github.com/SJTU-DENG-Lab/DexOPD/blob/main/LICENSE).

## Citation

```bibtex
@misc{ke2026dexopd,
  title  = {DexOPD: Dual-Extrapolation On-Policy Distillation for Diffusion Models},
  author = {Yunchao Ke and Yanzhe Hu and Yi Yang and Jiachun Jin and Zhihong Liu and Zhijie Deng},
  year   = {2026}
}
```
