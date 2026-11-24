# Third-Party Notices

DexOPD code is distributed under the [Apache License 2.0](LICENSE).

| Project | Use in DexOPD | Source |
| --- | --- | --- |
| DiffusionOPD | Scorer integrations, evaluation code, prompt splits, and frozen teacher LoRAs | [Apache-2.0 source](https://github.com/ali-vilab/DiffusionOPD/tree/0c85bf0193a6da24bb1f72152b612d353a154999) · [Teacher weights](https://huggingface.co/quanhaol/DiffusionOPD) |
| Flow-GRPO | SD3 prompt-encoding and reward utilities inherited through DiffusionOPD | [Apache-2.0 source](https://github.com/yifan123/flow_grpo) |
| Diffusers | SD3.5 pipeline, scheduler, and upstream SD3 prompt-encoding implementation | [Apache-2.0 source](https://github.com/huggingface/diffusers) |
| DiffusionNFT | Upstream evaluation implementation inherited through DiffusionOPD | [Apache-2.0 source](https://github.com/NVlabs/DiffusionNFT) |

The SD3.5 Medium base model and the DiffusionOPD LoRAs follow their
respective model-card terms. [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
· [DiffusionOPD adapters](https://huggingface.co/quanhaol/DiffusionOPD)

Scorer code, dependencies, and weights retain their upstream terms.
[GenEval](https://github.com/djghosh13/geneval)
· [MMDetection](https://github.com/open-mmlab/mmdetection)
· [OpenCLIP](https://github.com/mlfoundations/open_clip)
· [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
· [Improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
· [OpenAI CLIP](https://github.com/openai/CLIP)
· [PickScore](https://github.com/yuvalkirstain/PickScore)
· [HPSv2.1](https://github.com/tgxs002/HPSv2)
· [ImageReward](https://github.com/zai-org/ImageReward)

The training and evaluation prompt splits distributed by DiffusionOPD
retain their upstream dataset terms. [DiffusionOPD](https://github.com/ali-vilab/DiffusionOPD)
· [GenEval](https://github.com/djghosh13/geneval)
· [Flow-GRPO OCR prompts](https://github.com/yifan123/flow_grpo)
· [Pick-a-Pic](https://github.com/yuvalkirstain/PickScore)
· [DrawBench](https://imagen.research.google/)

## GenEval MIT License Notice

`scorers/geneval.py` incorporates evaluation logic adapted from [GenEval](https://github.com/djghosh13/geneval):

```text
MIT License

Copyright (c) 2023 Dhruba Ghosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
