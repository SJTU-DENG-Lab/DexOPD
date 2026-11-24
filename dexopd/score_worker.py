"""Score candidate images with the public Aesthetic, OCR, and GenEval models."""
import json
import math
import os
from pathlib import Path
import sys

BATCH_SIZE = {"aesthetic": 5, "ocr": 5, "geneval": 14}


def scorer(task, assets, device):
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if task != "ocr":
        torch.cuda.set_device(device)

    if task == "aesthetic":
        os.environ.setdefault("REWARD_CKPTS", str(assets / "reward_ckpts"))
        from scorers.aesthetic import AestheticScorer

        model = AestheticScorer(dtype=torch.float32, device=device)
        return lambda pixels, prompts, metadata: model(
            torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous(),
        ).cpu().tolist()

    if task == "ocr":
        from scorers.ocr import load_ocr

        model = load_ocr(root=os.environ.get("OCR_ROOT") or (assets / "ocr"))
        return lambda pixels, prompts, metadata: model(pixels, prompts)

    from PIL import Image

    os.environ.setdefault("REWARD_CKPTS", str(assets / "reward_ckpts"))
    os.environ.setdefault("GENEVAL_CONFIG", str(
        assets / "geneval" / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    ))
    os.environ.setdefault("GENEVAL_CLIP_CACHE", str(assets / "geneval"))
    from scorers.geneval import load_geneval

    model = load_geneval(device)
    return lambda pixels, prompts, metadata: model(
        [Image.fromarray(pixel) for pixel in pixels], metadata, only_strict=False,
    )[0]


def run():
    import numpy as np
    import torch

    task, upstream, assets, device = sys.argv[1:]
    sys.path.insert(0, upstream)
    # Keep the protocol stream separate from third-party stdout, including native libraries.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(response):
        output.write(json.dumps(response, allow_nan=False).encode("utf-8") + b"\n")

    try:
        model = scorer(task, Path(assets), device)
        send({"ready": True})
        stream = sys.stdin.buffer
        for line in stream:
            request = json.loads(line)
            count = math.prod(request["shape"])
            raw = stream.read(count)
            if len(raw) != count:
                raise EOFError("Incomplete score request.")
            pixels = np.frombuffer(raw, dtype=np.uint8).reshape(request["shape"]).copy()
            scores = []
            with torch.no_grad():
                for i in range(0, len(pixels), BATCH_SIZE[task]):
                    batch = slice(i, i + BATCH_SIZE[task])
                    scores.extend(model(pixels[batch], request["prompts"][batch], request["metadata"][batch]))
            send({"scores": [float(value) for value in scores]})
    except Exception as error:
        send({"error": str(error)})
        raise
    finally:
        output.close()


if __name__ == "__main__":
    run()
