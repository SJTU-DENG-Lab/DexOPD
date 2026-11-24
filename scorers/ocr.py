"""Score how closely detected text matches a prompt's quoted target."""

import os
from pathlib import Path


def edit_distance(recognized, target):
    previous = list(range(len(target) + 1))
    for i, actual in enumerate(recognized, start=1):
        current = [i]
        for j, expected in enumerate(target, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (actual != expected)))
        previous = current
    return previous[-1]


class OcrScorer:
    def __init__(self, backend):
        self.ocr = backend

    def __call__(self, images, prompts):
        targets = []
        for prompt in prompts:
            _, opening, quoted = prompt.partition('"')
            target, closing, _ = quoted.partition('"')
            target = target.replace(" ", "").lower()
            if not opening or not closing or not target.strip():
                raise ValueError(f"Expected non-empty quoted text in OCR prompt: {prompt!r}")
            targets.append(target)

        rewards = []
        for image, target in zip(images, targets, strict=True):
            if type(image).__module__.startswith("PIL."):
                import numpy as np

                image = np.array(image)
            result = self.ocr.ocr(image, cls=False)
            detections = result[0] if result else None
            text = "".join(value for _, (value, confidence) in detections or [] if confidence > 0)
            text = text.replace(" ", "").lower()
            distance = 0 if target in text else min(edit_distance(text, target), len(target))
            rewards.append(1.0 - distance / len(target))
        return rewards


def load_ocr(*, root=None, use_gpu=False):
    from paddleocr import PaddleOCR

    if root is None:
        root = os.environ.get("OCR_ROOT", "assets/ocr")
    root = Path(root)
    return OcrScorer(PaddleOCR(
        use_angle_cls=False, lang="en", use_gpu=use_gpu, show_log=False,
        det_model_dir=str(root / "det"),
        rec_model_dir=str(root / "rec"),
        cls_model_dir=str(root / "cls"),
        rec_char_dict_path=str(root / "en_dict.txt"),
    ))
