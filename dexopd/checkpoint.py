"""Save EMA LoRA for evaluation and enough state for an exact round restart."""
from dataclasses import asdict
import os
from pathlib import Path
import random

from accelerate.utils import broadcast_object_list
import numpy as np
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
import torch


def save_checkpoint(output, accelerator, model, optimizer, parameters, history, ema,
                    step, cfg, use_weights):
    root = Path(output) / "checkpoints"
    target = root / f"checkpoint-{step}"
    staging = root / f".checkpoint-{step}.incomplete"
    failure = [None]
    if accelerator.is_main_process:
        try:
            if target.exists():
                raise FileExistsError(target)
            staging.mkdir(parents=True)
        except OSError as error:
            failure[0] = str(error)
    broadcast_object_list(failure)
    if failure[0]:
        raise OSError(failure[0])
    np_algo, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    torch.save({
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(accelerator.device),
        "python": random.getstate(),
        "numpy": (np_algo, np_keys.tolist(), int(np_pos), int(np_has_gauss), float(np_cached)),
    }, staging / f"rng-{accelerator.process_index}.pt")
    if accelerator.is_main_process:
        model.save_pretrained(staging / "training_adapter", selected_adapters=["default"])
        with use_weights(parameters, ema):
            model.save_pretrained(staging / "lora", selected_adapters=["default"])
        torch.save({
            "format_version": 1,
            "step": step, "world_size": accelerator.num_processes, "settings": asdict(cfg),
            "optimizer": optimizer.state_dict(), "history": [value.cpu() for value in history],
            "ema": [value.cpu() for value in ema],
        }, staging / "state.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.rename(staging, target)
    accelerator.wait_for_everyone()


def load_checkpoint(path, accelerator, model, optimizer, history, ema, cfg):
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version; expected 1.")
    settings = asdict(cfg)
    stored = state["settings"]
    if (state["world_size"] != accelerator.num_processes
            or any(value != stored.get(key) for key, value in settings.items() if key != "updates")
            or state["step"] % cfg.updates_per_round
            or state["step"] >= cfg.updates):
        raise ValueError("Checkpoint and training recipe are incompatible.")
    adapter = load_peft_weights(str(path / "training_adapter"), device=str(accelerator.device))
    set_peft_model_state_dict(model, adapter, adapter_name="default")
    optimizer.load_state_dict(state["optimizer"])
    for current, saved in zip(history + ema, state["history"] + state["ema"], strict=True):
        current.copy_(saved)
    rng = torch.load(path / f"rng-{accelerator.process_index}.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"], accelerator.device)
    random.setstate(rng["python"])
    np_algo, np_keys, np_pos, np_has_gauss, np_cached = rng["numpy"]
    np.random.set_state((np_algo, np.asarray(np_keys, dtype=np.uint32), np_pos, np_has_gauss, np_cached))
    return state["step"]
