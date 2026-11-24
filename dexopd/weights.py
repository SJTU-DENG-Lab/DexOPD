"""Resident frozen LoRA state and evaluation averaging."""
from contextlib import contextmanager
from copy import deepcopy

import torch


def adapter_parameters(model, name):
    marker = f".{name}."
    return {
        key.replace(marker, ".<adapter>."): value
        for key, value in model.named_parameters()
        if "lora_" in key and marker in key
    }


class ResidentHistory:
    """Share the backbone and keep a second, frozen LoRA on its device."""

    def __init__(self, model):
        self.model = model
        source = adapter_parameters(model, "default")
        if not source or model.active_adapters != ["default"]:
            raise ValueError("Initialize history with the student adapter active.")
        config = deepcopy(model.peft_config["default"])
        if config.bias != "none" or config.modules_to_save:
            raise ValueError("Resident history requires LoRA-only student parameters.")
        config.inference_mode = True
        devices = sorted({p.device.index for p in source.values() if p.device.type == "cuda"})
        # Adding an adapter initializes tensors. Do not consume the rollout RNG.
        with torch.random.fork_rng(devices=devices):
            model.add_adapter("history", config)
        model.set_adapter("default")
        reference = adapter_parameters(model, "history")
        if source.keys() != reference.keys():
            raise ValueError("Student and history adapter layouts differ.")
        self.source = list(source.values())
        self.values = [reference[key] for key in source]
        for value in self.values:
            value.requires_grad_(False)
        self.synchronize()

    @torch.no_grad()
    def synchronize(self):
        """Initialize after distributed student synchronization, before resume."""
        for reference, source in zip(self.values, self.source, strict=True):
            reference.copy_(source)
        torch.clear_autocast_cache()

    @contextmanager
    def activate(self):
        if self.model.active_adapters != ["default"]:
            raise RuntimeError("History queries must start from the student adapter.")
        # PEFT set_adapter also toggles requires_grad. Keep history frozen and
        # restore the exact student flags, including when a forward raises.
        flags = [value.requires_grad for value in self.source]
        try:
            self.model.set_adapter("history")
            for value in self.values:
                value.requires_grad_(False)
            with torch.no_grad():
                yield
        finally:
            self.model.set_adapter("default")
            for value, enabled in zip(self.source, flags, strict=True):
                value.requires_grad_(enabled)
            for value in self.values:
                value.requires_grad_(False)


@torch.no_grad()
def evaluation_average(ema, parameters, completed_updates, *, calls=1):
    """Update evaluation EMA when (completed_updates + 1) % 8 == 0."""
    if (completed_updates + 1) % 8 == 0:
        decay = min((1 + completed_updates) / (10 + completed_updates), 0.9)
        for _ in range(calls):
            for average, parameter in zip(ema, parameters, strict=True):
                average.add_((1 - decay) * (parameter - average))
