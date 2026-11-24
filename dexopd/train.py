"""Train one student with frozen teachers and reward-scored student candidates."""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import sys
import time

import torch

from .objective import endpoint_direction, regression_loss, transition, transition_targets
from .prompt_encoding import encode_prompt
from .weights import ResidentHistory, evaluation_average
from .recipe import (
    RESOLUTION, STEPS, STUDENT_CFG, SUPERVISED_STEPS, TASKS, TEACHER_CFG,
    Settings, load_prompts, task_batch,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path(__file__).with_name("settings.json"))
    parser.add_argument("--small-batch", action="store_true")
    parser.add_argument("--base-model")
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--assets", type=Path, default=Path(os.environ.get("ASSET_ROOT", "assets")))
    parser.add_argument("--output", type=Path, default=Path("runs/train"))
    parser.add_argument("--ocr-python", default=os.environ.get("OCR_PYTHON", sys.executable))
    parser.add_argument("--geneval-python", default=os.environ.get("GENEVAL_PYTHON", sys.executable))
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


@dataclass
class Batch:
    latent: torch.Tensor
    teacher_target: torch.Tensor
    auxiliary_target: torch.Tensor
    active: torch.Tensor
    prompt: torch.Tensor
    pooled: torch.Tensor

    @classmethod
    def join(cls, batches):
        return cls(*(torch.cat([getattr(batch, field.name) for batch in batches])
                     for field in fields(cls)))

    def select(self, start, stop):
        return type(self)(*(getattr(self, field.name)[start:stop] for field in fields(self)))

    def shuffled(self):
        order = torch.randperm(len(self.latent), device=self.latent.device)
        return type(self)(*(getattr(self, field.name)[order] for field in fields(self)))


@contextmanager
def use_weights(parameters, values):
    with torch.no_grad():
        original = [param.detach().clone() for param in parameters]
        try:
            for param, value in zip(parameters, values, strict=True):
                param.copy_(value)
            torch.clear_autocast_cache()
            yield
        finally:
            for param, value in zip(parameters, original, strict=True):
                param.copy_(value)
            torch.clear_autocast_cache()


def guidance(prediction, scale, *, fp32=False):
    if fp32:
        prediction = prediction.float()
    if scale > 1:
        uncond, cond = prediction.chunk(2)
        prediction = uncond + scale * (cond - uncond)
    return prediction.float()


class Rollouts:
    def __init__(self, pipeline, model, accelerator, parameters, settings, scorers,
                 resident_history=None):
        self.pipeline = pipeline
        self.student = accelerator.unwrap_model(model)
        self.device = accelerator.device
        self.parameters = parameters
        self.settings = settings
        self.scorers = scorers
        self.resident_history = resident_history
        self.encode_prompt = encode_prompt
        self.timesteps = pipeline.scheduler.timesteps
        self.sigmas = pipeline.scheduler.sigmas
        self.delta = self.sigmas[1:] - self.sigmas[:-1]
        self.negative, self.negative_pooled = self.encode([""])

    def encode(self, prompts):
        encoders = [self.pipeline.text_encoder, self.pipeline.text_encoder_2,
                    self.pipeline.text_encoder_3]
        tokenizers = [self.pipeline.tokenizer, self.pipeline.tokenizer_2,
                      self.pipeline.tokenizer_3]
        with torch.no_grad():
            prompt, pooled = self.encode_prompt(encoders, tokenizers, prompts, 128)
        return prompt.to(self.device), pooled.to(self.device)

    def predict(self, model, latent, index, prompt, pooled, scale):
        if scale > 1:
            n = len(latent)
            latent = torch.cat((latent, latent))
            prompt = torch.cat((self.negative.expand(n, -1, -1), prompt))
            pooled = torch.cat((self.negative_pooled.expand(n, -1), pooled))
        return model(
            hidden_states=latent, timestep=self.timesteps[index].expand(len(latent)),
            encoder_hidden_states=prompt, pooled_projections=pooled, return_dict=False,
        )[0]

    def native(self, initial, prompt, pooled):
        latents, velocities = [initial], []
        for index in range(STEPS):
            prediction = self.predict(self.student, latents[-1], index, prompt, pooled, STUDENT_CFG)
            velocities.append(guidance(prediction, STUDENT_CFG, fp32=True))
            latent = transition(latents[-1], guidance(prediction, STUDENT_CFG), self.delta[index])
            torch.randn_like(latent)
            latents.append(latent)
        return latents, velocities

    def candidates(self, initial, prompt, pooled, history):
        n = len(initial)
        gains = self.settings.student_gains[1:]
        latent = initial.repeat((len(gains), 1, 1, 1))
        prompt = prompt.repeat((len(gains), 1, 1))
        pooled = pooled.repeat((len(gains), 1))
        # Candidate-major rows within each unconditional and conditional block.
        gain = torch.tensor(gains, device=self.device).repeat_interleave(n).repeat(2)
        gain = gain.reshape(-1, 1, 1, 1)
        for index in range(STEPS):
            current = self.predict(self.student, latent, index, prompt, pooled, STUDENT_CFG)
            context = (self.resident_history.activate() if self.resident_history is not None
                       else use_weights(self.parameters, history))
            with context:
                reference = self.predict(self.student, latent, index, prompt, pooled, STUDENT_CFG)
            velocity = (current.float() + gain * (current.float() - reference.float())).to(current.dtype)
            latent = transition(latent, guidance(velocity, STUDENT_CFG), self.delta[index])
        return list(latent.split(n))

    def score(self, task, endpoints, prompts, metadata):
        vae = self.pipeline.vae
        latent = endpoints.flatten(0, 1) / vae.config.scaling_factor + vae.config.shift_factor
        decoded = vae.decode(latent.to(vae.dtype), return_dict=False)[0]
        if not torch.isfinite(decoded).all():
            raise FloatingPointError(f"Nonfinite {task} VAE decode.")
        normalized = self.pipeline.image_processor.postprocess(decoded, output_type="np")
        pixels = (normalized * 255).round().astype("uint8")
        repeats = endpoints.shape[1]
        scores = self.scorers[task].score(
            pixels, [p for p in prompts for _ in range(repeats)],
            [m for m in metadata for _ in range(repeats)],
        )
        scores = torch.tensor(scores, device=self.device, dtype=torch.float64).reshape(endpoints.shape[:2])
        if not torch.isfinite(scores).all():
            raise FloatingPointError(f"Nonfinite {task} reward.")
        return scores

    @torch.no_grad()
    def collect(self, task, rows, history):
        prompts, metadata = map(list, zip(*rows))
        prompt, pooled = self.encode(prompts)
        initial = self.pipeline.prepare_latents(
            len(rows), self.student.config.in_channels, RESOLUTION, RESOLUTION,
            prompt.dtype, self.device, None, None,
        ).float()
        native, old_velocities = self.native(initial, prompt, pooled)
        endpoints = torch.stack([native[-1], *self.candidates(initial, prompt, pooled, history)], dim=1)
        direction, active = endpoint_direction(endpoints, self.score(task, endpoints, prompts, metadata))
        scale = TEACHER_CFG[task]
        self.student.set_adapter(task)
        try:
            experts = [
                guidance(self.predict(self.student, native[i], i, prompt, pooled, scale), scale)
                for i in range(SUPERVISED_STEPS)
            ]
        finally:
            self.student.set_adapter("default")
        with self.student.disable_adapter():
            bases = [
                guidance(self.predict(self.student, native[i], i, prompt, pooled, scale), scale, fp32=True)
                for i in range(SUPERVISED_STEPS)
            ]
        teacher, auxiliary = [], []
        for index, (expert, base) in enumerate(zip(experts, bases, strict=True)):
            teacher_target, auxiliary_target = transition_targets(
                native[index], self.delta[index], old_velocities[index], expert, base, direction,
                self.settings.teacher_extrapolation,
            )
            teacher.append(teacher_target)
            auxiliary.append(auxiliary_target)
        return Batch(
            torch.stack(native[:SUPERVISED_STEPS], dim=1), torch.stack(teacher, dim=1),
            torch.stack(auxiliary, dim=1), active, prompt, pooled,
        )


def update(model, optimizer, accelerator, rollouts, batch, parameters, microbatch):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for start in range(0, len(batch.latent), microbatch):
        group = batch.select(start, start + microbatch)
        for index in range(SUPERVISED_STEPS):
            with accelerator.accumulate(model):
                latent = group.latent[:, index]
                velocity = rollouts.predict(model, latent, index, group.prompt, group.pooled, STUDENT_CFG)
                prediction = transition(latent, guidance(velocity, STUDENT_CFG), rollouts.delta[index])
                loss = regression_loss(
                    prediction, group.teacher_target[:, index], group.auxiliary_target[:, index],
                    group.active, rollouts.sigmas[index], rollouts.settings.auxiliary_weight,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(parameters, 1.0)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("The update has a nonfinite gradient norm.")
                    optimizer.step()
            losses.append(loss.detach())
    if not accelerator.sync_gradients or accelerator.optimizer_step_was_skipped:
        raise RuntimeError("The optimizer did not complete its scheduled update.")
    return torch.stack(losses).mean(), grad_norm.detach()


def resolve_teachers(cfg, accelerator):
    if cfg.teacher_repo:
        from accelerate.utils import broadcast_object_list
        from huggingface_hub import snapshot_download

        snapshot = [None]
        if accelerator.is_main_process:
            snapshot[0] = snapshot_download(
                repo_id=cfg.teacher_repo,
                allow_patterns=[f"{subfolder}/*" for subfolder in cfg.teachers.values()],
            )
        broadcast_object_list(snapshot)
        teachers = {task: str(Path(snapshot[0]) / subfolder)
                    for task, subfolder in cfg.teachers.items()}
    else:
        teachers = cfg.teachers

    for task, path in teachers.items():
        if not Path(path).is_dir():
            raise FileNotFoundError(f"{task} teacher adapter directory: {path}")
    return teachers


def load_teacher(model, path, task):
    from peft import LoraConfig

    options = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    supported = {field.name for field in fields(LoraConfig)}
    # The public adapters record defaults for features added after PEFT 0.10.
    newer_defaults = {
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "lora_bias": False,
        "qalora_group_size": 16,
        "target_parameters": None,
        "trainable_token_indices": None,
        "use_qalora": False,
    }
    for name, default in newer_defaults.items():
        if name not in supported and name in options and options[name] == default:
            del options[name]
    options["inference_mode"] = True
    model.add_adapter(task, LoraConfig(**options))
    model.load_adapter(path, adapter_name=task, is_trainable=False)


def train(args, cfg, upstream):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import StableDiffusion3Pipeline
    from peft import LoraConfig, get_peft_model
    from .checkpoint import load_checkpoint, save_checkpoint
    from .score_client import Scorer
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rows = load_prompts(upstream)
    cases_per_rank = len(TASKS) * cfg.cases_per_task_per_rank
    microbatch = min(cases_per_rank, 3)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=SUPERVISED_STEPS * (cases_per_rank // microbatch),
    )
    if accelerator.num_processes != 8:
        raise RuntimeError("Launch this recipe with 8 processes to match the 72/24 global cases-per-update budget.")
    if accelerator.device.type != "cuda":
        raise RuntimeError("Training requires a CUDA device on every rank.")
    set_seed(cfg.seed, device_specific=True)
    teachers = resolve_teachers(cfg, accelerator)
    pipeline = StableDiffusion3Pipeline.from_pretrained(cfg.base_model)
    for encoder in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        encoder.to(accelerator.device, dtype=torch.bfloat16).requires_grad_(False).eval()
    vae_dtype = torch.float16 if args.small_batch else torch.float32
    pipeline.vae.to(accelerator.device, dtype=vae_dtype).requires_grad_(False).eval()
    pipeline.scheduler.set_timesteps(STEPS, device=accelerator.device)
    raw = get_peft_model(pipeline.transformer.to(accelerator.device, dtype=torch.float32), LoraConfig(
        r=32, lora_alpha=64, init_lora_weights="gaussian",
        target_modules=["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
                        "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"],
    ))
    for task in TASKS:
        load_teacher(raw, teachers[task], task)
    raw.set_adapter("default")
    parameters = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    resident_history = ResidentHistory(raw)
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8,
    )
    model, optimizer = accelerator.prepare(raw, optimizer)
    resident_history.synchronize()
    history = resident_history.values
    ema = [parameter.detach().clone() for parameter in parameters]
    budget = cfg.budget(accelerator.num_processes)
    global_step = 0
    if args.resume:
        global_step = load_checkpoint(args.resume, accelerator, raw, optimizer, history, ema, cfg)
    accelerator.print(json.dumps(budget, sort_keys=True))

    with ExitStack() as stack:
        scorers = {
            task: stack.enter_context(Scorer(
                task, upstream, args.assets, accelerator.device,
                {"ocr": args.ocr_python, "geneval": args.geneval_python}.get(task, sys.executable),
            ))
            for task in TASKS
        }
        rollouts = Rollouts(pipeline, model, accelerator, parameters, cfg, scorers, resident_history)
        for round_index in range(global_step // cfg.updates_per_round, budget["rounds"]):
            accelerator.wait_for_everyone()
            torch.cuda.synchronize(accelerator.device)
            started = time.perf_counter()
            raw.eval()
            records = []
            for task_index, task in enumerate(TASKS):
                local = [
                    task_batch(
                        rows[task], seed=cfg.seed, round_index=round_index, slot=slot,
                        task_index=task_index, cfg=cfg, world_size=accelerator.num_processes,
                        rank=accelerator.process_index,
                    )
                    for slot in range(cfg.updates_per_round)
                ]
                if args.small_batch:
                    records.append(rollouts.collect(task, [row for slot in local for row in slot], history))
                else:
                    records.append(Batch.join([rollouts.collect(task, slot, history) for slot in local]))
            if not args.small_batch:
                round_batch = Batch.join(records).shuffled()
            raw.train()
            losses = []
            grad_norms = []
            for slot in range(cfg.updates_per_round):
                start = slot * cfg.cases_per_task_per_rank
                stop = start + cfg.cases_per_task_per_rank
                if args.small_batch:
                    batch = Batch.join([record.select(start, stop) for record in records])
                else:
                    batch = round_batch.select(slot * cases_per_rank, (slot + 1) * cases_per_rank)
                evaluation_average(ema, parameters, global_step, calls=2)
                loss, grad_norm = update(model, optimizer, accelerator, rollouts, batch, parameters, microbatch)
                losses.append(loss)
                grad_norms.append(grad_norm)
                global_step += 1
                evaluation_average(ema, parameters, global_step)
            with torch.no_grad():
                for reference, parameter in zip(history, parameters, strict=True):
                    reference.mul_(cfg.history_decay).add_(parameter, alpha=1 - cfg.history_decay)
            torch.clear_autocast_cache()
            torch.cuda.synchronize(accelerator.device)
            elapsed = torch.tensor(time.perf_counter() - started, device=accelerator.device, dtype=torch.float64)
            elapsed = accelerator.gather(elapsed.reshape(1)).max().item()
            loss = accelerator.reduce(torch.stack(losses).mean(), reduction="mean").item()
            grad_norm = accelerator.gather(torch.stack(grad_norms).max().reshape(1)).max().item()
            accelerator.print(json.dumps({
                "round": round_index + 1, "updates": global_step,
                "loss": loss, "max_grad_norm": grad_norm,
                "round_seconds": elapsed, "seconds_per_update": elapsed / cfg.updates_per_round,
            }), flush=True)
            if global_step % cfg.save_every == 0 or global_step == cfg.updates:
                accelerator.wait_for_everyone()
                torch.cuda.synchronize(accelerator.device)
                save_started = time.perf_counter()
                save_checkpoint(args.output, accelerator, raw, optimizer, parameters, history, ema,
                                global_step, cfg, use_weights)
                torch.cuda.synchronize(accelerator.device)
                save_seconds = torch.tensor(time.perf_counter() - save_started, device=accelerator.device, dtype=torch.float64)
                save_seconds = accelerator.gather(save_seconds.reshape(1)).max().item()
                accelerator.print(json.dumps({"checkpoint_updates": global_step,
                                               "checkpoint_seconds": save_seconds}), flush=True)


def main():
    args = arguments()
    cfg = Settings.load(args.settings, args.updates, small_batch=args.small_batch,
                        base_model=args.base_model, teacher_root=args.teacher_root)
    train(args, cfg, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
