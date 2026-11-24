"""Configuration and task-balanced data assignment."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

TASKS = ("aesthetic", "ocr", "geneval")
STEPS = 10
SUPERVISED_STEPS = 9
RESOLUTION = 512
STUDENT_CFG = 4.5
TEACHER_CFG = {"aesthetic": 4.5, "ocr": 4.5, "geneval": 1.0}
TRAIN_FILES = {
    "aesthetic": "pickscore/train.txt",
    "ocr": "ocr/train.txt",
    "geneval": "geneval/train_metadata.jsonl",
}


@dataclass(frozen=True)
class Settings:
    base_model: str
    teacher_repo: str
    teachers: dict
    teacher_extrapolation: float
    auxiliary_weight: float
    student_gains: list
    history_decay: float
    learning_rate: float
    cases_per_task_per_rank: int
    updates_per_round: int
    updates: int
    save_every: int
    seed: int

    @classmethod
    def load(cls, path, updates=None, *, small_batch=False, base_model=None, teacher_root=None):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if updates is not None:
            raw["updates"] = updates
        cfg = cls(**raw)
        if set(cfg.teachers) != set(TASKS):
            raise ValueError("Provide exactly one frozen teacher per task.")
        if small_batch:
            cfg = replace(cfg, cases_per_task_per_rank=1)
        if base_model:
            cfg = replace(cfg, base_model=base_model)
        if teacher_root is not None:
            teachers = {task: str(Path(teacher_root) / subfolder)
                        for task, subfolder in cfg.teachers.items()}
            cfg = replace(cfg, teacher_repo="", teachers=teachers)
        for key in ("cases_per_task_per_rank", "updates_per_round", "updates", "save_every"):
            if type(getattr(cfg, key)) is not int or getattr(cfg, key) < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if cfg.updates_per_round != 2:
            raise ValueError("This recipe uses two optimizer updates per round.")
        if cfg.updates % cfg.updates_per_round or cfg.save_every % cfg.updates_per_round:
            raise ValueError("Training and save budgets must end at a complete round.")
        if not cfg.student_gains or cfg.student_gains[0] != 0.0 or len(cfg.student_gains) < 2:
            raise ValueError("The candidate list must start with the reused zero-gain endpoint.")
        return cfg

    def budget(self, world_size):
        if world_size < 1:
            raise ValueError("World size must be positive.")
        per_task = world_size * self.cases_per_task_per_rank
        return {
            "ranks": world_size,
            "cases_per_task_per_update": per_task,
            "cases_per_update": per_task * len(TASKS),
            "cases_per_task_per_round": per_task * self.updates_per_round,
            "cases_per_round": per_task * len(TASKS) * self.updates_per_round,
            "updates_per_round": self.updates_per_round,
            "rounds": self.updates // self.updates_per_round,
            "updates": self.updates,
        }


def load_prompts(upstream):
    result = {}
    for task, relative_path in TRAIN_FILES.items():
        rows = (upstream / "dataset" / relative_path).read_text(encoding="utf-8").splitlines()
        if task == "geneval":
            records = [json.loads(row) for row in rows]
            result[task] = [(row["prompt"], row) for row in records]
        else:
            result[task] = [(row.strip(), {}) for row in rows]
    return result


def task_batch(rows, *, seed, round_index, slot, task_index, cfg, world_size, rank):
    import torch

    count = cfg.cases_per_task_per_rank * world_size
    if count > len(rows):
        raise ValueError("Dataset is smaller than one global task batch.")
    batch_index = (round_index * cfg.updates_per_round + slot) * len(TASKS) + task_index
    generator = torch.Generator().manual_seed(seed + batch_index)
    selected = torch.randperm(len(rows), generator=generator)[:count]
    selected = selected[torch.randperm(count, generator=generator)]
    start = rank * cfg.cases_per_task_per_rank
    return [rows[i] for i in selected[start:start + cfg.cases_per_task_per_rank].tolist()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path(__file__).with_name("settings.json"))
    parser.add_argument("--small-batch", action="store_true")
    parser.add_argument("--ranks", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(Settings.load(args.settings, small_batch=args.small_batch).budget(args.ranks), indent=2))
