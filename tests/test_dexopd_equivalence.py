"""CPU regression checks; run with python -m unittest discover -s tests -v."""
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import torch

from dexopd.checkpoint import load_checkpoint, save_checkpoint
from dexopd.recipe import Settings
from dexopd.train import Rollouts, load_teacher, resolve_teachers, use_weights
from dexopd.weights import ResidentHistory, evaluation_average


def small_student():
    torch.manual_seed(42)
    transformer = SD3Transformer2DModel(
        sample_size=4, patch_size=2, in_channels=4, num_layers=1,
        attention_head_dim=4, num_attention_heads=2, joint_attention_dim=8,
        caption_projection_dim=8, pooled_projection_dim=8,
        out_channels=4, pos_embed_max_size=4,
    )
    model = get_peft_model(transformer, LoraConfig(
        r=2, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
    ))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model


def small_rollout(model, bank):
    schedule = FlowMatchEulerDiscreteScheduler()
    schedule.set_timesteps(10, device="cpu")
    rollout = Rollouts.__new__(Rollouts)
    rollout.student = model
    rollout.device = torch.device("cpu")
    rollout.parameters = list(bank.source)
    rollout.resident_history = bank
    rollout.settings = SimpleNamespace(student_gains=[0., 0.05, 0.25, 0.5])
    rollout.timesteps, rollout.sigmas = schedule.timesteps, schedule.sigmas
    rollout.delta = schedule.sigmas[1:] - schedule.sigmas[:-1]
    rollout.negative = torch.randn(1, 3, 8)
    rollout.negative_pooled = torch.randn(1, 8)
    return rollout


class StateEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_teacher_config_loads_with_pinned_peft(self):
        model = small_student()
        expected = get_peft_model_state_dict(model)
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            adapter_config = Path(directory) / "adapter_config.json"
            options = json.loads(adapter_config.read_text(encoding="utf-8"))
            options.update(lora_bias=False, use_qalora=False, qalora_group_size=16,
                           corda_config=None, eva_config=None, exclude_modules=None,
                           target_parameters=None, trainable_token_indices=None)
            adapter_config.write_text(json.dumps(options), encoding="utf-8")
            load_teacher(model, directory, "teacher")
        actual = get_peft_model_state_dict(model, adapter_name="teacher")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(model.peft_config["teacher"].inference_mode)

    def test_teacher_snapshot_is_shared_across_ranks(self):
        settings_path = Path(__file__).parents[1] / "dexopd/settings.json"
        hub = Settings.load(settings_path)
        with TemporaryDirectory() as directory:
            local = Settings.load(settings_path, teacher_root=directory)
            for path in local.teachers.values():
                Path(path).mkdir(parents=True)
            with patch("huggingface_hub.snapshot_download", return_value=directory) as download:
                main = SimpleNamespace(is_main_process=True)
                worker = SimpleNamespace(is_main_process=False)

                def receive_snapshot(value):
                    value[0] = directory

                with patch("accelerate.utils.broadcast_object_list") as broadcast:
                    self.assertEqual(resolve_teachers(hub, main), local.teachers)
                    broadcast.assert_called_once_with([directory])
                    broadcast.side_effect = receive_snapshot
                    self.assertEqual(resolve_teachers(hub, worker), local.teachers)
                self.assertEqual(resolve_teachers(local, worker), local.teachers)
                download.assert_called_once_with(
                    repo_id=hub.teacher_repo,
                    allow_patterns=[f"{subfolder}/*" for subfolder in hub.teachers.values()],
                )

    def test_resident_candidates_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = small_student().eval()
                parameters = [p for p in model.parameters() if p.requires_grad]
                rng = torch.get_rng_state().clone()
                bank = ResidentHistory(model)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual([id(p) for p in parameters], [id(p) for p in bank.source])
                with torch.no_grad():
                    for history, current in zip(bank.values, bank.source):
                        history.copy_(current * 0.6 + 0.02)
                reference = [p.detach().clone() for p in bank.values]
                rollout = small_rollout(model, bank)
                latent, prompt, pooled = torch.randn(2, 4, 4, 4), torch.randn(2, 3, 8), torch.randn(2, 8)
                before = [p.detach().clone() for p in parameters]
                versions = [p._version for p in parameters]
                context = lambda: (torch.autocast("cpu", dtype=dtype)
                                   if dtype == torch.bfloat16 else nullcontext())
                with torch.no_grad(), context():
                    resident = rollout.candidates(latent, prompt, pooled, bank.values)
                self.assertEqual(versions, [p._version for p in parameters])
                rollout.resident_history = None
                with torch.no_grad(), context():
                    copied = rollout.candidates(latent, prompt, pooled, reference)
                self.assertEqual([v + 20 for v in versions], [p._version for p in parameters])
                for actual, expected in zip(resident, copied):
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                for actual, expected in zip(parameters, before):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertFalse(any(p.requires_grad or p.grad is not None for p in bank.values))
                with self.assertRaisesRegex(RuntimeError, "forward failure"):
                    with bank.activate():
                        raise RuntimeError("forward failure")
                self.assertEqual(model.active_adapters, ["default"])
                self.assertTrue(all(p.requires_grad for p in parameters))
                # A later training backward must still reach precisely the student.
                optimizer = torch.optim.AdamW(parameters, lr=3e-4)
                with context():
                    prediction = rollout.predict(model, latent, 0, prompt, pooled, 4.5)
                prediction.float().square().mean().backward()
                gradients = [p.grad.clone() for p in parameters]
                optimizer.zero_grad(set_to_none=True)
                with bank.activate(), context():
                    rollout.predict(model, latent, 0, prompt, pooled, 4.5)
                with context():
                    prediction = rollout.predict(model, latent, 0, prompt, pooled, 4.5)
                prediction.float().square().mean().backward()
                for p, expected in zip(parameters, gradients):
                    torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
                optimizer.step()
                self.assertFalse(any(p.grad is not None for p in bank.values))

    def test_evaluation_ema_microbatch_cadence(self):
        parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        packed, reference = [parameter.detach().clone()], [parameter.detach().clone()]
        for completed in range(1000):
            evaluation_average(packed, [parameter], completed, calls=2)
            for microbatch in range(3):
                step = completed
                if microbatch == 2:
                    with torch.no_grad():
                        parameter.add_(torch.tensor([0.003, -0.002]))
                    step += 1
                if (step + 1) % 8 == 0:
                    decay = min((1 + step) / (10 + step), 0.9)
                    reference[0].add_((1 - decay) * (parameter.detach() - reference[0]))
            evaluation_average(packed, [parameter], completed + 1)
            torch.testing.assert_close(packed[0], reference[0], rtol=0, atol=0)

    def test_checkpoint_history_roundtrip(self):
        accelerator = Accelerator(cpu=True)
        model = small_student()
        bank = ResidentHistory(model)
        optimizer = torch.optim.AdamW(bank.source, lr=3e-4)
        ema = [p.detach().clone() for p in bank.source]
        with torch.no_grad():
            for value in bank.values:
                value.add_(0.25)
        expected = [p.detach().clone() for p in bank.values]
        cfg = Settings.load(Path(__file__).parents[1] / "dexopd/settings.json")
        with TemporaryDirectory() as output, \
                patch("torch.cuda.get_rng_state", return_value=torch.get_rng_state()), \
                patch("torch.cuda.set_rng_state"):
            save_checkpoint(output, accelerator, model, optimizer, bank.source,
                            bank.values, ema, 2, cfg, use_weights)
            checkpoint = Path(output) / "checkpoints/checkpoint-2"
            self.assertFalse((checkpoint / "training_adapter/history").exists())
            with torch.no_grad():
                for value in bank.values:
                    value.zero_()
            step = load_checkpoint(checkpoint, accelerator, model, optimizer, bank.values, ema, cfg)
            self.assertEqual(step, 2)
            for actual, saved in zip(bank.values, expected):
                torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            self.assertFalse(any(p.requires_grad for p in bank.values))


if __name__ == "__main__":
    unittest.main()
