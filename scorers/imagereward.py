import os
import torch
import ImageReward as RM


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        cache_root = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache"))
        self.model = (
            RM.load(
                "ImageReward-v1.0",
                device=device,
                download_root=os.path.join(cache_root, "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts, images):
        _, rewards = self.model.inference_rank(prompts, images)
        matrix = torch.as_tensor(rewards, dtype=self.dtype, device=self.device).reshape(len(prompts), len(prompts))
        return torch.diagonal(matrix, 0).contiguous()
