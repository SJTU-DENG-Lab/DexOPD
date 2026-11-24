"""Frozen transition targets and the regression objective."""
import torch


def endpoint_direction(endpoints, rewards):
    """Compute the unit-RMS reward covariance independently for each prompt."""
    if endpoints.ndim < 3 or rewards.shape != endpoints.shape[:2]:
        raise ValueError("Endpoints and rewards must share the case and gain axes.")
    x = endpoints.detach().float()
    r = rewards.detach().to(device=x.device, dtype=torch.float64)
    if not bool(torch.isfinite(x).all() and torch.isfinite(r).all()):
        raise FloatingPointError("Nonfinite endpoint or reward.")
    centered_x = x - x.mean(dim=1, keepdim=True)
    centered_r = (r - r.mean(dim=1, keepdim=True)).float()
    reward_shape = (*r.shape, *((1,) * (x.ndim - 2)))
    cov = (centered_x * centered_r.reshape(reward_shape)).mean(dim=1)
    rms = cov.flatten(1).square().mean(dim=1).sqrt()
    active = (r.amax(dim=1) != r.amin(dim=1)) & (rms > 1e-8)
    case_shape = (len(x), *((1,) * (cov.ndim - 1)))
    direction = cov / rms.clamp_min(1e-8).reshape(case_shape)
    return torch.where(active.reshape(case_shape), direction, 0.0), active


def transition(latent, velocity, step_size):
    return latent.float() + velocity.float() * step_size.float()


def transition_targets(latent, step_size, old_velocity, expert, base, direction, coefficient):
    teacher_velocity = expert.float() + coefficient * (expert.float() - base.float())
    auxiliary_velocity = old_velocity.float() - direction.float()
    return (
        transition(latent, teacher_velocity, step_size).detach(),
        transition(latent, auxiliary_velocity, step_size).detach(),
    )


def regression_loss(prediction, teacher_target, auxiliary_target, active, sigma, beta):
    teacher_delta = prediction.float() - teacher_target.detach()
    auxiliary_delta = prediction.float() - auxiliary_target.detach()
    teacher_loss = (0.5 * teacher_delta.square()).flatten(1).mean(dim=1)
    auxiliary_loss = auxiliary_delta.square().flatten(1).mean(dim=1)
    auxiliary_loss = auxiliary_loss * sigma * active
    return teacher_loss.mean() + beta * auxiliary_loss.mean()
