import torch
from torch.distributions import Bernoulli, Normal


def _continuous_center(mean, mode):
    if mode == "legacy":
        return mean[..., :6].float()
    if mode == "temporally_correlated":
        return torch.atanh(mean[..., :6].float().clamp(-0.999, 0.999))
    raise ValueError(f"未知连续动作探索模式：{mode}")


def _correlated_log_prob(latents, mean, sigma, rho):
    """固定协方差的8步Gaussian对数密度；时间相关、无额外可训练参数。"""
    if not 0.0 <= rho < 1.0:
        raise ValueError("动作块时间相关系数必须在[0,1)内")
    diff = latents - mean
    n = diff.shape[-2]
    var = sigma * sigma * (1.0 - rho)
    quad = (diff.square().sum(dim=-2) -
            rho / (1.0 - rho + n * rho) * diff.sum(dim=-2).square()) / var
    logdet = n * torch.log(torch.as_tensor(sigma * sigma, device=diff.device)) + (
        (n - 1) * torch.log(torch.as_tensor(1.0 - rho, device=diff.device)) +
        torch.log(torch.as_tensor(1.0 - rho + n * rho, device=diff.device))
    )
    return -0.5 * (quad + n * 1.8378770664093453 + logdet).sum(dim=-1)


def sample_action_chunk(mean, std=0.12, *, mode="legacy", temperature=1.0, rho=0.8):
    """连续六维使用tanh-Gaussian，夹爪使用Bernoulli。"""
    if mean.shape[-2:] != (8, 7) or std <= 0 or temperature <= 0:
        raise ValueError("GRPO需要8×7动作和正的探索标准差")
    sigma = std * temperature
    center = _continuous_center(mean, mode)
    if mode == "temporally_correlated":
        common = torch.randn_like(center[..., :1, :])
        independent = torch.randn_like(center)
        raw = center + sigma * (rho ** 0.5 * common + (1.0 - rho) ** 0.5 * independent)
    else:
        raw = Normal(center, sigma).sample()
    cont = raw.tanh()
    gripper_sharpness = 14.0 if mode == "temporally_correlated" else 5.0
    gripper = Bernoulli(logits=gripper_sharpness * (mean[..., 6].float() - 0.5) / temperature).sample()
    actions = torch.cat((cont, gripper.unsqueeze(-1)), dim=-1)
    return actions, chunk_log_prob(mean, actions, std, mode=mode, temperature=temperature, rho=rho)


def chunk_log_prob(mean, actions, std=0.12, *, mode="legacy", temperature=1.0, rho=0.8):
    if mean.shape != actions.shape or mean.shape[-2:] != (8, 7):
        raise ValueError("当前动作与过去采样动作维度必须完全一致")
    if std <= 0 or temperature <= 0:
        raise ValueError("动作探索标准差和温度必须是正数")
    center = _continuous_center(mean, mode)
    sigma = std * temperature
    bounded = actions[..., :6].float().clamp(-0.999999, 0.999999)
    raw = torch.atanh(bounded)
    if mode == "temporally_correlated":
        cont = _correlated_log_prob(raw, center, sigma, rho)
    else:
        cont = Normal(center, sigma).log_prob(raw).sum(dim=(-1, -2))
    jacobian = torch.log1p(-bounded.square()).sum(dim=(-1, -2))
    gripper_sharpness = 14.0 if mode == "temporally_correlated" else 5.0
    grip = Bernoulli(logits=gripper_sharpness * (mean[..., 6].float() - 0.5) / temperature).log_prob(
        actions[..., 6].float()
    )
    return cont - jacobian + grip.sum(dim=-1)


def grpo_clipped_loss(new_log_probs, old_log_probs, advantages, epsilon=0.2):
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != advantages.shape:
        raise ValueError("GRPO每个动作块必须有对应的旧策略概率和组内优势")
    ratio = torch.exp((new_log_probs - old_log_probs).clamp(-15, 15))
    raw = ratio * advantages
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages
    return -torch.minimum(raw, clipped).mean()
