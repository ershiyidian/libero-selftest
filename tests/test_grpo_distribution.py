import math

import torch
from torch.distributions import MultivariateNormal

from selftest.grpo_distribution import sample_action_chunk, chunk_log_prob


def test_temporal_noise_log_probability_matches_multivariate_gaussian():
    torch.manual_seed(31)
    mean = torch.randn(6, 8, 7).clamp(-0.8, 0.8)
    std, rho, temperature = 0.1, 0.7, 1.2
    actions, old_logp = sample_action_chunk(
        mean, std, mode="temporally_correlated", temperature=temperature, rho=rho
    )
    new_logp = chunk_log_prob(
        mean, actions, std, mode="temporally_correlated", temperature=temperature, rho=rho
    )
    torch.testing.assert_close(old_logp, new_logp, atol=1e-6, rtol=0)

    covariance = (std * temperature)**2 * (
        (1-rho)*torch.eye(8) + rho * torch.ones(8, 8)
    )
    continuous = torch.atanh(actions[..., :6].clamp(-0.999999, 0.999999))
    center = torch.atanh(mean[..., :6].clamp(-0.999, 0.999))
    gaussian = MultivariateNormal(center.transpose(1, 2), covariance_matrix=covariance)
    expected = (gaussian.log_prob(continuous.transpose(1, 2)).sum(dim=-1) -
                torch.log1p(-actions[..., :6].square()).sum(dim=(1, 2)))
    probabilities = torch.sigmoid(14*(mean[..., 6] - 0.5)/temperature)
    grip = torch.where(actions[..., 6] == 1, probabilities, 1-probabilities)
    expected = expected + grip.log().sum(dim=-1)
    torch.testing.assert_close(old_logp, expected, atol=1e-4, rtol=1e-4)


def test_temporal_actions_are_more_continuous_and_baseline_preserved():
    torch.manual_seed(20261011)
    mean = torch.full((1600, 8, 7), 0.25)
    mean[..., 6] = 0.9
    correlated, logp = sample_action_chunk(mean, 0.1, mode="temporally_correlated", rho=0.8)
    plain, _ = sample_action_chunk(mean, 0.1, mode="legacy")
    corr_delta = (correlated[:, 1:, :6] - correlated[:, :-1, :6]).square().mean()
    plain_delta = (plain[:, 1:, :6] - plain[:, :-1, :6]).square().mean()
    assert corr_delta < plain_delta * 0.35
    assert torch.isfinite(logp).all()
    assert (correlated[..., 6] == 0).sum() < (plain[..., 6] == 0).sum() * 0.15
    assert abs(float(correlated[..., :6].mean()) - 0.25) < 0.03
    center = mean[0, 0, 0]
    direct = torch.tanh(center)
    assert abs(float(center) - float(direct)) > 0.003


def test_backward_through_new_likelihood():
    means = torch.zeros(3, 8, 7, requires_grad=True)
    with torch.no_grad():
        actions, _ = sample_action_chunk(means, 0.13, mode="temporally_correlated")
    score = chunk_log_prob(means, actions, 0.13, mode="temporally_correlated")
    score.mean().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert means.grad.abs().sum() > 0
