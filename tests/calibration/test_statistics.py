import torch
from torch import nn

import mlkit as mk
from mlkit.calibration import StatisticAccumulator, limit_token_energy


def test_token_energy_limit_scales_only_outlier_rows() -> None:
    inputs = torch.randn(64, 16)
    inputs[5] *= 1000
    limited = limit_token_energy(inputs, 100.0)
    energy = inputs.square().sum(1)
    threshold = 100 * energy.median()
    torch.testing.assert_close(limited[5].square().sum(), threshold, rtol=1e-4, atol=0)
    ordinary = torch.arange(64) != 5
    assert torch.equal(limited[ordinary], inputs[ordinary])
    direction = torch.nn.functional.cosine_similarity(limited[5], inputs[5], dim=0)
    torch.testing.assert_close(direction, torch.ones(()), rtol=1e-5, atol=0)


def test_token_energy_limit_returns_ordinary_inputs_unchanged() -> None:
    inputs = torch.randn(64, 16)
    assert limit_token_energy(inputs, 100.0) is inputs
    inputs[5] *= 1000
    assert limit_token_energy(inputs, None) is inputs
    assert limit_token_energy(inputs[:1], 100.0).shape == (1, 16)


def test_zero_median_energy_disables_the_limit() -> None:
    inputs = torch.zeros(8, 4)
    inputs[0] = 1
    assert limit_token_energy(inputs, 100.0) is inputs


def test_hessian_accumulates_in_place_across_batches() -> None:
    batches = [torch.randn(32, 12) for _ in range(3)]
    accumulator = StatisticAccumulator("H")
    for batch in batches:
        accumulator.update(batch)
    rows = torch.cat(batches)
    torch.testing.assert_close(accumulator.result(), rows.T @ rows / len(rows))
    assert accumulator.result() is accumulator.result()


def test_outlier_tokens_do_not_dominate_calibrated_hessians() -> None:
    inputs = torch.randn(64, 16)
    inputs[5] *= 1000
    observed = {}

    @mk.quantizer
    def record(weight, context):
        observed[context.name] = context.H.clone()
        return mk.Q(weight, bits=32 * weight.numel())

    model = nn.Sequential(nn.Linear(16, 8))
    mk.quantize(model, record, calib=[inputs], cache_dir=None)
    limited = limit_token_energy(inputs, 100.0)
    torch.testing.assert_close(observed["0"], limited.T @ limited / 64)
    mk.quantize(model, record, calib=[inputs], cache_dir=None, token_energy_limit=None)
    torch.testing.assert_close(observed["0"], inputs.T @ inputs / 64)


def test_limited_statistics_preserve_ordinary_tokens_under_error_feedback() -> None:
    """A massive feature on a few tokens must not consume the precision of every other token."""
    generator = torch.Generator(device="cuda").manual_seed(101)
    weights = torch.randn(32, 64, generator=generator) * 0.05
    mixing = torch.randn(64, 64, generator=generator) * 0.3 + torch.eye(64)
    ordinary = torch.randn(4096, 64, generator=generator) @ mixing
    outliers = torch.randn(4, 64, generator=generator) @ mixing
    outliers[:, 5] = 3000
    inputs = torch.cat([ordinary, outliers])
    ordinary_hessian = ordinary.T @ ordinary / len(ordinary)
    format = mk.int(4, group=16)

    def ordinary_loss(reconstruction: torch.Tensor) -> float:
        return float(mk.proxy_loss(weights, reconstruction, mk.Ctx(H=ordinary_hessian)))

    def feedback(rows: torch.Tensor) -> torch.Tensor:
        hessian = rows.T @ rows / len(rows)
        return mk.gptq(format, refit=16)(weights, mk.Ctx(H=hessian)).w

    direct = ordinary_loss(format(weights).w)
    unlimited = ordinary_loss(feedback(inputs))
    limited = ordinary_loss(feedback(limit_token_energy(inputs, 100.0)))
    assert limited < direct
    assert limited < unlimited
