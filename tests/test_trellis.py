import itertools

import pytest
import torch

import mlkit as mk


def test_viterbi_matches_exhaustive_paths() -> None:
    states, transitions, length = 3, 1, 4
    codebook = torch.tensor([-1.2, -0.8, -0.3, 0.0, 0.2, 0.5, 0.9, 1.3])
    value = torch.tensor([[0.8, -0.2, 0.4, -0.7]])
    costs = []
    for initial in range(2**states):
        for incoming in itertools.product(range(2**transitions), repeat=length - 1):
            path = [initial]
            for transition in incoming:
                path.append(((path[-1] << transitions) | transition) & (2**states - 1))
            reconstruction = codebook[torch.tensor(path)]
            costs.append((value[0] - reconstruction).square().sum())
    result = mk.viterbi(value, codebook, states, transitions)
    torch.testing.assert_close((value - result).square().sum(), torch.stack(costs).min())


@pytest.mark.cuda
@pytest.mark.parametrize("L", [4, 8, 12])
def test_fused_viterbi_matches_torch(L: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    values = torch.randn(
        4, 17, device="cuda", generator=torch.Generator(device="cuda").manual_seed(59)
    )
    codes = mk.one_mad(torch.arange(2**L, device="cuda"))
    expected = mk.viterbi(values, codes, L, 2, backend="torch")
    result = mk.viterbi(values, codes, L, 2, backend="triton")
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_trellis_codec_and_feedback() -> None:
    weights = torch.randn(8, 16)
    quantizer = mk.trellis(L=6, k=2, tile=4, chunk=2)
    context = mk.Ctx(H=torch.eye(16))
    result = quantizer(weights, context)
    assert result.codec == "trellis"
    assert result.w.shape == weights.shape
    assert result.bits == 2 * weights.numel() + 4 * weights.numel() / 16
    reference_context = mk.Ctx(H=torch.eye(16))
    feedback = mk.ldlq(quantizer, step=4)(weights, reference_context)
    torch.testing.assert_close(feedback.w, result.w)
    assert feedback.codec == "trellis"


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_trellis_checkpoint(tmp_path) -> None:
    module = torch.nn.Sequential(torch.nn.Linear(16, 8, bias=False))
    converted = mk.quantize(module, mk.trellis(L=6, tile=4), calib=None)
    converted.save(tmp_path / "trellis")
    restored = mk.load(tmp_path / "trellis", model=lambda: torch.nn.Sequential(
        torch.nn.Linear(16, 8, bias=False)
    ))
    torch.testing.assert_close(converted.module[0].weight, restored.module[0].weight,
                               rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_parallel_trellis_decoder_matches_differentiable_reconstruction() -> None:
    result = mk.trellis(L=8, tile=4)(torch.randn(8, 16))
    result.params["scale"].requires_grad_(True)
    differentiable = result.w
    differentiable.square().mean().backward()
    assert result.params["scale"].grad is not None
    result.params["scale"].requires_grad_(False)
    torch.testing.assert_close(result.w, differentiable.detach(), rtol=1e-6, atol=1e-6)
