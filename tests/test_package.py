import pytest

import mlkit


@pytest.mark.device_independent
def test_package_version() -> None:
    assert mlkit.__version__ == "0.1.0"


@pytest.mark.device_independent
def test_public_names_are_exported() -> None:
    assert len(set(mlkit.__all__)) == len(mlkit.__all__)
    assert all(hasattr(mlkit, name) for name in mlkit.__all__)


@pytest.mark.device_independent
def test_deferred_data_does_not_require_tokenizer_or_download() -> None:
    source = mlkit.data("wikitext2", n=8, seq=128)
    assert isinstance(source, mlkit.DataSource)
    assert source.seq == 128
