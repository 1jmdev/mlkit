import mlkit


def test_package_version() -> None:
    assert mlkit.__version__ == "0.1.0"


def test_deferred_data_does_not_require_tokenizer_or_download() -> None:
    source = mlkit.data("wikitext2", n=8, seq=128)
    assert isinstance(source, mlkit.DataSource)
    assert source.seq == 128
