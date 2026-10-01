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


@pytest.mark.device_independent
def test_package_exports_do_not_shadow_submodules() -> None:
    """A re-exported function named like its module would hide that module from attribute access."""
    import importlib
    import pkgutil
    import types

    for package_info in pkgutil.walk_packages(mlkit.__path__, prefix="mlkit."):
        if not package_info.ispkg:
            continue
        package = importlib.import_module(package_info.name)
        for module_info in pkgutil.iter_modules(package.__path__):
            attribute = getattr(package, module_info.name, None)
            if attribute is not None:
                assert isinstance(attribute, types.ModuleType), (
                    f"{package_info.name}.{module_info.name} is shadowed by a re-export"
                )
