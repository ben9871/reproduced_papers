"""Basic smoke tests ensuring the reproduction entry points import."""

def test_import_implementation():
    import importlib

    mod = importlib.import_module("implementation")
    assert hasattr(mod, "main")


def test_import_merlin_module():
    import importlib

    mod = importlib.import_module("lib.legacy_merlin_reproduction")
    assert hasattr(mod, "main")


def test_import_models():
    import importlib

    mod = importlib.import_module("model")
    assert hasattr(mod, "SingleGI")
