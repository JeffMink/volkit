"""Smoke test: the package and its subpackages import."""

import importlib

import pytest

import volkit


def test_version_is_set() -> None:
    assert volkit.__version__ == "0.1.0"


@pytest.mark.parametrize("name", ["pricing", "iv", "surface", "viz"])
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"volkit.{name}")
