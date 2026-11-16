"""Public API and namespace-boundary contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
from importlib.metadata import version
from importlib.util import resolve_name
from pathlib import Path

import pytest

import mveeg


def test_root_namespace_is_deliberately_small():
    assert mveeg.__all__ == ["decoding", "encoding", "prep", "__version__"]
    assert not hasattr(mveeg, "transform_metadata")


def test_version_comes_from_project_metadata():
    assert mveeg.__version__ == version("mveeg")


def test_preprocessing_public_api():
    assert set(mveeg.prep.__all__) == {
        "DatasetPipeline",
        "ExternalPipeline",
        "RawPipeline",
        "init_external",
        "init_pipeline",
        "open_pipeline",
        "steps",
    }
    assert hasattr(mveeg.prep.DatasetPipeline, "preprocess_epochs")
    assert hasattr(mveeg.prep.DatasetPipeline, "select_subjects")
    assert not hasattr(mveeg.prep, "preprocess_epochs")


def test_model_public_apis():
    assert set(mveeg.decoding.__all__) == {"DecodingPipeline", "init_pipeline"}
    assert set(mveeg.encoding.__all__) == {"EncodingPipeline", "init_pipeline"}
    for pipeline in (mveeg.decoding.DecodingPipeline, mveeg.encoding.EncodingPipeline):
        assert hasattr(pipeline, "transform_metadata")


def test_all_metadata_transform_apis_share_one_signature():
    pipelines = (
        mveeg.prep.RawPipeline,
        mveeg.prep.ExternalPipeline,
        mveeg.decoding.DecodingPipeline,
        mveeg.encoding.EncodingPipeline,
    )
    signatures = [inspect.signature(pipeline.transform_metadata) for pipeline in pipelines]
    for signature in signatures:
        assert list(signature.parameters) == ["self", "variables"]
        assert signature.parameters["variables"].kind is inspect.Parameter.VAR_KEYWORD


@pytest.mark.parametrize(
    "module",
    [
        "mveeg._shared",
        "mveeg.io",
        "mveeg.summaries",
        "mveeg.validation",
        "mveeg.prep.dataset",
        "mveeg.prep.external",
        "mveeg.decoding.analysis",
        "mveeg.encoding.analysis",
    ],
)
def test_deleted_modules_are_not_importable(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def _imported_modules(path: Path, package: Path) -> list[str]:
    module = ".".join(path.relative_to(package.parent).with_suffix("").parts)
    parent = module.rpartition(".")[0]
    imports = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = resolve_name("." * node.level + name, parent)
            imports.append(name)
            imports.extend(f"{name}.{alias.name}" for alias in node.names)
    return imports


@pytest.mark.parametrize(
    ("relative_path", "source", "expected"),
    [
        ("prep/example.py", "import mveeg.decoding", ["mveeg.decoding"]),
        ("prep/example.py", "from mveeg import decoding", ["mveeg", "mveeg.decoding"]),
        ("prep/example.py", "from .. import decoding", ["mveeg", "mveeg.decoding"]),
        (
            "prep/example.py",
            "from ..decoding import DecodingPipeline",
            ["mveeg.decoding", "mveeg.decoding.DecodingPipeline"],
        ),
        (
            "prep/pipeline/example.py",
            "from ... import decoding",
            ["mveeg", "mveeg.decoding"],
        ),
        ("prep/example.py", "from . import local", ["mveeg.prep", "mveeg.prep.local"]),
        ("prep/example.py", '"""Use mveeg.decoding."""', []),
    ],
)
def test_import_detection_resolves_relative_paths(tmp_path, relative_path, source, expected):
    package = tmp_path / "mveeg"
    path = package / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    assert _imported_modules(path, package) == expected


def test_public_pipelines_do_not_import_each_other():
    package = Path("src/mveeg")
    namespaces = ("prep", "decoding", "encoding")
    for namespace in namespaces:
        for path in (package / namespace).rglob("*.py"):
            forbidden = [
                name
                for name in _imported_modules(path, package)
                if any(
                    name == f"mveeg.{other}" or name.startswith(f"mveeg.{other}.")
                    for other in namespaces
                    if other != namespace
                )
            ]
            assert forbidden == [], f"{path} crosses public namespaces: {forbidden}"


def test_private_foundations_do_not_depend_on_public_pipelines():
    package = Path("src/mveeg")
    paths = [package / "_provenance.py"]
    for directory in ("_dataset", "_analysis", "_results"):
        paths.extend((package / directory).rglob("*.py"))
    for path in paths:
        forbidden = [
            name
            for name in _imported_modules(path, package)
            if any(
                name == f"mveeg.{namespace}" or name.startswith(f"mveeg.{namespace}.")
                for namespace in ("prep", "decoding", "encoding")
            )
        ]
        assert forbidden == [], f"{path} depends on public pipelines: {forbidden}"
