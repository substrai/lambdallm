"""Tests for Lambda Layer build script with dependency optimization."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lambdallm.deploy.layer_builder import (
    LayerBuilder,
    LayerBuildResult,
    LayerBuildError,
    build_layer,
    DEFAULT_STRIP_PATTERNS,
    DEFAULT_EXCLUDE_MODULES,
    LAMBDA_LAYER_MAX_SIZE_MB,
)


class TestLayerBuilderInit:
    """Test LayerBuilder initialization and configuration."""

    def test_default_configuration(self):
        builder = LayerBuilder(requirements=["substrai-lambdallm"])
        assert builder.requirements == ["substrai-lambdallm"]
        assert builder.python_version == "3.12"
        assert builder.output_dir == "./build"
        assert builder.compile_bytecode is True
        assert builder.layer_name == "lambdallm-layer"

    def test_custom_configuration(self):
        builder = LayerBuilder(
            requirements=["boto3", "pyyaml"],
            python_version="3.11",
            output_dir="/tmp/build",
            compile_bytecode=False,
            layer_name="custom-layer",
        )
        assert builder.requirements == ["boto3", "pyyaml"]
        assert builder.python_version == "3.11"
        assert builder.output_dir == "/tmp/build"
        assert builder.compile_bytecode is False
        assert builder.layer_name == "custom-layer"

    def test_default_strip_patterns_include_caches(self):
        assert "__pycache__" in DEFAULT_STRIP_PATTERNS
        assert "*.pyc" in DEFAULT_STRIP_PATTERNS
        assert "tests" in DEFAULT_STRIP_PATTERNS
        assert "docs" in DEFAULT_STRIP_PATTERNS

    def test_default_exclude_modules(self):
        assert "pip" in DEFAULT_EXCLUDE_MODULES
        assert "setuptools" in DEFAULT_EXCLUDE_MODULES
        assert "wheel" in DEFAULT_EXCLUDE_MODULES


class TestLayerBuilderStripPatterns:
    """Test file stripping logic."""

    def test_strip_pycache_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            cache_dir = target / "mypackage" / "__pycache__"
            cache_dir.mkdir(parents=True)
            (cache_dir / "module.cpython-312.pyc").write_text("bytecode")

            builder = LayerBuilder(requirements=[])
            count = builder._strip_patterns(target)
            assert count > 0
            assert not cache_dir.exists()

    def test_strip_test_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            test_dir = target / "mypackage" / "tests"
            test_dir.mkdir(parents=True)
            (test_dir / "test_module.py").write_text("def test(): pass")

            builder = LayerBuilder(requirements=[])
            count = builder._strip_patterns(target)
            assert count > 0
            assert not test_dir.exists()

    def test_strip_preserves_source_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            src_dir = target / "mypackage"
            src_dir.mkdir()
            src_file = src_dir / "core.py"
            src_file.write_text("def main(): pass")

            builder = LayerBuilder(requirements=[])
            builder._strip_patterns(target)
            assert src_file.exists()


class TestLayerBuilderExcludeModules:
    """Test module exclusion logic."""

    def test_remove_directory_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            pip_dir = target / "pip"
            pip_dir.mkdir()
            (pip_dir / "__init__.py").write_text("")

            builder = LayerBuilder(requirements=[])
            excluded = builder._remove_excluded_modules(target)
            assert "pip" in excluded
            assert not pip_dir.exists()

    def test_remove_single_file_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            module_file = target / "setuptools.py"
            module_file.write_text("# setuptools")

            builder = LayerBuilder(requirements=[], exclude_modules=["setuptools"])
            excluded = builder._remove_excluded_modules(target)
            assert "setuptools" in excluded
            assert not module_file.exists()

    def test_skip_nonexistent_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            builder = LayerBuilder(requirements=[], exclude_modules=["nonexistent"])
            excluded = builder._remove_excluded_modules(target)
            assert excluded == []


class TestLayerBuilderCompile:
    """Test bytecode compilation."""

    def test_compile_python_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            pkg_dir = target / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("x = 1")
            (pkg_dir / "core.py").write_text("def main(): return 42")

            builder = LayerBuilder(requirements=[])
            count = builder._compile_bytecode(target)
            assert count >= 2


class TestLayerBuilderEstimate:
    """Test size estimation."""

    def test_estimate_known_packages(self):
        builder = LayerBuilder(requirements=["boto3", "pyyaml"])
        estimate = builder.estimate_size()
        # boto3 ~15MB + pyyaml ~0.5MB, with 0.6 discount
        assert estimate > 5.0
        assert estimate < 20.0

    def test_estimate_unknown_packages(self):
        builder = LayerBuilder(requirements=["unknown-pkg"])
        estimate = builder.estimate_size()
        # Unknown defaults to 1.0 MB * 0.6
        assert estimate == pytest.approx(0.6, abs=0.1)


class TestBuildLayerConvenience:
    """Test the convenience build_layer function."""

    def test_default_requirements(self):
        with patch.object(LayerBuilder, "build") as mock_build:
            mock_build.return_value = LayerBuildResult(
                zip_path="/tmp/layer.zip",
                size_mb=4.5,
                python_version="3.12",
                packages_included=["substrai-lambdallm[bedrock]"],
                packages_excluded=[],
                content_hash="abc123",
                optimizations_applied=[],
            )
            result = build_layer()
            assert result.size_mb == 4.5

    def test_custom_requirements(self):
        with patch.object(LayerBuilder, "build") as mock_build:
            mock_build.return_value = LayerBuildResult(
                zip_path="/tmp/layer.zip",
                size_mb=2.0,
                python_version="3.11",
                packages_included=["pyyaml"],
                packages_excluded=[],
                content_hash="def456",
                optimizations_applied=[],
            )
            result = build_layer(
                requirements=["pyyaml"],
                python_version="3.11",
                layer_name="minimal-layer",
            )
            assert result.python_version == "3.11"


class TestLayerBuildResult:
    """Test LayerBuildResult dataclass."""

    def test_result_fields(self):
        result = LayerBuildResult(
            zip_path="/build/layer.zip",
            size_mb=12.5,
            python_version="3.12",
            packages_included=["boto3", "pyyaml"],
            packages_excluded=["pip", "setuptools"],
            content_hash="a1b2c3d4",
            optimizations_applied=["stripped 45 files", "compiled 120 files"],
        )
        assert result.zip_path == "/build/layer.zip"
        assert result.size_mb == 12.5
        assert len(result.packages_included) == 2
        assert len(result.packages_excluded) == 2
        assert len(result.optimizations_applied) == 2
