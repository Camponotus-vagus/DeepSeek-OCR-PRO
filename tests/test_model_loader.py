"""Tests for the marker-pdf model loader helpers."""

import sys
import types
from unittest.mock import MagicMock, patch



def _install_fake_marker(monkeypatch, model_dict):
    """Install a fake `marker.models` module exposing create_model_dict()."""
    fake_marker = types.ModuleType("marker")
    fake_models = types.ModuleType("marker.models")
    fake_models.create_model_dict = MagicMock(return_value=model_dict)
    fake_marker.models = fake_models
    monkeypatch.setitem(sys.modules, "marker", fake_marker)
    monkeypatch.setitem(sys.modules, "marker.models", fake_models)
    return fake_models.create_model_dict


class TestCreateMarkerModels:
    def test_returns_model_dict(self, monkeypatch):
        expected = {"layout": object(), "detection": object()}
        create = _install_fake_marker(monkeypatch, expected)

        from surya_ocr.engine.model_loader import create_marker_models

        result = create_marker_models()
        create.assert_called_once_with()
        assert result is expected


class TestUnloadModels:
    def test_clears_dict_and_collects(self):
        from surya_ocr.engine.model_loader import unload_models

        model_dict = {"a": 1, "b": 2}
        with patch("gc.collect") as mock_gc:
            unload_models(model_dict)

        assert model_dict == {}
        mock_gc.assert_called_once()

    def test_none_is_safe(self):
        from surya_ocr.engine.model_loader import unload_models
        # Should not raise
        unload_models(None)

    def test_calls_cuda_empty_cache_when_available(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_cuda = types.SimpleNamespace(
            is_available=MagicMock(return_value=True),
            empty_cache=MagicMock(),
        )
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        from surya_ocr.engine.model_loader import unload_models

        unload_models({})
        fake_cuda.is_available.assert_called_once()
        fake_cuda.empty_cache.assert_called_once()

    def test_skips_empty_cache_when_cuda_unavailable(self, monkeypatch):
        fake_torch = types.ModuleType("torch")
        fake_cuda = types.SimpleNamespace(
            is_available=MagicMock(return_value=False),
            empty_cache=MagicMock(),
        )
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        from surya_ocr.engine.model_loader import unload_models

        unload_models({})
        fake_cuda.empty_cache.assert_not_called()

    def test_ignores_missing_torch(self, monkeypatch):
        # Hide torch so the `import torch` inside unload_models fails
        monkeypatch.setitem(sys.modules, "torch", None)

        from surya_ocr.engine.model_loader import unload_models
        # Should swallow ImportError and not raise
        unload_models({})
