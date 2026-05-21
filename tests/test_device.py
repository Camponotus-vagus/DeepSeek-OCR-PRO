"""Tests for device detection and CPU thread configuration."""

import os
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch


from surya_ocr.utils import device as device_mod


# ---------- get_physical_cores ----------

class TestGetPhysicalCores:
    def test_darwin_uses_sysctl(self, monkeypatch):
        monkeypatch.setattr(device_mod.platform, "system", lambda: "Darwin")
        fake_result = MagicMock(returncode=0, stdout="8\n")
        with patch.object(subprocess, "run", return_value=fake_result) as mock_run:
            assert device_mod.get_physical_cores() == 8
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][:2] == ["sysctl", "-n"]

    def test_darwin_sysctl_failure_falls_back(self, monkeypatch):
        monkeypatch.setattr(device_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        fake_result = MagicMock(returncode=1, stdout="")
        with patch.object(subprocess, "run", return_value=fake_result):
            # Falls through to logical/2 heuristic → 8
            assert device_mod.get_physical_cores() == 8

    def test_linux_parses_cpuinfo(self, monkeypatch, tmp_path):
        monkeypatch.setattr(device_mod.platform, "system", lambda: "Linux")
        cpuinfo = (
            "processor\t: 0\ncore id\t\t: 0\n\n"
            "processor\t: 1\ncore id\t\t: 1\n\n"
            "processor\t: 2\ncore id\t\t: 0\n\n"
            "processor\t: 3\ncore id\t\t: 1\n"
        )
        m = MagicMock()
        m.__enter__ = lambda self: iter(cpuinfo.splitlines(keepends=True))
        m.__exit__ = lambda self, *a: False

        with patch("builtins.open", return_value=m) as mock_open:
            cores = device_mod.get_physical_cores()
        mock_open.assert_called_with("/proc/cpuinfo")
        # Two unique "core id" lines
        assert cores == 2

    def test_linux_cpuinfo_missing_falls_back(self, monkeypatch):
        monkeypatch.setattr(device_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(os, "cpu_count", lambda: 12)
        with patch("builtins.open", side_effect=OSError("nope")):
            assert device_mod.get_physical_cores() == 6  # 12 // 2

    def test_unknown_platform_uses_heuristic(self, monkeypatch):
        monkeypatch.setattr(device_mod.platform, "system", lambda: "Windows")
        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        assert device_mod.get_physical_cores() == 2

    def test_cpu_count_none_returns_one(self, monkeypatch):
        monkeypatch.setattr(device_mod.platform, "system", lambda: "Windows")
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        # (None or 2) // 2 == 1, then max(1, 1) == 1
        assert device_mod.get_physical_cores() == 1

    def test_unexpected_exception_returns_safe_default(self, monkeypatch):
        def boom():
            raise RuntimeError("unexpected")

        monkeypatch.setattr(os, "cpu_count", boom)
        assert device_mod.get_physical_cores() == 4


# ---------- detect_device ----------

def _install_torch(monkeypatch, *, cuda_available=False, mps_attr=True, mps_available=False):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=MagicMock(return_value=cuda_available))
    if mps_attr:
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=MagicMock(return_value=mps_available))
        )
    else:
        fake_torch.backends = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


class TestDetectDevice:
    def test_explicit_override_returned_verbatim(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=False)
        assert device_mod.detect_device("cuda") == "cuda"
        assert device_mod.detect_device("cpu") == "cpu"

    def test_cuda_when_available(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=True)
        assert device_mod.detect_device("auto") == "cuda"

    def test_mps_on_apple_silicon(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=False, mps_available=True)
        monkeypatch.setattr(device_mod.platform, "machine", lambda: "arm64")
        assert device_mod.detect_device("auto") == "mps"

    def test_intel_mac_falls_back_to_cpu(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=False, mps_available=True)
        monkeypatch.setattr(device_mod.platform, "machine", lambda: "x86_64")
        assert device_mod.detect_device("auto") == "cpu"

    def test_no_acceleration_returns_cpu(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=False, mps_available=False)
        assert device_mod.detect_device("auto") == "cpu"

    def test_torch_without_mps_attribute(self, monkeypatch):
        _install_torch(monkeypatch, cuda_available=False, mps_attr=False)
        assert device_mod.detect_device("auto") == "cpu"


# ---------- configure_cpu_threads ----------

class TestConfigureCpuThreads:
    def test_sets_env_vars_and_torch_threads(self, monkeypatch):
        monkeypatch.setattr(device_mod, "get_physical_cores", lambda: 6)
        # Clean slate for env vars
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

        fake_torch = types.ModuleType("torch")
        fake_torch.set_num_threads = MagicMock()
        fake_torch.set_num_interop_threads = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        device_mod.configure_cpu_threads()

        assert os.environ["OMP_NUM_THREADS"] == "6"
        assert os.environ["MKL_NUM_THREADS"] == "6"
        fake_torch.set_num_threads.assert_called_once_with(6)
        fake_torch.set_num_interop_threads.assert_called_once_with(2)

    def test_interop_capped_at_core_count(self, monkeypatch):
        monkeypatch.setattr(device_mod, "get_physical_cores", lambda: 1)
        fake_torch = types.ModuleType("torch")
        fake_torch.set_num_threads = MagicMock()
        fake_torch.set_num_interop_threads = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        device_mod.configure_cpu_threads()
        fake_torch.set_num_interop_threads.assert_called_once_with(1)

    def test_torch_failure_is_swallowed(self, monkeypatch, caplog):
        monkeypatch.setattr(device_mod, "get_physical_cores", lambda: 4)
        fake_torch = types.ModuleType("torch")
        fake_torch.set_num_threads = MagicMock(side_effect=RuntimeError("nope"))
        fake_torch.set_num_interop_threads = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        # Should not raise
        device_mod.configure_cpu_threads()
        # Env vars still set
        assert os.environ["OMP_NUM_THREADS"] == "4"
