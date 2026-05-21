"""Tests for path resolution utilities."""


from surya_ocr.utils.paths import ensure_output_dir, resolve_model_path


class TestEnsureOutputDir:
    def test_creates_subdirectory(self, tmp_path):
        out = ensure_output_dir(str(tmp_path / "out"), "report.pdf")
        assert out.exists()
        assert out.is_dir()
        assert out.name == "report"

    def test_uses_pdf_stem_only(self, tmp_path):
        out = ensure_output_dir(str(tmp_path), "/some/where/book.pdf")
        assert out.name == "book"

    def test_idempotent(self, tmp_path):
        first = ensure_output_dir(str(tmp_path), "x.pdf")
        second = ensure_output_dir(str(tmp_path), "x.pdf")
        assert first == second
        assert first.exists()

    def test_creates_nested_parent(self, tmp_path):
        out = ensure_output_dir(str(tmp_path / "a" / "b" / "c"), "doc.pdf")
        assert out.exists()
        assert out == tmp_path / "a" / "b" / "c" / "doc"


class TestResolveModelPath:
    def test_returns_path_when_valid(self, tmp_path):
        model_dir = tmp_path / "my_models"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        result = resolve_model_path(str(model_dir))
        assert result == model_dir.resolve()

    def test_returns_resolved_path_when_missing(self, tmp_path, monkeypatch):
        # Ensure neither CWD/models nor package/models exist
        monkeypatch.chdir(tmp_path)
        missing = tmp_path / "nope"
        result = resolve_model_path(str(missing))
        # When nothing matches, returns the resolved input path
        assert result == missing.resolve()

    def test_falls_back_to_cwd_models(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cwd_models = tmp_path / "models"
        cwd_models.mkdir()
        (cwd_models / "config.json").write_text("{}")

        # Provide an invalid primary path so it falls through
        result = resolve_model_path(str(tmp_path / "does_not_exist"))
        assert result == cwd_models

    def test_ignores_directory_without_config_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # CWD/models exists but has no config.json
        (tmp_path / "models").mkdir()

        result = resolve_model_path(str(tmp_path / "missing"))
        # Should fall back to the resolved original path, not the empty models dir
        assert result == (tmp_path / "missing").resolve()
