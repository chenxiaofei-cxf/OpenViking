# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for source_path deduplication in add-resource."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class _DummyTelemetry:
    def set(self, *args, **kwargs):
        return None

    def set_error(self, *args, **kwargs):
        return None

    def measure(self, *args, **kwargs):
        class _CtxMgr:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _CtxMgr()


class _FakeVikingFS:
    """Fake VikingFS for testing deduplication."""

    def __init__(self):
        self._source_metas = {}  # uri -> {source_path, source_format}
        self._directories = {}  # uri -> True
        self.agfs = MagicMock()

    def bind_request_context(self, ctx):
        class _CtxMgr:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _CtxMgr()

    async def exists(self, uri, ctx=None):
        return uri in self._directories

    async def mkdir(self, uri, exist_ok=False, ctx=None):
        self._directories[uri] = True
        return None

    async def delete_temp(self, temp_dir_path, ctx=None):
        return None

    def _uri_to_path(self, uri, ctx=None):
        return f"/mock/{uri.replace('viking://', '')}"

    async def write_source_meta(self, uri, source_path, source_format=None, ctx=None):
        """Store source metadata."""
        self._source_metas[uri] = {
            "source_path": source_path,
            "source_format": source_format,
        }

    async def read_source_meta(self, uri, ctx=None):
        """Read source metadata."""
        return self._source_metas.get(uri)

    async def ls(self, uri, ctx=None):
        """List directories."""
        entries = []
        for dir_uri in self._directories:
            if dir_uri.startswith(uri) and dir_uri != uri:
                # Simple check: direct children only
                remaining = dir_uri[len(uri) :].lstrip("/")
                if "/" not in remaining and remaining:
                    entries.append({"uri": dir_uri, "isDir": True, "name": remaining})
        return entries

    async def find_uri_by_source_path(self, source_path, scope="resources", ctx=None):
        """Find existing resource by source_path."""
        normalized = self._normalize_source_path(source_path)
        for uri, meta in self._source_metas.items():
            if self._normalize_source_path(meta.get("source_path", "")) == normalized:
                return uri
        return None

    @staticmethod
    def _normalize_source_path(source_path: str) -> str:
        """Normalize source path for comparison."""
        if not source_path:
            return ""
        if source_path.startswith(("http://", "https://", "git@", "git://")):
            return source_path.rstrip("/")
        try:
            from pathlib import Path

            path = Path(source_path).resolve()
            return str(path)
        except Exception:
            return source_path


class _DummyVikingDB:
    def get_embedder(self):
        return None


# ============ Tests for VikingFS source_path methods ============


@pytest.mark.asyncio
async def test_vikingfs_write_and_read_source_meta():
    """Test writing and reading source metadata."""
    fake_fs = _FakeVikingFS()

    # Write source meta
    await fake_fs.write_source_meta(
        uri="viking://resources/test-doc",
        source_path="/path/to/test.doc",
        source_format="docx",
    )

    # Read it back
    meta = await fake_fs.read_source_meta("viking://resources/test-doc")
    assert meta is not None
    assert meta["source_path"] == "/path/to/test.doc"
    assert meta["source_format"] == "docx"


@pytest.mark.asyncio
async def test_vikingfs_find_uri_by_source_path():
    """Test finding existing resource by source_path."""
    fake_fs = _FakeVikingFS()

    # Add a resource with source metadata
    fake_fs._directories["viking://resources/my-project"] = True
    await fake_fs.write_source_meta(
        uri="viking://resources/my-project",
        source_path="/home/user/projects/my-project",
        source_format="repository",
    )

    # Find it
    found_uri = await fake_fs.find_uri_by_source_path(source_path="/home/user/projects/my-project")
    assert found_uri == "viking://resources/my-project"


@pytest.mark.asyncio
async def test_vikingfs_find_uri_by_source_path_not_found():
    """Test finding non-existent source_path returns None."""
    fake_fs = _FakeVikingFS()

    found_uri = await fake_fs.find_uri_by_source_path(source_path="/nonexistent/path")
    assert found_uri is None


@pytest.mark.asyncio
async def test_vikingfs_normalize_source_path():
    """Test source path normalization."""
    # Local paths should be resolved to absolute
    normalized = _FakeVikingFS._normalize_source_path("./relative/path")
    assert not normalized.startswith(".")

    # URLs should be kept as-is (with trailing slash removed)
    assert (
        _FakeVikingFS._normalize_source_path("https://github.com/user/repo/")
        == "https://github.com/user/repo"
    )
    assert (
        _FakeVikingFS._normalize_source_path("git@github.com:user/repo.git")
        == "git@github.com:user/repo.git"
    )


# ============ Tests for ResourceProcessor deduplication ============


@pytest.mark.asyncio
async def test_resource_processor_dedup_finds_existing_resource(monkeypatch):
    """Test that ResourceProcessor finds and reuses existing resource."""
    from openviking.utils.resource_processor import ResourceProcessor

    fake_fs = _FakeVikingFS()

    # Simulate an existing resource
    fake_fs._directories["viking://resources/existing-doc"] = True
    fake_fs._source_metas["viking://resources/existing-doc"] = {
        "source_path": "/path/to/document.pdf",
        "source_format": "pdf",
    }

    monkeypatch.setattr(
        "openviking.utils.resource_processor.get_current_telemetry",
        lambda: _DummyTelemetry(),
    )
    monkeypatch.setattr("openviking.utils.resource_processor.get_viking_fs", lambda: fake_fs)

    rp = ResourceProcessor(vikingdb=_DummyVikingDB(), media_storage=None)
    rp._get_media_processor = MagicMock()
    rp._get_media_processor.return_value.process = AsyncMock(
        return_value=SimpleNamespace(
            temp_dir_path="viking://temp/tmpdir",
            source_path="/path/to/document.pdf",
            source_format="pdf",
            meta={},
            warnings=[],
        )
    )

    result = await rp.process_resource(
        path="/path/to/document.pdf",
        ctx=SimpleNamespace(account_id="test"),
        build_index=False,
        summarize=False,
    )

    # Should return the existing resource URI
    assert result["status"] == "success"
    assert result["root_uri"] == "viking://resources/existing-doc"
    assert result.get("is_update") is True


@pytest.mark.asyncio
async def test_resource_processor_creates_new_when_no_duplicate(monkeypatch):
    """Test that ResourceProcessor creates new resource when no duplicate exists."""
    from openviking.utils.resource_processor import ResourceProcessor

    fake_fs = _FakeVikingFS()

    monkeypatch.setattr(
        "openviking.utils.resource_processor.get_current_telemetry",
        lambda: _DummyTelemetry(),
    )
    monkeypatch.setattr("openviking.utils.resource_processor.get_viking_fs", lambda: fake_fs)

    # Mock the lock manager module
    mock_lock_manager = MagicMock()
    mock_lock_manager.create_handle = MagicMock(return_value=SimpleNamespace(id="handle-1"))
    mock_lock_manager.acquire_subtree = AsyncMock(return_value=False)
    mock_lock_manager.release = AsyncMock()
    mock_lock_manager.get_handle = MagicMock(return_value=None)

    # Create a mock LockContext that works as async context manager
    class MockLockContext:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(
        "openviking.storage.transaction.get_lock_manager", lambda: mock_lock_manager
    )
    monkeypatch.setattr("openviking.storage.transaction.LockContext", MockLockContext)

    rp = ResourceProcessor(vikingdb=_DummyVikingDB(), media_storage=None)
    rp._get_media_processor = MagicMock()
    rp._get_media_processor.return_value.process = AsyncMock(
        return_value=SimpleNamespace(
            temp_dir_path="viking://temp/tmpdir",
            source_path="/path/to/new-document.pdf",
            source_format="pdf",
            meta={},
            warnings=[],
        )
    )

    context_tree = SimpleNamespace(
        root=SimpleNamespace(
            uri="viking://resources/new-doc", temp_uri="viking://temp/new-doc_tmp"
        ),
        _candidate_uri="viking://resources/new-doc",
    )
    rp.tree_builder.finalize_from_temp = AsyncMock(return_value=context_tree)
    rp.tree_builder._resolve_unique_uri = AsyncMock(return_value="viking://resources/new-doc")

    # Add a mock for agfs.mv
    fake_fs.agfs.mv = MagicMock(return_value={"status": "ok"})

    result = await rp.process_resource(
        path="/path/to/new-document.pdf",
        ctx=SimpleNamespace(account_id="test"),
        build_index=False,
        summarize=False,
    )

    # Should create new resource
    assert result["status"] == "success"
    assert result["root_uri"] == "viking://resources/new-doc"
    assert result.get("is_update") is None

    # Source meta should be written
    assert "viking://resources/new-doc" in fake_fs._source_metas
