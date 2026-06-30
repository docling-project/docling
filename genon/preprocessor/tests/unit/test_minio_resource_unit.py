from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stub_minio_env(monkeypatch):
    """`common.settings.minio_config` import 시점의 env 검증을 우회."""
    monkeypatch.setenv("MINIO_ENDPOINT", "test-endpoint:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret")
    monkeypatch.setenv("HOSTNAME", "preprocessor-test-0")
    monkeypatch.setenv("PREPROCESSOR_ID", "226")
    monkeypatch.setenv("MQ_HOST", "x")
    monkeypatch.setenv("MQ_PORT", "0")
    monkeypatch.setenv("MQ_USER", "x")
    monkeypatch.setenv("MQ_PASSWORD", "x")
    monkeypatch.setenv("MQ_VHOST", "/")
    monkeypatch.setenv("MQ_EXCHANGE_TYPE", "topic")
    monkeypatch.setenv("MQ_EXCHANGE_NAME_LOG", "log")
    monkeypatch.setenv("MQ_QUEUE_NAME_LOG", "log")
    monkeypatch.setenv("MQ_QUEUE_BIND_ROUTING_KEY_LOG", "*.#")


def _make_obj(object_name: str, is_dir: bool = False):
    return SimpleNamespace(object_name=object_name, is_dir=is_dir)


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_file_lock_acquires_and_writes_pid(tmp_path: Path):
    from util.minio_resource import FileLock

    lock_file = tmp_path / "sub" / ".lock"

    with FileLock(str(lock_file), timeout_sec=1, poll_interval=0.05):
        assert lock_file.exists()
        content = lock_file.read_text()
        assert f"pid={os.getpid()}" in content

    # __exit__ 후에도 파일 자체는 남아 있음(해제만 됨)
    assert lock_file.exists()


@pytest.mark.unit
def test_file_lock_times_out_when_held_elsewhere(tmp_path: Path):
    from util.minio_resource import FileLock

    lock_file = tmp_path / ".lock"
    lock_file.touch()

    held = open(lock_file, "a+")
    try:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.time()
        with pytest.raises(TimeoutError):
            with FileLock(str(lock_file), timeout_sec=1, poll_interval=0.05):
                pass
        assert time.time() - start >= 1.0
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()


@pytest.mark.unit
def test_file_lock_blocks_concurrent_acquire(tmp_path: Path):
    from util.minio_resource import FileLock

    lock_file = tmp_path / ".lock"
    order: list[str] = []

    def first():
        with FileLock(str(lock_file), timeout_sec=5, poll_interval=0.05):
            order.append("first-enter")
            time.sleep(0.3)
            order.append("first-exit")

    def second():
        time.sleep(0.05)
        with FileLock(str(lock_file), timeout_sec=5, poll_interval=0.05):
            order.append("second-enter")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert order == ["first-enter", "first-exit", "second-enter"]


# ---------------------------------------------------------------------------
# download_resource_files
# ---------------------------------------------------------------------------

def _patch_minio(monkeypatch, objects):
    """`minio.Minio`를 MagicMock으로 치환하고 반환.

    `download_resource_files`는 호출 시점에 `from minio import Minio`로
    `minio.Minio`를 조회하므로(lazy import), 모듈 속성이 아닌 `minio.Minio`를 patch한다.
    """
    import minio

    minio_mock = MagicMock()
    minio_mock.list_objects.return_value = iter(objects)
    client_factory = MagicMock(return_value=minio_mock)
    monkeypatch.setattr(minio, "Minio", client_factory)
    return minio_mock, client_factory


@pytest.mark.unit
def test_download_resource_files_downloads_all_objects(tmp_path: Path, monkeypatch):
    from util.minio_resource import download_resource_files

    dest = tmp_path / "resource"
    objs = [
        _make_obj("226/resource/a.txt"),
        _make_obj("226/resource/sub/b.json"),
    ]
    minio_mock, factory = _patch_minio(monkeypatch, objs)

    download_resource_files(bucket_name="preprocessor", resource_id=226, path=str(dest))

    # 디렉터리 생성
    assert dest.is_dir()

    # list_objects 호출 인자 확인
    minio_mock.list_objects.assert_called_once_with(
        "preprocessor", prefix="226/resource", recursive=True
    )

    # fget_object 2회, 경로 정확
    assert minio_mock.fget_object.call_count == 2
    calls = {call.kwargs["object_name"]: call.kwargs for call in minio_mock.fget_object.call_args_list}
    assert calls["226/resource/a.txt"]["file_path"] == str(dest / "a.txt")
    assert calls["226/resource/a.txt"]["bucket_name"] == "preprocessor"
    assert calls["226/resource/sub/b.json"]["file_path"] == str(dest / "sub" / "b.json")

    # 하위 디렉터리 실제 생성되었는지 확인
    assert (dest / "sub").is_dir()

    # Minio 클라이언트는 한 번만 생성
    assert factory.call_count == 1


@pytest.mark.unit
def test_download_resource_files_skips_existing(tmp_path: Path, monkeypatch):
    from util.minio_resource import download_resource_files

    dest = tmp_path / "resource"
    dest.mkdir()
    existing = dest / "a.txt"
    existing.write_text("already here")

    objs = [_make_obj("226/resource/a.txt")]
    minio_mock, _ = _patch_minio(monkeypatch, objs)

    download_resource_files(bucket_name="preprocessor", resource_id=226, path=str(dest))

    minio_mock.fget_object.assert_not_called()
    assert existing.read_text() == "already here"


@pytest.mark.unit
def test_download_resource_files_skips_directory_entries(tmp_path: Path, monkeypatch):
    from util.minio_resource import download_resource_files

    dest = tmp_path / "resource"
    objs = [
        _make_obj("226/resource/dir/", is_dir=True),
        _make_obj("226/resource/a.txt"),
    ]
    minio_mock, _ = _patch_minio(monkeypatch, objs)

    download_resource_files(bucket_name="preprocessor", resource_id=226, path=str(dest))

    assert minio_mock.fget_object.call_count == 1
    args = minio_mock.fget_object.call_args.kwargs
    assert args["object_name"] == "226/resource/a.txt"


@pytest.mark.unit
def test_download_resource_files_empty_list(tmp_path: Path, monkeypatch):
    from util.minio_resource import download_resource_files

    dest = tmp_path / "resource"
    minio_mock, _ = _patch_minio(monkeypatch, [])

    download_resource_files(bucket_name="preprocessor", resource_id=226, path=str(dest))

    assert dest.is_dir()
    minio_mock.fget_object.assert_not_called()


@pytest.mark.unit
def test_download_resource_files_propagates_exception(tmp_path: Path, monkeypatch):
    from util.minio_resource import download_resource_files

    dest = tmp_path / "resource"
    objs = [_make_obj("226/resource/a.txt")]
    minio_mock, _ = _patch_minio(monkeypatch, objs)
    minio_mock.fget_object.side_effect = RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        download_resource_files(bucket_name="preprocessor", resource_id=226, path=str(dest))


@pytest.mark.unit
def test_download_resource_files_skips_when_minio_not_configured(tmp_path: Path, monkeypatch):
    """PREPROCESSOR_ID 가 있어도 MINIO_* 미설정이면 예외 없이 건너뛴다."""
    from util.minio_resource import download_resource_files

    # autouse fixture 가 set 한 MINIO_* 를 제거 (PREPROCESSOR_ID 는 유지)
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)

    dest = tmp_path / "resource"
    _, factory = _patch_minio(monkeypatch, [_make_obj("226/resource/a.txt")])

    # 예외 없이 반환
    download_resource_files(bucket_name="preprocessor", resource_id=226, path=str(dest))

    # MinIO 클라이언트 미생성, 디렉터리/lock 미생성
    factory.assert_not_called()
    assert not dest.exists()
    assert not (dest / ".download_resource_files.lock").exists()


@pytest.mark.unit
def test_download_resource_files_ignores_empty_relative_path(tmp_path: Path, monkeypatch):
    """prefix와 object_name이 정확히 일치해 rel_path가 비는 경우 스킵."""
    from util.minio_resource import download_resource_files

    dest = tmp_path / "resource"
    objs = [
        _make_obj("226/resource"),  # rel_path == ""
        _make_obj("226/resource/a.txt"),
    ]
    minio_mock, _ = _patch_minio(monkeypatch, objs)

    download_resource_files(bucket_name="preprocessor", resource_id=226, path=str(dest))

    assert minio_mock.fget_object.call_count == 1
    assert minio_mock.fget_object.call_args.kwargs["object_name"] == "226/resource/a.txt"
