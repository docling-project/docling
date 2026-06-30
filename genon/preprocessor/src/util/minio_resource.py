import os
import time
import errno
import fcntl

from common.settings import MinioConfig
from common.logger import Logger

logger = Logger.getLogger(__name__)


class FileLock:
    def __init__(self, lock_path: str, timeout_sec: int = 600, poll_interval: float = 0.2):
        self.lock_path = lock_path
        self.timeout_sec = timeout_sec
        self.poll_interval = poll_interval
        self._fd = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self._fd = open(self.lock_path, "a+")

        start = time.time()
        while True:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd.seek(0)
                self._fd.truncate()
                self._fd.write(f"pid={os.getpid()} acquired_at={time.time()}\n")
                self._fd.flush()
                return self
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if (time.time() - start) >= self.timeout_sec:
                    raise TimeoutError(f"Timed out acquiring lock: {self.lock_path}")
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        finally:
            if self._fd:
                self._fd.close()
                self._fd = None


def download_resource_files(bucket_name: str, resource_id: int, path: str):
    from minio import Minio

    # MinIO 환경변수는 사용 시점에만 검증한다 (모듈 import 시점이 아님).
    # PREPROCESSOR_ID 가 있어도 MINIO_* 가 미설정이면 다운로드를 건너뛰고
    # 부팅을 계속한다 (MinIO 미사용 환경 / 시크릿 오설정 대응).
    minio_config = MinioConfig()
    if not minio_config.is_configured():
        logger.warning(
            "MINIO_* 환경변수 미설정 — 리소스 다운로드를 건너뜁니다. "
            "(resource_id=%s)", resource_id,
        )
        return

    os.makedirs(path, exist_ok=True)

    lock_file = os.path.join(path, ".download_resource_files.lock")

    with FileLock(lock_file, timeout_sec=3600):
        logger.info(f'Acquired lock: {lock_file} (pid={os.getpid()})')

        minio_client = Minio(
            endpoint=minio_config.MINIO_ENDPOINT,
            access_key=minio_config.MINIO_ACCESS_KEY,
            secret_key=minio_config.MINIO_SECRET_KEY,
            secure=False
        )

        prefix = f"{resource_id}/resource"
        objects = list(minio_client.list_objects(bucket_name, prefix=prefix, recursive=True))

        try:
            logger.info(f'Downloading {len(objects)} resource files for {bucket_name} {resource_id}')

            for i, obj in enumerate(objects):
                if obj.is_dir:
                    continue

                rel_path = obj.object_name[len(prefix):].lstrip("/\\")
                if not rel_path:
                    continue

                destination_file = os.path.join(path, rel_path)
                os.makedirs(os.path.dirname(destination_file), exist_ok=True)

                if os.path.exists(destination_file):
                    logger.info(
                        f'[SKIP {i+1}/{len(objects)}] "{destination_file}" already exists'
                    )
                    continue

                logger.info(
                    f'Downloading [{i+1}/{len(objects)}] "{obj.object_name}" '
                    f'to "{destination_file}"...'
                )

                minio_client.fget_object(
                    bucket_name=bucket_name,
                    object_name=obj.object_name,
                    file_path=destination_file
                )

            logger.info('Completed!')
        except Exception as e:
            logger.error(f'Failed to download resource files: {e}')
            raise
        finally:
            logger.info(f'Releasing lock: {lock_file} (pid={os.getpid()})')
