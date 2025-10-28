import asyncio
from fastapi import Request, HTTPException
import aiofiles
import aiohttp
import json
import traceback
import os
import importlib
import importlib.util
import subprocess
import sys


def genos_import(package_name: str, version: str = None, install_name: str = None):
    # 기본 패키지 이름 결정 (하위 모듈이 있을 수 있으므로 첫 번째 부분만 사용)
    package_parts = package_name.split('.')
    base_package_name = package_parts[0]

    # 버전이 지정되어 있다면, 설치 대상 문자열을 조합합니다.
    # 설치명이랑 import 모듈명이 다를 수 있음 ex) bs4 (beautifulsoup4)
    install_target = f"{install_name or base_package_name}"
    if version:
        install_target += f"=={version}"

    if importlib.util.find_spec(base_package_name) is None:
        print(f"{base_package_name} 라이브러리가 설치되지 않았습니다. 설치를 진행합니다...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_target])
            print(f"{base_package_name} 설치 완료")
        except subprocess.CalledProcessError:
            print(f"{base_package_name} 설치 실패")
            sys.exit(1)

    try:
        imported_package = importlib.import_module(base_package_name)
        result = imported_package

        # 하위 모듈을 가져오는 부분
        for i, part in enumerate(package_parts[1:], 1):
            try:
                result = getattr(result, part)
            except AttributeError:
                current_path = ".".join(package_parts[:i])
                next_path = f"{current_path}.{part}"
                try:
                    result = importlib.import_module(next_path)
                except ImportError:
                    if i == len(package_parts) - 1:
                        result = importlib.import_module(package_name)
                    else:
                        raise ImportError(f"import 실패: {next_path}")
        return result
    except ImportError:
        print(f"{package_name} 설치 후에도 임포트 실패")
        print(traceback.format_exc())
        sys.exit(1)


async def upload_files(file_list: list[dict], request: Request, concurrency: int = 1):
    """
    여러 파일을 동시에 업로드합니다.

    Parameters:
        file_list: [{'path': '파일 경로', 'name': 'object 이름'}] 으로 이루어진 리스트
        request: FastAPI Request 객체 (헤더에서 doc_id와 minio_form을 읽음)
        concurrency: 동시에 업로드할 최대 파일 수 (기본값: 5)

    Returns:
        업로드된 파일의 object_name 리스트
    """

    doc_id = request.headers.get('x-genos-doc-id')
    minio_form = request.headers.get('x-genos-minio-form')
    is_temp_doc = request.headers.get('x-genos-is-temp-doc') == 'true'

    # 보고서 생성 임시 문서 전처리 대응
    if minio_form is None or doc_id is None:
        return

    form_data = json.loads(minio_form)

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        async def _upload_single(org_path: str, dst_name: str):
            async with semaphore:
                try:
                    object_name = f'{doc_id}/{dst_name}' if not is_temp_doc else f'temp-document/{doc_id}/{dst_name}'
                    form_data['key'] = object_name

                    async with aiofiles.open(org_path, 'rb') as f:
                        file_bytes = await f.read()

                    data = aiohttp.FormData()
                    for key, value in form_data.items():
                        data.add_field(key, str(value))
                    data.add_field('file', file_bytes, filename=object_name, content_type='application/octet-stream')

                    async with session.post('http://llmops-minio-service:9000/user-media', data=data) as resp:
                        if resp.status >= 300:
                            raise HTTPException(status_code=resp.status, detail=await resp.text())

                except Exception as e:
                    print(f"Error on Upload file from {org_path}: {e}")
                finally:
                    await asyncio.to_thread(os.remove, org_path)

        tasks = [_upload_single(item['path'], item['name']) for item in file_list]
        results = await asyncio.gather(*tasks)
        print(f"\n\n{results=}\n\n", flush=True)
    return results


def merge_overlapping_bboxes(bboxes, x_tolerance=1, y_tolerance=1):
    def is_overlap(b1, b2):
        if b1['page'] != b2['page']:
            return False

        l1, r1, t1, btm1 = b1['bbox']['l'], b1['bbox']['r'], b1['bbox']['t'], b1['bbox']['b']
        l2, r2, t2, btm2 = b2['bbox']['l'], b2['bbox']['r'], b2['bbox']['t'], b2['bbox']['b']

        if (r1 < l2 - x_tolerance or
                l1 > r2 + x_tolerance or
                btm1 < t2 - y_tolerance or
                t1 > btm2 + y_tolerance):
            return False
        return True

    def merge_bboxes(b1, b2):
        return {
            'page': b1['page'],
            'type': 'text',
            'bbox': {
                'l': min(b1['bbox']['l'], b2['bbox']['l']),
                't': min(b1['bbox']['t'], b2['bbox']['t']),
                'r': max(b1['bbox']['r'], b2['bbox']['r']),
                'b': max(b1['bbox']['b'], b2['bbox']['b']),
            }
        }

    changed = True
    while changed:
        changed = False
        merged = []
        for current in bboxes:
            if current['type'] != 'text':
                merged.append(current)
                continue

            merged_in = False
            for i, existing in enumerate(merged):
                if is_overlap(existing, current):
                    merged[i] = merge_bboxes(existing, current)
                    changed = True
                    merged_in = True
                    break
            if not merged_in:
                merged.append(current)
        bboxes = merged
    return bboxes