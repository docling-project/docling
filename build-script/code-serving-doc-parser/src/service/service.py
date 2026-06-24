"""
해당 파일은 service.py의 interface 예시를 보여주기 위한 샘플 코드일 뿐,
실제 배포 시에는 service 경로 없이 Docker Container 형태로 배포되며,
service 폴더에는 선택된 code serving repository의 commit hash가 clone 되어 실행됩니다.
"""
from typing import Any, Dict


async def service(config: Dict[str, Any], data: Dict[str, Any]):
    data.update(config=config)
    return data


# import asyncio
# from fastapi.responses import StreamingResponse


# async def _stream(config: dict, data: dict):
#     for i in range(1, 6):
#         yield f"{i}\n"
#         await asyncio.sleep(0.2)
#     yield "Done\n"


# async def service(config: dict, data: dict):
#     return StreamingResponse(
#         _stream(config, data),
#         media_type="text/plain",
#     )
