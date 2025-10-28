import os
import aiofile
import asyncio
import watchfiles
from typing import Union

from common.logger import Logger
from common.msg_queue import MsgQueue
from common.settings import settings, msg_queue_config


logger = Logger.getLogger(__name__)
log_msg_queue = MsgQueue(
    exchange_name=msg_queue_config.MQ_EXCHANGE_NAME_LOG,
    queue_name=msg_queue_config.MQ_QUEUE_NAME_LOG,
    queue_bind_routing_key=msg_queue_config.MQ_QUEUE_BIND_ROUTING_KEY_LOG
)

def _make_body(msg: Union[str, bytes]):
    return f'{settings.POD_ID}#{msg}'

async def watch(path: str):
    # 파일이 없으면 5초 대기 후 다시 시도
    while not os.path.exists(path):
        logger.warning(f"File not found: {path}. Retrying in 5 seconds...")
        await asyncio.sleep(5) 

    async with aiofile.AIOFile(path, mode="r", encoding="utf-8-sig") as af:
        reader = aiofile.Reader(af)

        async for changes in watchfiles.awatch(path):
            logger.info(f"CHANGE: {repr(changes)}")

            async for line in reader:
                try:
                    log_msg_queue.publish(
                        routing_key=msg_queue_config.MQ_ROUTING_KEY_LOG, 
                        body=_make_body(line)
                    )
                    logger.info(f"NEW LOG LINE: \n{line}")

                except Exception as e:
                    logger.error(f"LOG MSG QUEUE Error: {repr(e)}")


async def watch_multiple(paths: list[str]):
    tasks = [watch(path) for path in paths]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(
        watch_multiple(settings.LOG_PATH)
    )
