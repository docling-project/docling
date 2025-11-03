from __future__ import annotations
import os, json, asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import weaviate
from weaviate.collections.classes.filters import Filter
from weaviate.collections.classes.grpc import Sort

from genos_tools.config import ProfileConfig

class AsyncSessionContext:
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.session = None
    async def __aenter__(self):
        self.session = self.session_factory()
        return self.session
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

async def _get_vdb(session: AsyncSession, vdb_id: int) -> str | None:
    res = await session.execute(text("SELECT `index` FROM vector_database WHERE id = :vdb_id"), {"vdb_id": vdb_id})
    return res.unique().scalars().first()

async def _get_docs(session: AsyncSession, vdb_id: int) -> list[int]:
    res = await session.execute(text("SELECT id FROM document WHERE vdb_id = :vdb_id AND is_active=1"), {"vdb_id": vdb_id})
    return list(res.unique().scalars().all())

async def run(cfg: ProfileConfig, vdb_id: int, output_dir_override: str | None = None) -> None:
    engine = create_async_engine(cfg.db.url, echo=False, future=True)
    Session = async_sessionmaker(bind=engine, class_=AsyncSession, autocommit=False, autoflush=False, future=True)

    async with AsyncSessionContext(Session) as session:
        vdb_index = await _get_vdb(session, vdb_id)
        if not vdb_index:
            raise SystemExit(f"[error] vdb_id={vdb_id} 의 index를 찾지 못했습니다.")

        doc_ids = await _get_docs(session, vdb_id)
        base_dir = output_dir_override or cfg.app.output_dir
        out_dir = os.path.join(base_dir, str(vdb_id))
        os.makedirs(out_dir, exist_ok=True)

        client = weaviate.use_async_with_local(
            host=cfg.weaviate.host,
            port=cfg.weaviate.port,
            grpc_port=cfg.weaviate.grpc_port,
            skip_init_checks=cfg.weaviate.skip_init_checks,
        )
        await client.connect()
        try:
            collection = client.collections.get(vdb_index)
            for doc_id in doc_ids:
                resp = await collection.query.fetch_objects(
                    filters=Filter.by_property("doc_id").equal(doc_id),
                    sort=Sort.by_property("i_chunk_on_doc", ascending=True),
                )
                data = [obj.properties for obj in resp.objects]
                with open(os.path.join(out_dir, f"{data[0].get('file_name', '') if len(data) > 0 else ""}-{doc_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False, default=str)
        finally:
            await client.close()

def main(cfg: ProfileConfig, vdb_id: int, output_dir: str | None):
    return asyncio.run(run(cfg, vdb_id, output_dir))


if __name__ == "__main__":
    main()
