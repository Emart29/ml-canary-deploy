import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.base import create_all_tables, AsyncSessionLocal
from store.metadata import MetadataStore


async def main():
    print("[init] Creating database tables...")
    await create_all_tables()
    print("[init] Tables ready.")

    print("[init] Running heart disease canary demo...")
    from examples.heart_disease.demo import run_demo
    await run_demo()

    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        models = await meta.list_model_names()
        deployments = await meta.list_deployments()
        print("[init] Summary:")
        print(f"[init]   models registered : {len(models)} {models}")
        print(f"[init]   deployments       : {len(deployments)}")
        for d in deployments:
            events = await meta.get_events(d.id, limit=100)
            status = d.status.value if hasattr(d.status, "value") else d.status
            print(f"[init]     - {d.name}: status={status}, events={len(events)}")
    print("[init] Done.")


if __name__ == "__main__":
    asyncio.run(main())
