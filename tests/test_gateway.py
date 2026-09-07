from __future__ import annotations

import asyncio

from pydantic import SecretStr

from assistai.config import Settings
from assistai.gateway import Gateway
from tests.fakes import PRIMARY, QUARANTINE, catalog_response, client_for, manifest


async def test_heartbeat_emits_until_shutdown() -> None:
    gateway = Gateway(Settings(heartbeat_seconds=0.01))

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.05)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.beats >= 2


async def test_shutdown_does_not_wait_out_the_interval() -> None:
    """SIGTERM must not block for the heartbeat interval before the process exits."""
    gateway = Gateway(Settings(heartbeat_seconds=3600))

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.01)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.beats == 1


async def test_repeated_shutdown_requests_are_harmless() -> None:
    gateway = Gateway(Settings(heartbeat_seconds=0.01))

    task = asyncio.create_task(gateway.run())
    gateway.request_shutdown("first")
    gateway.request_shutdown("second")
    await asyncio.wait_for(task, timeout=1)

    assert task.done()


async def test_skips_validation_without_api_key() -> None:
    gateway = Gateway(Settings(heartbeat_seconds=0.01))

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.validated_refs == ()


async def test_validates_pinned_models_when_key_present() -> None:
    client = client_for(lambda _req: catalog_response([PRIMARY, QUARANTINE]))
    gateway = Gateway(
        Settings(FIREWORKS_API_KEY=SecretStr("fw-secret"), heartbeat_seconds=0.01),
        client=client,
        manifest=manifest(),
    )

    task = asyncio.create_task(gateway.run())
    await asyncio.sleep(0.02)
    gateway.request_shutdown("test")
    await asyncio.wait_for(task, timeout=1)

    assert gateway.validated_refs == (PRIMARY, QUARANTINE)
    await client.aclose()
