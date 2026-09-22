"""Brokered jobs: schedules that always talk, watches that die.

The model does not get cron. It gets named job objects stored in SQLite.
A schedule messages the owner on every run, even when the report is empty.
A watch stays quiet unless it finds something, fails, or hits its TTL.
Jobs report; they never file, draft, write, or relay.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from assistai.errors import JobError, StoreError
from assistai.inference.types import ToolSpec
from assistai.relay import active_agent

JobKind = Literal["schedule", "watch"]

JOB_CREATE = "job_create"
JOB_LIST = "job_list"
JOB_CANCEL = "job_cancel"
JOB_RESCHEDULE = "job_reschedule"
JOB_TOOLS = (JOB_CREATE, JOB_LIST, JOB_CANCEL, JOB_RESCHEDULE)

MIN_EVERY_SECONDS = 60
MAX_EVERY_SECONDS = 30 * 24 * 3600
MAX_TTL_SECONDS = 30 * 24 * 3600
MAX_NAME_CHARS = 80
MAX_PROMPT_CHARS = 2000

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
_NOTHING = frozenset(
    {
        "",
        "none",
        "nothing",
        "nothing found",
        "nothing to report",
        "nothing_found",
    }
)

JOB_CREATE_SPEC = ToolSpec(
    name=JOB_CREATE,
    description=(
        "Propose a repeating job for this person. A schedule always texts them "
        "when it runs, even if the report is empty. A watch stays silent unless "
        "it finds something, fails, or expires, and it requires ttl_seconds. "
        "Jobs only report; they cannot file, draft, write, or relay. "
        "They may search, fetch, and read the shared calendar. "
        "every_seconds is the interval (86400 for daily)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["schedule", "watch"],
            },
            "name": {
                "type": "string",
                "description": "Short name used later to list or cancel, e.g. morning email.",
            },
            "prompt": {
                "type": "string",
                "description": "What to check each run. Keep it to a report, not an action.",
            },
            "every_seconds": {
                "type": "integer",
                "description": "Seconds between runs. 86400 is once a day.",
            },
            "ttl_seconds": {
                "type": "integer",
                "description": "Required for a watch. The watch is cancelled when this elapses.",
            },
        },
        "required": ["kind", "name", "prompt", "every_seconds"],
        "additionalProperties": False,
    },
)

JOB_LIST_SPEC = ToolSpec(
    name=JOB_LIST,
    description="List this person's active jobs: name, kind, interval, and when they next run.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)

JOB_CANCEL_SPEC = ToolSpec(
    name=JOB_CANCEL,
    description=(
        "Propose cancelling one of this person's jobs by name or id. "
        "It is cancelled only after they confirm."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Job name as listed."},
            "id": {"type": "string", "description": "Job id if the name is ambiguous."},
        },
        "additionalProperties": False,
    },
)

JOB_RESCHEDULE_SPEC = ToolSpec(
    name=JOB_RESCHEDULE,
    description=(
        "Propose a new interval for an existing job. "
        "The stored interval is what runs after confirm."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Job name as listed."},
            "id": {"type": "string", "description": "Job id if the name is ambiguous."},
            "every_seconds": {
                "type": "integer",
                "description": "New seconds between runs.",
            },
        },
        "required": ["every_seconds"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True)
class Job:
    """One persisted schedule or watch. Cancelled rows stay for audit."""

    id: str
    agent: str
    kind: JobKind
    name: str
    prompt: str
    every_seconds: int
    ttl_seconds: int | None
    created_at: float
    expires_at: float | None
    next_run_at: float
    last_run_at: float | None
    cancelled_at: float | None
    pending_text: str | None = None
    pending_untrusted: bool = False
    pending_after: Literal["advance", "cancel"] | None = None
    pending_delivered: bool = False


@dataclass(frozen=True)
class JobOutcome:
    """One report-only run. ``found`` is false when the model had nothing to say."""

    text: str
    found: bool
    untrusted: bool


@dataclass(frozen=True)
class JobCreate:
    kind: JobKind
    name: str
    prompt: str
    every_seconds: int
    ttl_seconds: int | None


@dataclass(frozen=True)
class JobReschedule:
    every_seconds: int
    name: str | None
    id: str | None


def new_job_id() -> str:
    return "j" + secrets.token_hex(4)


def parse_create(arguments: dict[str, Any]) -> JobCreate:
    kind = arguments.get("kind")
    if kind not in ("schedule", "watch"):
        raise JobError("kind must be schedule or watch")
    name = _parse_name(arguments.get("name"))
    prompt = _parse_prompt(arguments.get("prompt"))
    every_seconds = _parse_every(arguments.get("every_seconds"))
    ttl_seconds = _parse_ttl(arguments.get("ttl_seconds"), kind=kind, every_seconds=every_seconds)
    return JobCreate(
        kind=kind,
        name=name,
        prompt=prompt,
        every_seconds=every_seconds,
        ttl_seconds=ttl_seconds,
    )


def parse_reschedule(arguments: dict[str, Any]) -> JobReschedule:
    every_seconds = _parse_every(arguments.get("every_seconds"))
    name = arguments.get("name")
    job_id = arguments.get("id")
    parsed_name = _parse_name(name) if name is not None else None
    parsed_id = _parse_id(job_id) if job_id is not None else None
    if parsed_name is None and parsed_id is None:
        raise JobError("reschedule needs a name or id")
    return JobReschedule(every_seconds=every_seconds, name=parsed_name, id=parsed_id)


def parse_cancel(arguments: dict[str, Any]) -> tuple[str | None, str | None]:
    name = arguments.get("name")
    job_id = arguments.get("id")
    parsed_name = _parse_name(name) if name is not None else None
    parsed_id = _parse_id(job_id) if job_id is not None else None
    if parsed_name is None and parsed_id is None:
        raise JobError("cancel needs a name or id")
    return parsed_name, parsed_id


def is_nothing(text: str) -> bool:
    """Watches stay quiet on these; schedules still send a placeholder.

    The run prompt asks for the single word NONE, and models punctuate it, so
    trailing marks are ignored. ``no`` is left alone: it is a real answer.
    """
    return text.strip().rstrip(".!").strip().lower() in _NOTHING


def format_job_message(name: str, body: str) -> str:
    return f"[job: {name}]\n{body}"


def format_expire_message(name: str) -> str:
    return f"[job: {name}]\nThis watch ended. Nothing further will run unless you create it again."


def format_fail_message(name: str) -> str:
    return f"[job: {name}]\nThis check failed. I could not complete it."


def job_user_prompt(job: Job) -> str:
    return (
        f"Scheduled job '{job.name}': {job.prompt}\n\n"
        "Report only. Do not create jobs, relay, file, draft, or propose actions. "
        "You may search, fetch, and read the shared calendar; cite URLs. "
        "If you have nothing to report, reply with the single word NONE."
    )


def bind_jobs(store: Any) -> dict[str, Callable[[dict[str, Any]], Awaitable[str]]]:
    """Handlers closed over the live store. Bound when the channel starts."""

    async def create(arguments: dict[str, Any]) -> str:
        spec = parse_create(arguments)
        agent = active_agent()
        now = _now()
        if store.find_job(agent.name, name=spec.name) is not None:
            raise JobError(f"a job named {spec.name!r} already exists")
        expires_at = (now + spec.ttl_seconds) if spec.ttl_seconds is not None else None
        saved: Job | None = None
        last_error: StoreError | None = None
        for _ in range(5):
            job = Job(
                id=new_job_id(),
                agent=agent.name,
                kind=spec.kind,
                name=spec.name,
                prompt=spec.prompt,
                every_seconds=spec.every_seconds,
                ttl_seconds=spec.ttl_seconds,
                created_at=now,
                expires_at=expires_at,
                next_run_at=now + spec.every_seconds,
                last_run_at=None,
                cancelled_at=None,
            )
            try:
                store.save_job(job)
            except StoreError as exc:
                last_error = exc
                if store.find_job(agent.name, name=spec.name) is not None:
                    raise JobError(f"a job named {spec.name!r} already exists") from exc
                continue
            saved = job
            break
        if saved is None:
            raise last_error if last_error is not None else StoreError("job could not be saved")
        job = saved
        return json.dumps(
            {
                "ok": True,
                "id": job.id,
                "name": job.name,
                "kind": job.kind,
                "every_seconds": job.every_seconds,
                "next_run_at": job.next_run_at,
            },
            ensure_ascii=False,
        )

    async def list_jobs(_arguments: dict[str, Any]) -> str:
        agent = active_agent()
        jobs = store.list_jobs(agent.name)
        payload = [
            {
                "id": job.id,
                "name": job.name,
                "kind": job.kind,
                "every_seconds": job.every_seconds,
                "next_run_at": job.next_run_at,
                "expires_at": job.expires_at,
            }
            for job in jobs
        ]
        return json.dumps({"jobs": payload}, ensure_ascii=False)

    async def cancel(arguments: dict[str, Any]) -> str:
        agent = active_agent()
        name, job_id = parse_cancel(arguments)
        job = store.find_job(agent.name, name=name, job_id=job_id)
        if job is None:
            raise JobError("no matching job")
        store.cancel_job(job.id, at=_now())
        return json.dumps({"ok": True, "id": job.id, "name": job.name}, ensure_ascii=False)

    async def reschedule(arguments: dict[str, Any]) -> str:
        spec = parse_reschedule(arguments)
        agent = active_agent()
        job = store.find_job(agent.name, name=spec.name, job_id=spec.id)
        if job is None:
            raise JobError("no matching job")
        if job.ttl_seconds is not None and spec.every_seconds > job.ttl_seconds:
            raise JobError("every_seconds is longer than the watch TTL")
        now = _now()
        next_run_at = now + spec.every_seconds
        if not store.reschedule_job(
            job.id, every_seconds=spec.every_seconds, next_run_at=next_run_at
        ):
            raise JobError("no matching job")
        return json.dumps(
            {
                "ok": True,
                "id": job.id,
                "name": job.name,
                "every_seconds": spec.every_seconds,
                "next_run_at": next_run_at,
            },
            ensure_ascii=False,
        )

    return {
        JOB_CREATE: create,
        JOB_LIST: list_jobs,
        JOB_CANCEL: cancel,
        JOB_RESCHEDULE: reschedule,
    }


def _parse_name(raw: object) -> str:
    if not isinstance(raw, str):
        raise JobError("name must be a string")
    name = " ".join(raw.split())
    if not _NAME.fullmatch(name):
        raise JobError("name is not a usable job title")
    if len(name) > MAX_NAME_CHARS:
        raise JobError("name is too long")
    return name


def _parse_prompt(raw: object) -> str:
    if not isinstance(raw, str):
        raise JobError("prompt must be a string")
    prompt = raw.strip()
    if not prompt:
        raise JobError("prompt is empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise JobError("prompt is too long")
    return prompt


def _parse_every(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise JobError("every_seconds must be an integer")
    value = int(raw)
    if value != raw:
        raise JobError("every_seconds must be an integer")
    if value < MIN_EVERY_SECONDS or value > MAX_EVERY_SECONDS:
        raise JobError("every_seconds is out of range")
    return value


def _parse_ttl(raw: object, *, kind: JobKind, every_seconds: int) -> int | None:
    if kind == "schedule":
        if raw is None:
            return None
        raise JobError("a schedule has no TTL; cancel it instead")
    if raw is None:
        raise JobError("a watch requires ttl_seconds")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise JobError("ttl_seconds must be an integer")
    value = int(raw)
    if value != raw:
        raise JobError("ttl_seconds must be an integer")
    if value < every_seconds or value > MAX_TTL_SECONDS:
        raise JobError("ttl_seconds is out of range")
    return value


def _parse_id(raw: object) -> str:
    if not isinstance(raw, str) or not re.fullmatch(r"j[0-9a-f]{8}", raw.strip()):
        raise JobError("id is not a job id")
    return raw.strip()


def _now() -> float:
    return time.time()
