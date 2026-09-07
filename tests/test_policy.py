from __future__ import annotations

from pathlib import Path

import pytest

from assistai.signal.policy import MAX_PENDING, AccessPolicy


def _policy(tmp_path: Path, *, clock: list[float] | None = None) -> AccessPolicy:
    ticks = clock if clock is not None else [0.0]

    def now() -> float:
        return ticks[0]

    return AccessPolicy(
        ("+15555550101",),
        persist_path=tmp_path / "allow.json",
        pairing_ttl_seconds=10,
        clock=now,
    )


def test_bootstrap_is_allowed(tmp_path: Path) -> None:
    policy = _policy(tmp_path)

    assert policy.decide("+15555550101") == "allow"
    assert policy.decide("+15555550102") == "unknown"


def test_same_sender_keeps_the_same_code(tmp_path: Path) -> None:
    policy = _policy(tmp_path)

    first = policy.request_pair("+15555550102")
    second = policy.request_pair("+15555550102")

    assert first == second
    assert len(first) == 6


def test_approve_admits_and_persists(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    code = policy.request_pair("+15555550102")

    assert policy.approve(code, approver="+15555550101") == "+15555550102"
    assert policy.decide("+15555550102") == "allow"

    reloaded = AccessPolicy(
        ("+15555550101",),
        persist_path=tmp_path / "allow.json",
        pairing_ttl_seconds=10,
    )
    assert reloaded.decide("+15555550102") == "allow"


def test_paired_member_cannot_approve_others(tmp_path: Path) -> None:
    """Pairing grants access to the bot, never the authority to admit others."""
    policy = _policy(tmp_path)
    first = policy.request_pair("+15555550102")
    policy.approve(first, approver="+15555550101")
    second = policy.request_pair("+15555550103")

    with pytest.raises(PermissionError):
        policy.approve(second, approver="+15555550102")

    assert policy.decide("+15555550103") == "unknown"


def test_operator_status_is_env_only(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    code = policy.request_pair("+15555550102")
    policy.approve(code, approver="+15555550101")

    assert policy.is_operator("+15555550101") is True
    assert policy.is_operator("+15555550102") is False
    assert policy.allowed("+15555550102") is True


def test_expired_code_is_rejected(tmp_path: Path) -> None:
    ticks = [0.0]
    policy = _policy(tmp_path, clock=ticks)
    code = policy.request_pair("+15555550102")
    ticks[0] = 11

    assert policy.approve(code, approver="+15555550101") is None
    assert policy.decide("+15555550102") == "unknown"


def test_pending_pairs_are_bounded(tmp_path: Path) -> None:
    policy = _policy(tmp_path)

    for i in range(300):
        policy.request_pair(f"+1555555{i:04d}")

    assert len(policy._pending) <= MAX_PENDING


def test_parse_approve_requires_the_whole_message() -> None:
    policy = AccessPolicy((), persist_path=Path("/unused"), pairing_ttl_seconds=10)

    assert policy.parse_approve("/approve 123456") == "123456"
    assert policy.parse_approve("/pair 000001") == "000001"
    assert policy.parse_approve("please /approve 123456") is None
    assert policy.parse_approve("/approve 12345") is None
