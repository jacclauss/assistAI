from __future__ import annotations

from assistai.signal.envelopes import parse_inbound
from tests.signal_fakes import envelope


def test_reads_rest_api_wrapper() -> None:
    inbound = parse_inbound(envelope(sender="+15555550101", text="hello"))

    assert inbound is not None
    assert inbound.sender == "+15555550101"
    assert inbound.text == "hello"


def test_reads_jsonrpc_params() -> None:
    inbound = parse_inbound(
        {
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {"envelope": envelope(text="from rpc")["envelope"]},
        }
    )

    assert inbound is not None
    assert inbound.text == "from rpc"


def test_reads_jsonrpc_result_wrapper() -> None:
    inbound = parse_inbound(
        {"params": {"result": {"envelope": envelope(text="nested")["envelope"]}}}
    )

    assert inbound is not None
    assert inbound.text == "nested"


def test_prefers_source_number() -> None:
    inbound = parse_inbound(
        {
            "envelope": {
                "source": "+15555550999",
                "sourceNumber": "+15555550101",
                "timestamp": 1,
                "dataMessage": {"message": "hi", "timestamp": 1},
            }
        }
    )

    assert inbound is not None
    assert inbound.sender == "+15555550101"


def test_ignores_receipts_and_typing() -> None:
    assert (
        parse_inbound({"envelope": {"sourceNumber": "+15555550101", "receiptMessage": {}}}) is None
    )
    assert (
        parse_inbound({"envelope": {"sourceNumber": "+15555550101", "typingMessage": {}}}) is None
    )


def test_ignores_group_messages() -> None:
    assert parse_inbound(envelope(group=True)) is None


def test_ignores_empty_text() -> None:
    assert parse_inbound(envelope(text="   ")) is None


def test_reads_edits() -> None:
    inbound = parse_inbound(
        {
            "envelope": {
                "sourceNumber": "+15555550101",
                "timestamp": 2,
                "editMessage": {"dataMessage": {"message": "corrected", "timestamp": 2}},
            }
        }
    )

    assert inbound is not None
    assert inbound.text == "corrected"


def test_ignores_malformed() -> None:
    assert parse_inbound(None) is None
    assert parse_inbound("not json") is None
    assert (
        parse_inbound(
            {"envelope": {"sourceNumber": "not-a-number", "dataMessage": {"message": "x"}}}
        )
        is None
    )
