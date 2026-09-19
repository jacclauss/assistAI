from __future__ import annotations

from assistai.signal.envelopes import parse_inbound
from tests.signal_fakes import envelope


def test_reads_rest_api_wrapper() -> None:
    inbound = parse_inbound(envelope(sender="+15555550101", text="hello"))

    assert inbound is not None
    assert inbound.sender == "+15555550101"
    assert inbound.text == "hello"


def test_reads_uuid_when_number_privacy_hides_e164() -> None:
    inbound = parse_inbound(envelope(sender="429cce0e-9174-4d7a-a98b-1cb9208b1951", text="hello"))

    assert inbound is not None
    assert inbound.sender == "429cce0e-9174-4d7a-a98b-1cb9208b1951"
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


def test_notes_attachments_without_keeping_bytes() -> None:
    inbound = parse_inbound(
        {
            "envelope": {
                "sourceNumber": "+15555550101",
                "timestamp": 1,
                "dataMessage": {
                    "message": "see this",
                    "timestamp": 1,
                    "attachments": [
                        {"filename": "list.pdf", "contentType": "application/pdf", "size": 1200}
                    ],
                },
            }
        }
    )

    assert inbound is not None
    assert inbound.text == "see this"
    assert inbound.attachments[0].name == "list.pdf"
    assert inbound.attachments[0].content_type == "application/pdf"
    assert inbound.attachments[0].size == 1200


def test_attachment_only_is_still_a_dm() -> None:
    inbound = parse_inbound(
        {
            "envelope": {
                "sourceNumber": "+15555550101",
                "timestamp": 1,
                "dataMessage": {
                    "attachments": [
                        {"filename": "photo.jpg", "contentType": "image/jpeg", "size": 8}
                    ],
                },
            }
        }
    )

    assert inbound is not None
    assert inbound.text == ""
    assert inbound.attachments[0].name == "photo.jpg"


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
