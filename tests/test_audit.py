from datetime import datetime, timezone

import pytest

from hey_jude.cli import main as cli_main
from hey_jude.models import ChatMessage, FoundEntity
from hey_jude.routes import _record_decisions, _record_input, _record_output
from hey_jude.services.audit import (
    GENESIS,
    AuditLog,
    AuditRecord,
    segment_path,
    verify_chain,
)


def _now(day=1, hour=12):
    return datetime(2026, 5, day, hour, 0, 0, tzinfo=timezone.utc)


def _record(request_id="r1", status="completed"):
    return AuditRecord(request_id=request_id, route="chat_completions", status=status)


def _read_lines(path):
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_chain_links_each_record_to_previous(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path), rotation="none")
    first = log.append(_record("r1"), _now())
    second = log.append(_record("r2"), _now())

    assert first["seq"] == 1
    assert first["prev"] == GENESIS
    assert second["seq"] == 2
    assert second["prev"] == first["hash"]


def test_verify_chain_accepts_untampered_log(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path))
    for i in range(5):
        log.append(_record(f"r{i}"), _now())

    result = verify_chain(str(path))
    assert result.ok
    assert result.records_checked == 5


def test_verify_chain_detects_tampered_middle_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path))
    for i in range(3):
        log.append(_record(f"r{i}"), _now())

    lines = path.read_text().splitlines()
    tampered = lines[1].replace('"r1"', '"HACKED"')
    assert tampered != lines[1]
    lines[1] = tampered
    path.write_text("\n".join(lines) + "\n")

    result = verify_chain(str(path))
    assert not result.ok
    assert result.first_bad_seq == 2


def test_verify_chain_detects_deleted_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path))
    for i in range(3):
        log.append(_record(f"r{i}"), _now())

    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")

    result = verify_chain(str(path))
    assert not result.ok


def test_hmac_chain_requires_key_to_verify(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path), hmac_key="secret")
    log.append(_record("r1"), _now())

    assert verify_chain(str(path), hmac_key="secret").ok
    assert not verify_chain(str(path)).ok
    assert not verify_chain(str(path), hmac_key="wrong").ok


def test_reopening_recovers_chain_state(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(destination=str(path)).append(_record("r1"), _now())

    reopened = AuditLog(destination=str(path))
    second = reopened.append(_record("r2"), _now())

    assert second["seq"] == 2
    assert verify_chain(str(path)).ok


def test_reopening_corrupt_last_line_refuses_append(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(destination=str(path)).append(_record("r1"), _now())
    path.write_text(path.read_text() + "this is not json\n")

    with pytest.raises(ValueError, match="corrupt chain"):
        AuditLog(destination=str(path)).append(_record("r2"), _now())


def test_monthly_rotation_writes_independent_segments(tmp_path):
    dest = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(dest), rotation="monthly")
    log.append(_record("may"), datetime(2026, 5, 10, tzinfo=timezone.utc))
    log.append(_record("jun"), datetime(2026, 6, 10, tzinfo=timezone.utc))

    may = tmp_path / "audit-2026-05.jsonl"
    jun = tmp_path / "audit-2026-06.jsonl"
    assert may.exists() and jun.exists()
    # Each segment is its own chain starting from GENESIS.
    assert _read_lines(may)[0]["prev"] == GENESIS
    assert _read_lines(jun)[0]["prev"] == GENESIS
    assert _read_lines(jun)[0]["seq"] == 1
    assert verify_chain(str(may)).ok
    assert verify_chain(str(jun)).ok


def test_segment_path_no_rotation_is_verbatim():
    assert str(segment_path("a/audit.jsonl", "none", _now())) == "a/audit.jsonl"


def test_segment_path_daily_inserts_date():
    path = segment_path("a/audit.jsonl", "daily", _now(day=3))
    assert str(path) == "a/audit-2026-05-03.jsonl"


async def test_record_async_wrapper_appends(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path))
    body = await log.record(_record("r1"), _now())
    assert body["seq"] == 1
    assert verify_chain(str(path)).ok


# --- content tiers (routes helpers) ---


def _msgs():
    return [ChatMessage(role="user", content="Acme Corp client matter")]


def test_metadata_tier_stores_only_hashes():
    rec = _record()
    _record_input(rec, _msgs(), "metadata")
    _record_output(rec, _msgs(), "metadata")
    assert rec.input_sha256 and rec.output_sha256
    assert rec.input is None
    assert rec.output is None


def test_anonymized_tier_stores_output_not_input():
    rec = _record()
    _record_input(rec, _msgs(), "anonymized")
    _record_output(rec, _msgs(), "anonymized")
    assert rec.input is None
    assert rec.output == [{"role": "user", "content": "Acme Corp client matter"}]


def test_full_tier_stores_input_and_output():
    rec = _record()
    _record_input(rec, _msgs(), "full")
    _record_output(rec, _msgs(), "full")
    assert rec.input == [{"role": "user", "content": "Acme Corp client matter"}]
    assert rec.output == [{"role": "user", "content": "Acme Corp client matter"}]


# --- per-entity decisions ---


def _found():
    return [
        FoundEntity(
            text="Acme Corp",
            entity_type="ORGANIZATION",
            action="replace",
            replacement="COMPANY_01",
            reason="real company name",
        ),
        FoundEntity(
            text="the Agreement",
            entity_type="MISC",
            action="keep",
            replacement=None,
            reason="generic defined term",
        ),
    ]


def test_decisions_metadata_tier_omits_raw_text():
    rec = _record()
    _record_decisions(rec, _found(), "metadata")
    assert rec.decisions == [
        {"entity_type": "ORGANIZATION", "action": "replace", "reason": "real company name"},
        {"entity_type": "MISC", "action": "keep", "reason": "generic defined term"},
    ]
    serialized = str(rec.decisions)
    assert "Acme Corp" not in serialized
    assert "COMPANY_01" not in serialized


def test_decisions_full_tier_includes_text_and_replacement():
    rec = _record()
    _record_decisions(rec, _found(), "full")
    assert rec.decisions[0] == {
        "entity_type": "ORGANIZATION",
        "action": "replace",
        "reason": "real company name",
        "text": "Acme Corp",
        "replacement": "COMPANY_01",
    }


def test_decisions_survive_the_hash_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    rec = _record()
    _record_decisions(rec, _found(), "metadata")
    body = AuditLog(destination=str(path)).append(rec, _now())
    assert body["decisions"][0]["action"] == "replace"
    assert verify_chain(str(path)).ok


# --- CLI ---


def test_cli_verify_ok(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    AuditLog(destination=str(path)).append(_record("r1"), _now())
    assert cli_main(["audit", "verify", str(path)]) == 0
    assert "chain intact" in capsys.readouterr().out


def test_cli_verify_detects_tampering(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path))
    log.append(_record("r1"), _now())
    log.append(_record("r2"), _now())
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"r1"', '"X"')
    path.write_text("\n".join(lines) + "\n")

    assert cli_main(["audit", "verify", str(path)]) == 1
    assert "BROKEN" in capsys.readouterr().err


def test_cli_query_filters_by_matter(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(destination=str(path))
    a = _record("r1")
    a.matter_id = "M-100"
    b = _record("r2")
    b.matter_id = "M-200"
    log.append(a, _now())
    log.append(b, _now())

    assert cli_main(["audit", "query", str(path), "--matter", "M-100"]) == 0
    out = capsys.readouterr().out
    assert "r1" in out
    assert "r2" not in out
