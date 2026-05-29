"""Tamper-evident request audit log.

Each request produces one JSON line. Records are hash-chained: every line
carries the hash of the previous line, and its own hash covers that previous
hash plus the record body. Editing or deleting any historical record breaks the
chain from that point on, so tampering is detectable even by someone with write
access to the file (`verify_chain`).

The log is intended to be kept internal. By design it stores no raw client PII
at the default content level — only metadata plus SHA-256 digests of the input
and the (already PII-free) anonymized output, which is enough to prove what was
processed without turning the audit trail itself into a confidential-data store.

Immutability is bounded by retention duties, not infinite: logs rotate into
period segments (e.g. one file per month) so an old segment can be destroyed
under a retention schedule without invalidating the active chain. Each segment
is an independent chain. Suspend rotation/purge while under legal hold.
"""

import asyncio
import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

GENESIS = "GENESIS"


@dataclass
class AuditRecord:
    """One request's audit envelope.

    Content fields (`input_sha256`, `output_sha256`, `input`, `output`) are
    populated according to the configured content level. `matter_id` and `actor`
    come from request headers when enabled. `seq`, `ts`, `prev`, and `hash` are
    set by the log on append and must not be supplied by callers.
    """

    request_id: str
    route: str
    status: str
    matter_id: str | None = None
    actor: str | None = None
    client_ip: str | None = None
    provider: str | None = None
    model: str | None = None
    anonymization_mode: str | None = None
    safety_net_passed: bool | None = None
    entities_detected: int | None = None
    sensitivity: str | None = None
    # Per-entity anonymization decisions. At the `metadata` content level each
    # entry carries entity_type/action/reason only (no raw entity text); at
    # `full` it may also include the original text and its replacement.
    decisions: Any = None
    error: str | None = None
    external_latency_ms: int | None = None
    total_ms: int | None = None
    input_sha256: str | None = None
    output_sha256: str | None = None
    input: Any = None
    output: Any = None
    # Chain fields, assigned by AuditLog.append.
    seq: int | None = None
    ts: str | None = None
    prev: str | None = None
    hash: str | None = None


def _canonical(record: dict[str, Any]) -> str:
    """Deterministic serialization of the record body, excluding its own hash.

    `hash` is excluded because it is derived from this string; `prev` is kept in
    the body so the chain link is itself covered by the hash.
    """
    body = {k: v for k, v in record.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(prev: str, body: str, hmac_key: str | None) -> str:
    payload = (prev + body).encode("utf-8")
    if hmac_key:
        return hmac.new(hmac_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    records_checked: int
    first_bad_seq: int | None = None
    reason: str | None = None


def verify_chain(path: str, hmac_key: str | None = None) -> VerifyResult:
    """Walk a segment file and confirm every link hashes correctly.

    Returns the sequence number of the first record whose stored hash does not
    match its recomputed hash, or whose `prev` does not point at the prior line.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audit log not found: {path}")

    prev = GENESIS
    checked = 0
    expected_seq = 1
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return VerifyResult(
                    ok=False,
                    records_checked=checked,
                    first_bad_seq=None,
                    reason=f"line {line_number} is not valid JSON",
                )

            seq = record.get("seq")
            if record.get("prev") != prev:
                return VerifyResult(
                    ok=False,
                    records_checked=checked,
                    first_bad_seq=seq,
                    reason=f"prev hash mismatch at seq {seq}",
                )
            if seq != expected_seq:
                return VerifyResult(
                    ok=False,
                    records_checked=checked,
                    first_bad_seq=seq,
                    reason=f"expected seq {expected_seq}, found {seq}",
                )

            recomputed = compute_hash(prev, _canonical(record), hmac_key)
            if recomputed != record.get("hash"):
                return VerifyResult(
                    ok=False,
                    records_checked=checked,
                    first_bad_seq=seq,
                    reason=f"hash mismatch at seq {seq}",
                )

            prev = record["hash"]
            checked += 1
            expected_seq += 1

    return VerifyResult(ok=True, records_checked=checked)


def _segment_suffix(rotation: str, now: datetime) -> str:
    if rotation == "daily":
        return now.strftime("%Y-%m-%d")
    if rotation == "monthly":
        return now.strftime("%Y-%m")
    return ""


def segment_path(destination: str, rotation: str, now: datetime) -> Path:
    """Resolve the file a record written at `now` belongs in.

    With rotation off the destination is used verbatim; otherwise the period is
    inserted before the suffix (audit.jsonl -> audit-2026-05.jsonl).
    """
    base = Path(destination)
    suffix = _segment_suffix(rotation, now)
    if not suffix:
        return base
    stem = base.stem
    extension = base.suffix or ".jsonl"
    return base.with_name(f"{stem}-{suffix}{extension}")


def _last_chain_state(path: Path) -> tuple[int, str]:
    """Recover (seq, hash) from the final record of an existing segment.

    Fails loud if the last line is unreadable rather than silently starting a
    fresh chain over a corrupt file.
    """
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return 0, GENESIS
    try:
        record = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Audit log {path} ends with a non-JSON line; refusing to append "
            "to a corrupt chain"
        ) from exc
    seq = record.get("seq")
    chain_hash = record.get("hash")
    if not isinstance(seq, int) or not isinstance(chain_hash, str):
        raise ValueError(
            f"Audit log {path} last record is missing seq/hash; refusing to append"
        )
    return seq, chain_hash


@dataclass
class AuditLog:
    """Append-only hash-chained sink. One instance per running gateway.

    `destination` is either "stdout" or a file path. With a file destination and
    rotation enabled, each period gets its own segment file and its own chain;
    chain state is tracked per segment so a new month starts cleanly from
    GENESIS while older segments stay independently verifiable.
    """

    destination: str
    hmac_key: str | None = None
    rotation: str = "none"
    _segments: dict[str, tuple[int, str]] = field(default_factory=dict)
    _lock: "asyncio.Lock | None" = None

    @property
    def to_stdout(self) -> bool:
        return self.destination == "stdout"

    async def record(self, record: AuditRecord, now: datetime) -> dict:
        """Async-safe append. Serializes concurrent writers behind one lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await asyncio.to_thread(self.append, record, now)

    def _line(self, record: AuditRecord, prev: str, seq: int, now: datetime) -> dict:
        record.seq = seq
        record.prev = prev
        record.ts = now.isoformat()
        body = {k: v for k, v in asdict(record).items() if k != "hash"}
        record.hash = compute_hash(prev, _canonical(body), self.hmac_key)
        body["hash"] = record.hash
        return body

    def append(self, record: AuditRecord, now: datetime) -> dict:
        """Write one record, returning the serialized line that was emitted.

        Synchronous and self-contained: opens, appends with O_APPEND semantics,
        flushes, and fsyncs so a crash cannot lose an acknowledged record. The
        caller serializes concurrent appends.
        """
        if self.to_stdout:
            seq, prev = self._segments.get("stdout", (0, GENESIS))
            body = self._line(record, prev, seq + 1, now)
            self._segments["stdout"] = (seq + 1, body["hash"])
            print(json.dumps(body, ensure_ascii=False), flush=True)
            return body

        path = segment_path(self.destination, self.rotation, now)
        key = str(path)
        if key not in self._segments:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                self._segments[key] = _last_chain_state(path)
            else:
                self._segments[key] = (0, GENESIS)

        seq, prev = self._segments[key]
        body = self._line(record, prev, seq + 1, now)
        line = json.dumps(body, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._segments[key] = (seq + 1, body["hash"])
        return body
