"""Command-line tools for the Hey Jude audit log.

    hey-jude audit verify <path> [--hmac-key KEY]
    hey-jude audit query  <path> [--matter M] [--actor A] [--status S] [--since TS]

`verify` walks a segment's hash chain and exits non-zero on the first broken
link. `query` filters records for conflict checks, client audits, or discovery
production. Run `verify` against every segment that `query` reads from before
relying on its output.
"""

import argparse
import json
import sys

from hey_jude.services.audit import verify_chain


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_chain(args.path, hmac_key=args.hmac_key)
    if result.ok:
        print(f"OK: {result.records_checked} records, chain intact")
        return 0
    print(
        f"BROKEN: chain failed at seq {result.first_bad_seq}: {result.reason} "
        f"({result.records_checked} records verified before the break)",
        file=sys.stderr,
    )
    return 1


def _matches(record: dict, args: argparse.Namespace) -> bool:
    if args.matter is not None and record.get("matter_id") != args.matter:
        return False
    if args.actor is not None and record.get("actor") != args.actor:
        return False
    if args.status is not None and record.get("status") != args.status:
        return False
    if args.since is not None:
        ts = record.get("ts")
        if ts is None or ts < args.since:
            return False
    return True


def _cmd_query(args: argparse.Namespace) -> int:
    matched = 0
    with open(args.path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if _matches(record, args):
                print(json.dumps(record, ensure_ascii=False))
                matched += 1
    if matched == 0:
        print("No matching records.", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hey-jude")
    sub = parser.add_subparsers(dest="group", required=True)

    audit = sub.add_parser("audit", help="Audit log tools")
    audit_sub = audit.add_subparsers(dest="command", required=True)

    verify = audit_sub.add_parser("verify", help="Verify a segment's hash chain")
    verify.add_argument("path", help="Path to an audit segment (.jsonl)")
    verify.add_argument(
        "--hmac-key",
        default=None,
        help="HMAC key, if the log was written with audit_hmac_key set",
    )
    verify.set_defaults(func=_cmd_verify)

    query = audit_sub.add_parser("query", help="Filter audit records")
    query.add_argument("path", help="Path to an audit segment (.jsonl)")
    query.add_argument("--matter", default=None, help="Match matter_id exactly")
    query.add_argument("--actor", default=None, help="Match actor exactly")
    query.add_argument("--status", default=None, help="Match status exactly")
    query.add_argument(
        "--since",
        default=None,
        help="Only records with ts >= this ISO-8601 timestamp",
    )
    query.set_defaults(func=_cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
