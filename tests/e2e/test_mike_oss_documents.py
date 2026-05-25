#!/usr/bin/env python3
"""
End-to-end Mike OSS + Hey Jude document-format test.

This is intentionally opt-in because it starts local services and calls Gemini.

Direct CLI:
    GEMINI_API_KEY=... python3 tests/e2e/test_mike_oss_documents.py

Pytest:
    RUN_MIKE_OSS_E2E=1 GEMINI_API_KEY=... python3 -m pytest tests/e2e/test_mike_oss_documents.py -s

What it covers:
    1. Starts/reuses a reduced local Supabase stack.
    2. Applies Mike's schema and creates a local auth user.
    3. Starts Hey Jude with Gemini as anonymizer and destination.
    4. Starts Mike with HEY_JUDE_ENABLED=true.
    5. Sends all supported text-readable document types directly to Hey Jude.
    6. Sends each document's extracted text through Mike's authenticated chat route.

Mike's current chat API sends text to the model, not OpenAI-style structured file
parts. So Mike coverage verifies the Hey Jude gateway path for document-derived
text. Hey Jude direct coverage verifies actual file part extraction.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from typing import Iterator
from urllib.request import Request, urlopen

import httpx
import pytest
from docx import Document


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIKE_DIR = ROOT.parent / "mike-oss"
SUPABASE_EXCLUDES = ",".join(
    [
        "realtime",
        "storage-api",
        "imgproxy",
        "mailpit",
        "postgres-meta",
        "studio",
        "edge-runtime",
        "logflare",
        "vector",
        "supavisor",
    ]
)


@dataclass
class DocumentCase:
    name: str
    filename: str
    media_type: str
    raw: bytes
    person: str
    company: str
    topic: str

    @property
    def plain_text(self) -> str:
        return f"{self.person} at {self.company} asks about {self.topic}."


@dataclass
class SupabaseEnv:
    api_url: str
    db_url: str
    publishable_key: str
    secret_key: str
    token: str
    user_id: str


def b64_data_url(media_type: str, raw: bytes) -> str:
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def minimal_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
    objects.append(
        f"5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj\n"
    )
    content = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(content.encode("latin-1")))
        content += obj
    startxref = len(content.encode("latin-1"))
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        content += f"{offset:010d} 00000 n \n"
    content += (
        f"trailer << /Root 1 0 R /Size {len(objects) + 1} >>\n"
        f"startxref\n{startxref}\n%%EOF\n"
    )
    return content.encode("latin-1")


def docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def eml_bytes(text: str) -> bytes:
    message = EmailMessage()
    message["Subject"] = "Synthetic legal document"
    message["From"] = "sender@example.test"
    message["To"] = "recipient@example.test"
    message.set_content(text)
    return message.as_bytes()


def document_cases() -> list[DocumentCase]:
    rows = [
        ("txt", "matter.txt", "text/plain", "Tessa Text", "TextCo Holdings", "a licensing dispute"),
        ("md", "matter.md", "text/markdown", "Mira Markdown", "Markdown Ventures", "an NDA review"),
        ("html", "matter.html", "text/html", "Hannah Html", "Html Group", "a merger covenant"),
        ("eml", "matter.eml", "message/rfc822", "Evan Email", "Email Partners", "a privilege review"),
        ("rtf", "matter.rtf", "application/rtf", "Riley Richtext", "Richtext LLP", "a lease dispute"),
        ("docx", "matter.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Dana Docx", "Docx Capital", "an indemnity clause"),
        ("pdf", "matter.pdf", "application/pdf", "Paula Pdf", "Pdf Industries", "a settlement term"),
    ]
    cases = []
    for name, filename, media_type, person, company, topic in rows:
        text = f"{person} at {company} asks about {topic}."
        if name == "html":
            raw = f"<html><body><p>{text}</p></body></html>".encode()
        elif name == "eml":
            raw = eml_bytes(text)
        elif name == "rtf":
            raw = (r"{\rtf1\ansi " + text + "}").encode()
        elif name == "docx":
            raw = docx_bytes(text)
        elif name == "pdf":
            raw = minimal_pdf(text)
        else:
            raw = text.encode()
        cases.append(DocumentCase(name, filename, media_type, raw, person, company, topic))
    return cases


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
        check=True,
    )


def wait_http(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(Request(url), timeout=2) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def wait_port_http_post(url: str, headers: dict[str, str], payload: dict, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=5)
            if response.status_code < 500:
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for POST {url}")


@contextlib.contextmanager
def process(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> Iterator[subprocess.Popen[str]]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def supabase_status(workdir: Path) -> dict[str, str]:
    status = run(["supabase", "status", "-o", "json"], cwd=workdir, timeout=60)
    start = status.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"Could not parse supabase status:\n{status.stdout}")
    return json.loads(status.stdout[start:])


def ensure_supabase(workdir: Path, mike_dir: Path) -> dict[str, str]:
    workdir.mkdir(parents=True, exist_ok=True)
    if not (workdir / "supabase" / "config.toml").exists():
        run(["supabase", "init"], cwd=workdir, timeout=60)
    try:
        run(
            ["supabase", "start", "-x", SUPABASE_EXCLUDES],
            cwd=workdir,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        # If it is already running, status below will still prove that.
        if "already running" not in (exc.stdout or "").lower():
            raise
    status = supabase_status(workdir)
    schema = mike_dir / "backend" / "schema.sql"
    run(["psql", status["DB_URL"], "-v", "ON_ERROR_STOP=1", "-f", str(schema)], cwd=ROOT)
    return status


def create_local_user(api_url: str, publishable_key: str) -> tuple[str, str]:
    email = f"mike-e2e-{int(time.time())}@example.test"
    response = httpx.post(
        f"{api_url}/auth/v1/signup",
        headers={"apikey": publishable_key},
        json={"email": email, "password": "local-test-password-123"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["access_token"], data["user"]["id"]


def setup_supabase(args: argparse.Namespace) -> SupabaseEnv:
    status = ensure_supabase(args.supabase_workdir, args.mike_oss_dir)
    token, user_id = create_local_user(status["API_URL"], status["PUBLISHABLE_KEY"])
    return SupabaseEnv(
        api_url=status["API_URL"],
        db_url=status["DB_URL"],
        publishable_key=status["PUBLISHABLE_KEY"],
        secret_key=status.get("SECRET_KEY") or status["SERVICE_ROLE_KEY"],
        token=token,
        user_id=user_id,
    )


def start_hey_jude(args: argparse.Namespace) -> contextlib.AbstractContextManager[subprocess.Popen[str]]:
    env = {
        "GEMINI_API_KEY": args.gemini_api_key,
        "LOCAL_LLM_URL": "https://generativelanguage.googleapis.com/v1beta/openai",
        "LOCAL_LLM_MODEL": args.local_llm_model,
        "LOCAL_LLM_API_KEY": args.gemini_api_key,
        "EXTERNAL_LLM_MODEL": args.external_llm_model,
        "EXTERNAL_LLM_API_BASE": "",
        "API_KEY": args.hey_jude_api_key,
        "DOCUMENT_UNREADABLE_ACTION": "reject",
        "SAFETY_NET_STRICTNESS": "off",
        "PYTHONPATH": str(ROOT / "src"),
    }
    return process(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "hey_jude.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.hey_jude_port),
        ],
        cwd=ROOT,
        env=env,
    )


def start_mike(args: argparse.Namespace, supabase: SupabaseEnv) -> contextlib.AbstractContextManager[subprocess.Popen[str]]:
    env = {
        "PORT": str(args.mike_port),
        "SUPABASE_URL": supabase.api_url,
        "SUPABASE_SECRET_KEY": supabase.secret_key,
        "HEY_JUDE_ENABLED": "true",
        "HEY_JUDE_BASE_URL": f"http://127.0.0.1:{args.hey_jude_port}",
        "HEY_JUDE_API_KEY": args.hey_jude_api_key,
        "GEMINI_API_KEY": args.gemini_api_key,
    }
    return process(["npm", "run", "dev", "--prefix", "backend"], cwd=args.mike_oss_dir, env=env)


def assert_contains_response_text(response_text: str, expected: str, label: str) -> None:
    if expected not in response_text:
        raise AssertionError(f"{label}: expected {expected!r} in response:\n{response_text}")


def run_hey_jude_direct(args: argparse.Namespace, cases: list[DocumentCase]) -> None:
    client = httpx.Client(timeout=180)
    headers = {
        "Authorization": f"Bearer {args.hey_jude_api_key}",
        "Content-Type": "application/json",
    }
    for case in cases:
        expected = f"{case.person} at {case.company}"
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Read the attachment. Reply in one sentence and repeat "
                                f"the person and company exactly: {expected}."
                            ),
                        },
                        {
                            "type": "input_file",
                            "filename": case.filename,
                            "file_data": b64_data_url(case.media_type, case.raw),
                        },
                    ],
                }
            ],
        }
        response = client.post(
            f"http://127.0.0.1:{args.hey_jude_port}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        meta = data["heyjude_metadata"]
        if meta["status"] != "completed":
            raise AssertionError(f"{case.name}: unexpected metadata {meta}")
        if meta["entities_detected"] < 1:
            raise AssertionError(f"{case.name}: no entities detected in {meta}")
        assert_contains_response_text(content, expected, f"hey-jude {case.name}")
        print(f"hey-jude {case.name}: ok ({meta['entities_detected']} entities)")


def parse_sse_text(body: str) -> str:
    parts: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        data = json.loads(raw)
        if data.get("type") == "content_delta":
            parts.append(data.get("text", ""))
    return "".join(parts)


def run_mike_chat(args: argparse.Namespace, supabase: SupabaseEnv, cases: list[DocumentCase]) -> None:
    client = httpx.Client(timeout=240)
    headers = {
        "Authorization": f"Bearer {supabase.token}",
        "Content-Type": "application/json",
    }
    for case in cases:
        expected = f"{case.person} at {case.company}"
        payload = {
            "model": args.mike_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"MIKE_DOC_{case.name.upper()}_TEST: {case.plain_text} "
                        "This is a connectivity test, not a legal analysis task. "
                        f"Return exactly this string and nothing else: {expected}"
                    ),
                }
            ],
        }
        response = client.post(
            f"http://localhost:{args.mike_port}/chat",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        text = parse_sse_text(response.text)
        assert_contains_response_text(text, expected, f"mike {case.name}")
        print(f"mike {case.name}: ok")

    verify = run(
        [
            "psql",
            supabase.db_url,
            "-P",
            "pager=off",
            "-t",
            "-c",
            "select count(*) from public.chat_messages;",
        ],
        cwd=ROOT,
    )
    count = int(verify.stdout.strip())
    minimum = len(cases) * 2
    if count < minimum:
        raise AssertionError(f"Expected at least {minimum} persisted messages, found {count}")
    print(f"mike persistence: ok ({count} chat_messages)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--mike-oss-dir", type=Path, default=DEFAULT_MIKE_DIR)
    parser.add_argument("--supabase-workdir", type=Path, default=Path("/private/tmp/mike-oss-supabase"))
    parser.add_argument("--hey-jude-port", type=int, default=4005)
    parser.add_argument("--mike-port", type=int, default=3001)
    parser.add_argument("--hey-jude-api-key", default="sk-heyjude-dev")
    parser.add_argument("--local-llm-model", default="gemini-2.5-flash")
    parser.add_argument("--external-llm-model", default="gemini/gemini-2.5-flash-lite")
    parser.add_argument("--mike-model", default="gemini-3-flash-preview")
    parser.add_argument("--leave-supabase-running", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gemini_api_key:
        raise SystemExit("Set GEMINI_API_KEY or pass --gemini-api-key")
    if not args.mike_oss_dir.exists():
        raise SystemExit(f"Mike OSS directory not found: {args.mike_oss_dir}")
    if not shutil.which("supabase"):
        raise SystemExit("supabase CLI is required")
    if not shutil.which("psql"):
        raise SystemExit("psql is required")

    cases = document_cases()
    supabase = setup_supabase(args)
    print(f"local supabase: {supabase.api_url}")

    with start_hey_jude(args):
        wait_http(f"http://127.0.0.1:{args.hey_jude_port}/health", timeout=90)
        run_hey_jude_direct(args, cases)

        with start_mike(args, supabase):
            wait_http(f"http://localhost:{args.mike_port}/health", timeout=90)
            run_mike_chat(args, supabase, cases)

    if not args.leave_supabase_running:
        run(["supabase", "stop"], cwd=args.supabase_workdir, timeout=120)
    return 0


@pytest.mark.e2e
def test_mike_oss_documents_e2e() -> None:
    if os.environ.get("RUN_MIKE_OSS_E2E") != "1":
        pytest.skip("Set RUN_MIKE_OSS_E2E=1 and GEMINI_API_KEY to run")
    assert main([]) == 0


if __name__ == "__main__":
    raise SystemExit(main())
