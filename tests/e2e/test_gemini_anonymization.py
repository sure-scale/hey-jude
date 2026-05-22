#!/usr/bin/env python3
"""
End-to-end test of the LLM-first anonymization pipeline.

Three-model architecture:
  1. Ollama (local)       — anonymization (PII detection + replacement)
  2. Gemini Flash (cloud)  — destination LLM (receives anonymized request)
  3. Gemini Pro (cloud)    — evaluator (judges anonymization quality)

Usage:
    GEMINI_API_KEY=your-key python tests/e2e/test_gemini_anonymization.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hey_jude.config import Settings
from hey_jude.models import ChatMessage
from hey_jude.services.anonymizer import anonymize_messages

from fixtures.legal_docs.download import load_test_cases

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_EVAL_MODEL = os.environ.get("GEMINI_EVAL_MODEL", "gemini-pro-latest")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


async def call_gemini(messages: list[dict], *, model: str = GEMINI_MODEL) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(GEMINI_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


# --- Destination test ---

async def test_destination(sanitized_messages: list[ChatMessage]) -> dict:
    """Forward anonymized messages to Gemini Flash and check the response."""
    messages = [{"role": m.role, "content": m.content} for m in sanitized_messages]
    try:
        response = await call_gemini(messages, model=GEMINI_MODEL)
        return {"ok": True, "response": response, "error": None}
    except Exception as e:
        return {"ok": False, "response": None, "error": f"{type(e).__name__}: {e}"}


# --- Evaluation (Gemini Pro as judge) ---

EVAL_PROMPT = """\
You are an expert PII anonymization evaluator. You will be given:
1. The ORIGINAL text containing real PII
2. The SANITIZED text with PII replaced by placeholders
3. The MAPPING of original values to their replacements

Evaluate the anonymization quality on three criteria. For each, provide a score (0-10) and a brief explanation.

## Criteria

### 1. PII Leak Detection
Scan the SANITIZED text for any remaining real PII that was NOT replaced. Check for:
- Real person names, company names, organization names
- Email addresses, phone numbers, physical addresses
- SSNs, EINs, account numbers
- Any other identifying information from the original

Score 10 = no leaks found. Score 0 = major PII left in plaintext.

### 2. Semantic Coherence
Does the sanitized text still read logically? Are placeholders used consistently?
Can a reader understand the document structure and legal relationships?

Score 10 = perfectly readable, placeholders are consistent. Score 0 = incoherent.

### 3. Completeness
Did the anonymizer catch ALL significant PII entities? Compare what was detected
against what you can see in the original text. Minor misses (e.g., a generic title)
are less serious than missing a full name or email address.

Score 10 = every significant entity caught. Score 0 = most entities missed.

## Input

ORIGINAL TEXT:
{original_text}

SANITIZED TEXT:
{sanitized_text}

ENTITY MAPPING:
{mapping_json}

## Output

Respond with ONLY a JSON object (no markdown fencing):
{{"pii_leak_detection": {{"score": <0-10>, "leaked_items": [<list of leaked PII strings found in sanitized text>], "explanation": "<brief>"}}, "semantic_coherence": {{"score": <0-10>, "explanation": "<brief>"}}, "completeness": {{"score": <0-10>, "missed_entities": [<list of PII in original not in mapping>], "explanation": "<brief>"}}, "overall_score": <average of three scores, rounded to 1 decimal>}}"""


async def evaluate_anonymization(
    original_text: str,
    sanitized_text: str,
    mapping: dict[str, str],
) -> dict | None:
    """Ask Gemini Pro to evaluate anonymization quality."""
    prompt = EVAL_PROMPT.format(
        original_text=original_text[:6000],
        sanitized_text=sanitized_text[:6000],
        mapping_json=json.dumps(mapping, indent=2),
    )
    try:
        raw = await call_gemini(
            [{"role": "user", "content": prompt}],
            model=GEMINI_EVAL_MODEL,
        )
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"    EVAL ERROR: {type(e).__name__}: {e}")
        return None


# --- Inline test cases ---

LEGAL_TEXTS = [
    {
        "name": "Merger Agreement Excerpt",
        "messages": [
            ChatMessage(
                role="user",
                content=(
                    "Can you review this clause? "
                    "Pursuant to the Agreement and Plan of Merger dated March 15, 2024, "
                    "between Acme Industries, Inc. (the 'Acquirer') and Widget Corp. "
                    "(the 'Target'), John Richardson, CEO of Widget Corp., shall serve "
                    "as a consultant for a transition period of 12 months. The Purchaser "
                    "agrees to retain Dr. Emily Watson as Chief Scientific Officer. "
                    "Contact: john.richardson@widgetcorp.com, +1 (555) 234-5678."
                ),
            ),
        ],
    },
    {
        "name": "NDA with Multiple Parties",
        "messages": [
            ChatMessage(
                role="user",
                content=(
                    "Please analyze this NDA: "
                    "This Non-Disclosure Agreement is entered into by Goldman Sachs Group, Inc. "
                    "('Disclosing Party') and Sarah Martinez of Blackstone Inc. "
                    "('Receiving Party'). The Receiving Party shall not disclose any "
                    "Confidential Information to third parties including but not limited to "
                    "competitors such as Morgan Stanley or JPMorgan Chase. "
                    "All notices shall be sent to sarah.martinez@blackstone.com or "
                    "to the attention of Michael Chen at 200 Park Avenue, New York, NY 10166."
                ),
            ),
        ],
    },
    {
        "name": "Multi-turn Conversation",
        "messages": [
            ChatMessage(
                role="user",
                content=(
                    "I need help with a licensing dispute. TechVentures LLC licensed their "
                    "patent portfolio to Samsung Electronics on January 8, 2023. The key "
                    "contact was David Park at david.park@techventures.io."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Also, the arbitration clause names Robert Kim of Baker McKenzie as "
                    "the designated arbitrator. Samsung's in-house counsel Lisa Wong "
                    "has objected to the venue being in San Francisco, California."
                ),
            ),
        ],
    },
    {
        "name": "Employment Agreement",
        "messages": [
            ChatMessage(
                role="user",
                content=(
                    "Review this employment clause: "
                    "The Employee, Jennifer Adams (SSN: 987-65-4321), shall be employed "
                    "by Amazon Web Services, Inc. as Senior Vice President of Engineering, "
                    "reporting to the Board of Directors. The Employee's annual compensation "
                    "shall be $450,000, with a signing bonus of $200,000. The Employee's "
                    "office shall be located at 410 Terry Avenue North, Seattle, WA 98109. "
                    "Emergency contact: Thomas Adams, +1 (206) 555-0142."
                ),
            ),
        ],
    },
]


def print_separator():
    print("=" * 80)


def print_result(name: str, messages: list[ChatMessage], result):
    print_separator()
    print(f"TEST: {name}")
    print_separator()

    print("\n--- Original Messages ---")
    for i, msg in enumerate(messages):
        print(f"  [{i}] {msg.role}: {msg.content[:120]}...")

    print("\n--- Mapping ---")
    for original, placeholder in sorted(result.mapping.items()):
        print(f"  {original!r:40s} -> {placeholder}")

    print("\n--- Sanitized Messages ---")
    for i, msg in enumerate(result.sanitized_messages):
        print(f"  [{i}] {msg.role}: {msg.content[:200]}...")

    print("\n--- Entities Found ---")
    for e in result.entities_found:
        print(f"  {e.action:12s} | {e.text!r:30s} | type={e.entity_type:20s} | replacement={e.replacement}")

    print(f"\n--- Context Descriptors ---")
    for k, v in result.context_descriptors.items():
        print(f"  {k}: {v}")

    print(f"\n--- Sensitivity: {result.sensitivity} ---")

    leaked = []
    for msg in result.sanitized_messages:
        for original in result.mapping:
            if original in msg.content:
                leaked.append(original)
    if leaked:
        print(f"\n  *** LEAK DETECTED: {leaked} ***")
    else:
        print("\n  No leaks detected in sanitized output.")

    print()


def print_eval_result(eval_result: dict | None):
    if not eval_result:
        print("  EVAL: skipped (error)")
        return

    for criterion in ("pii_leak_detection", "semantic_coherence", "completeness"):
        section = eval_result.get(criterion, {})
        score = section.get("score", "?")
        explanation = section.get("explanation", "")
        print(f"  EVAL {criterion}: {score}/10 — {explanation}")

        if criterion == "pii_leak_detection":
            leaked = section.get("leaked_items", [])
            if leaked:
                print(f"        leaked: {leaked}")
        elif criterion == "completeness":
            missed = section.get("missed_entities", [])
            if missed:
                print(f"        missed: {missed}")

    overall = eval_result.get("overall_score", "?")
    print(f"  EVAL overall: {overall}/10")


async def wait_for_ollama(max_wait: int = 30) -> bool:
    """Wait for ollama to become responsive, with backoff."""
    for i in range(max_wait):
        if await check_ollama():
            return True
        await asyncio.sleep(1)
    return False


async def run_test_case(
    test_case: dict,
    settings: Settings,
    template: str,
    run_eval: bool,
) -> dict:
    """Run a single test case. Returns result dict with pass/fail/scores."""
    name = test_case["name"]
    messages = [ChatMessage(role=m.role, content=m.content) for m in test_case["messages"]]

    # Ensure ollama is healthy before each test
    if not await check_ollama():
        print(f"  Ollama unresponsive, waiting up to 30s...")
        if not await wait_for_ollama():
            print_separator()
            print(f"TEST: {name}")
            print(f"  RESULT: ERROR — Ollama not recovered")
            return {"name": name, "status": "error", "error": "Ollama not recovered"}

    try:
        result = await anonymize_messages(messages, settings, prompt_template=template)
    except Exception as e:
        print_separator()
        print(f"TEST: {name}")
        print(f"  RESULT: ERROR — {type(e).__name__}: {e}")
        return {"name": name, "status": "error", "error": str(e)}

    print_result(name, test_case["messages"], result)

    # Basic string-match leak check
    leak_found = False
    for msg in result.sanitized_messages:
        for original in result.mapping:
            if original in msg.content:
                leak_found = True

    # Destination test (Gemini Flash)
    print("  DESTINATION: forwarding to Gemini Flash...")
    dest = await test_destination(result.sanitized_messages)
    if dest["ok"]:
        print(f"  DESTINATION: OK ({len(dest['response'])} chars response)")
    else:
        print(f"  DESTINATION: FAIL — {dest['error']}")

    # Evaluation (Gemini Pro)
    eval_result = None
    if run_eval:
        print("  EVAL: running Gemini Pro evaluation...")
        original_text = "\n\n".join(m.content for m in test_case["messages"])
        sanitized_text = "\n\n".join(m.content for m in result.sanitized_messages)
        eval_result = await evaluate_anonymization(original_text, sanitized_text, result.mapping)
        print_eval_result(eval_result)

    if leak_found:
        status = "fail"
        print(f"  RESULT: FAIL (leak in output)")
    elif not result.mapping:
        status = "warn"
        print(f"  RESULT: WARN (no entities detected)")
    else:
        status = "pass"
        print(f"  RESULT: PASS")

    print()
    return {
        "name": name,
        "status": status,
        "mapping_count": len(result.mapping),
        "entities_count": len(result.entities_found),
        "destination_ok": dest["ok"],
        "eval": eval_result,
    }


async def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY env var not set.")
        print("Usage: GEMINI_API_KEY=your-key python tests/e2e/test_gemini_anonymization.py")
        return False

    settings = Settings(anonymization_mode="llm")

    # Only require ollama if using local endpoint
    using_ollama = "localhost" in settings.local_llm_url or "127.0.0.1" in settings.local_llm_url
    if using_ollama and not await check_ollama():
        print("ERROR: Ollama not reachable at localhost:11434.")
        print("Start ollama first, or set LOCAL_LLM_URL to a cloud endpoint.")
        return False

    anonymizer_label = f"ollama / {settings.local_llm_model}" if using_ollama else f"{settings.local_llm_model} @ {settings.local_llm_url}"
    print(f"Hey Jude E2E Test")
    print(f"  Anonymizer:  {anonymizer_label}")
    print(f"  Destination: {GEMINI_MODEL}")
    print(f"  Evaluator:   {GEMINI_EVAL_MODEL}")
    print_separator()

    prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "anonymize.txt"
    template = prompt_path.read_text()

    # Load EDGAR documents (auto-download if needed)
    print("Loading SEC EDGAR test documents (auto-downloading if needed)...")
    edgar_cases = load_test_cases(max_chars=4000)
    for case in edgar_cases:
        LEGAL_TEXTS.append({
            "name": case["name"],
            "messages": [
                ChatMessage(
                    role="user",
                    content=f"{case['prompt_prefix']}\n\n{case['content']}",
                ),
            ],
        })
    print(f"Loaded {len(edgar_cases)} EDGAR documents, {len(LEGAL_TEXTS)} total test cases\n")

    # Run all test cases with cooldown between each
    results = []
    for i, test_case in enumerate(LEGAL_TEXTS):
        if i > 0:
            await asyncio.sleep(2)
        r = await run_test_case(test_case, settings, template, run_eval=True)
        results.append(r)

    # Summary
    print_separator()
    print("SUMMARY")
    print_separator()

    passed = sum(1 for r in results if r["status"] == "pass")
    warned = sum(1 for r in results if r["status"] == "warn")
    failed = sum(1 for r in results if r["status"] == "fail")
    errored = sum(1 for r in results if r["status"] == "error")

    for r in results:
        status_icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "error": "ERR "}[r["status"]]
        eval_score = ""
        if r.get("eval") and r["eval"].get("overall_score") is not None:
            eval_score = f" (eval: {r['eval']['overall_score']}/10)"
        dest_str = " dest:OK" if r.get("destination_ok") else " dest:FAIL"
        print(f"  [{status_icon}] {r['name']}{dest_str}{eval_score}")

    print()
    print(f"  {passed} passed, {warned} warned, {failed} failed, {errored} errors — {len(results)} total")

    # Aggregate eval scores
    eval_scores = [
        r["eval"]["overall_score"]
        for r in results
        if r.get("eval") and r["eval"].get("overall_score") is not None
    ]
    if eval_scores:
        avg = sum(eval_scores) / len(eval_scores)
        print(f"  Average evaluation score: {avg:.1f}/10 ({len(eval_scores)} evaluated)")

    print_separator()
    return failed == 0 and errored == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
