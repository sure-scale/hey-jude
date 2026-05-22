#!/usr/bin/env python3
"""
End-to-end test of the LLM-first anonymization pipeline using Gemini 2.5 Flash.

Usage:
    GEMINI_API_KEY=your-key python tests/e2e/test_gemini_anonymization.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hey_jude.config import Settings
from hey_jude.models import ChatMessage
from hey_jude.services.anonymizer import anonymize_messages

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


async def call_gemini(prompt: str, settings: Settings) -> str:
    payload = {
        "model": GEMINI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(GEMINI_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


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
        print(f"  [{i}] {msg.role}: {msg.content}")

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


async def main():
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY env var not set.")
        print("Usage: GEMINI_API_KEY=your-key python tests/e2e/test_gemini_anonymization.py")
        return False

    print(f"Hey Jude E2E Test — {GEMINI_MODEL}")
    print(f"Model: {GEMINI_MODEL}")
    print_separator()

    prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "anonymize.txt"
    template = prompt_path.read_text()
    settings = Settings(anonymization_mode="llm")

    passed = 0
    failed = 0

    for test_case in LEGAL_TEXTS:
        name = test_case["name"]
        messages = [ChatMessage(role=m.role, content=m.content) for m in test_case["messages"]]

        try:
            with patch("hey_jude.services.anonymizer.call_local_llm", new=call_gemini):
                result = await anonymize_messages(
                    messages, settings, prompt_template=template
                )
            print_result(name, test_case["messages"], result)

            leak_found = False
            for msg in result.sanitized_messages:
                for original in result.mapping:
                    if original in msg.content:
                        leak_found = True

            if leak_found:
                print(f"  RESULT: FAIL (leak in output)")
                failed += 1
            elif not result.mapping:
                print(f"  RESULT: WARN (no entities detected)")
                passed += 1
            else:
                print(f"  RESULT: PASS")
                passed += 1

        except Exception as e:
            print_separator()
            print(f"TEST: {name}")
            print(f"  RESULT: ERROR — {type(e).__name__}: {e}")
            failed += 1

    print_separator()
    print(f"SUMMARY: {passed} passed, {failed} failed, {len(LEGAL_TEXTS)} total")
    print_separator()

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
