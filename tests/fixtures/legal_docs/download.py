#!/usr/bin/env python3
"""
Download public-domain legal documents from SEC EDGAR for anonymization testing.

All documents are US government public records (17 U.S.C. § 105).
Downloaded .txt files are gitignored to avoid distribution concerns with the AGPL repo.

Usage:
    python tests/fixtures/legal_docs/download.py
"""

import html.parser
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).parent

USER_AGENT = "HeyJude/0.1.0 nick@surescale.ai"

DOCUMENTS = {
    "settlement_teligent_sawyer.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/352998/000110465921126768/tm2129875d1_ex10-2.htm",
        "description": "Teligent / Timothy B. Sawyer — Settlement Agreement & General Release",
        "pii_notes": "Personal home address, personal Gmail, corporate emails, attorney emails, "
        "law firm addresses, company addresses, dollar amounts",
    },
    "settlement_tarantella_keddy.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/851560/000119312504177083/dex101.htm",
        "description": "Tarantella / Caroline Keddy — Settlement Agreement & General Release",
        "pii_notes": "Individual name, attorney name/address, court case number, dollar amounts",
    },
    "employment_ppd_hill.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1003124/000119312511292827/d226725dex10281.htm",
        "description": "PPD / Raymond H. Hill — CEO Employment Agreement",
        "pii_notes": "Executive name, company address (Wilmington NC), salary/compensation",
    },
    "employment_euramax_brown.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1026743/000102674315000027/a101ceoemploymentagreement.htm",
        "description": "Euramax International / Richard Brown — CEO Employment Agreement",
        "pii_notes": "Executive name, HQ location (Norcross GA), residences in FL and NC",
    },
    "consulting_walmart_simon.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/104169/000010416914000043/exhibit102-consultingagree.htm",
        "description": "Walmart / William S. Simon — Consulting Agreement",
        "pii_notes": "Multiple names, corporate mailing address, job titles",
    },
    "consulting_nda_hg_holdings.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/797465/000143774922020170/ex_409970.htm",
        "description": "HG Holdings / Brad G. Garner — Consulting Agreement + NDA",
        "pii_notes": "Individual names, titles, monthly compensation, signatures",
    },
    "transition_societypass_nguyen.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1817511/000101376223002574/ea186434ex10-4_society.htm",
        "description": "Society Pass / Dennis Nguyen — Transition, Release & Consulting Agreement",
        "pii_notes": "Executive name, Singapore company address, company email, salary/bonus",
    },
    "nda_lightwave_logic.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1325964/000155335018000222/lwlg_ex10z10.htm",
        "description": "Lightwave Logic — Director's Non-Disclosure Agreement (template)",
        "pii_notes": "Company name/address only; director fields are blank template placeholders",
    },
    "separation_crypto_co_gilbert.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1688126/000149315218007846/ex10-1.htm",
        "description": "The Crypto Company / James Gilbert — Separation Agreement & Mutual Release",
        "pii_notes": "Executive name, CEO name (Ron Levy), law firm (Drinker Biddle & Reath), "
        "dates, COBRA provisions, severance amounts",
    },
    "noncompete_cdi_stuart.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/18396/000119312505067086/dex10l.htm",
        "description": "CDI Corp / Jay G. Stuart — Release, Waiver & Non-Competition Agreement",
        "pii_notes": "Names, Philadelphia address (1600 Arch St), 7+ dollar amounts "
        "($310K severance, $50K stay bonus), apartment lease, furniture inventory",
    },
    "lease_mostofi_b4mc.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/823546/000107878215001042/f10k123113_ex10z6.htm",
        "description": "Mostofi & Co / B4MC Gold Mines — Executive Suite Sublease (Beverly Hills)",
        "pii_notes": "Names, CPA credentials, Beverly Hills address (468 N. Camden Dr), "
        "phone numbers, notary name/commission, PO box, rent amounts",
    },
    "loan_vpr_brands_frija.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1376231/000158095718000353/vprb072318ex10-1.htm",
        "description": "VPR Brands / Kevin Frija — Promissory Note",
        "pii_notes": "Email (Kevin.Frija@vprbrands.com), phone (954-715-7001), "
        "Fort Lauderdale address, EIN, attorney name/address, dollar amounts",
    },
    "license_implant_sciences_ibt.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1068874/000106887408000032/license1.htm",
        "description": "Implant Sciences / International Brachytherapy — IP License Agreement",
        "pii_notes": "CEO names, US address (Wakefield MA), Belgium address, "
        "patent number (US 6,183,409), royalty terms",
    },
    "purchase_bizright_bzrth.txt": {
        "url": "https://www.sec.gov/Archives/edgar/data/1830072/000168316820004073/filename6.htm",
        "description": "BizRight LLC / BZRTH Inc — Asset Purchase Agreement",
        "pii_notes": "Managing member name (Allan Huang), Irwindale CA address, "
        "purchase price ($2.6M), detailed financial line items",
    },
}


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Extract plain text from HTML, preserving paragraph structure."""

    _BLOCK_TAGS = frozenset(
        {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "table"}
    )

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        if tag.lower() in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag.lower() in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines = [" ".join(line.split()) for line in raw.split("\n")]
        text = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(content: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(content)
    return extractor.get_text()


def download_document(name: str, info: dict) -> bool:
    output_path = DOWNLOAD_DIR / name

    if output_path.exists():
        print(f"  SKIP  {name} (cached)")
        return True

    req = urllib.request.Request(info["url"], headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass

    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        text = html_to_text(raw)

        if len(text) < 100:
            print(f"  WARN  {name} — extracted text too short ({len(text)} chars)")
            return False

        output_path.write_text(text, encoding="utf-8")
        print(f"  OK    {name} ({len(text):,} chars)")
        return True

    except Exception as e:
        print(f"  FAIL  {name} — {type(e).__name__}: {e}")
        return False


_PROMPT_PREFIXES = {
    "settlement_": "Please review this settlement agreement and identify any issues:",
    "employment_": "Can you analyze this employment agreement?",
    "consulting_": "Review this consulting agreement for potential concerns:",
    "transition_": "Please review this transition and consulting agreement:",
    "nda_": "Can you analyze this non-disclosure agreement?",
    "separation_": "Review this separation agreement and general release:",
    "noncompete_": "Please analyze the restrictive covenants in this agreement:",
    "lease_": "Can you review this lease agreement?",
    "license_": "Review this license agreement for key terms:",
    "loan_": "Please review this loan agreement:",
    "purchase_": "Can you analyze this purchase agreement?",
}


def ensure_downloaded() -> dict[str, Path]:
    """Download all docs if needed, return dict of name -> path for available docs."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    available = {}

    for name, info in DOCUMENTS.items():
        path = DOWNLOAD_DIR / name
        if path.exists() or download_document(name, info):
            available[name] = path
        time.sleep(0.2)

    return available


def load_test_cases(max_chars: int = 4000) -> list[dict]:
    """Load downloaded docs as test case dicts. Auto-downloads if needed.

    Returns list of dicts with keys: name, content, prompt_prefix, pii_notes.
    Caller is responsible for wrapping in ChatMessage objects.
    """
    available = ensure_downloaded()
    cases = []

    for filename, path in available.items():
        info = DOCUMENTS[filename]
        text = path.read_text(encoding="utf-8")

        if max_chars and len(text) > max_chars:
            cut = text[:max_chars].rsplit("\n", 1)[0]
            text = cut if cut else text[:max_chars]

        prefix = "Please review this legal document:"
        for key, prompt in _PROMPT_PREFIXES.items():
            if filename.startswith(key):
                prefix = prompt
                break

        cases.append({
            "name": f"[EDGAR] {info['description']}",
            "content": text,
            "prompt_prefix": prefix,
            "pii_notes": info["pii_notes"],
        })

    return cases


def main():
    print(f"Downloading {len(DOCUMENTS)} legal documents from SEC EDGAR (public domain)...")
    print(f"Target: {DOWNLOAD_DIR}\n")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for name, info in DOCUMENTS.items():
        if download_document(name, info):
            ok += 1
        else:
            fail += 1
        time.sleep(0.2)

    print(f"\nDone: {ok} downloaded, {fail} failed, {len(DOCUMENTS)} total")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
