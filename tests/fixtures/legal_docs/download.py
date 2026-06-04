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


# Real named individual (person, org) per document, used to build the
# identity-essential prompt for the IRREDUCIBLE track. The SAME document is
# reducible under a "review this agreement" prompt (masking the names fully
# answers it) and irreducible under a prompt that asks about the real person
# behind the placeholder — reducibility is a property of the REQUEST, not the
# document. None where the document names no single individual whose identity
# the request would hinge on (a blank template, or an entity-to-entity deal);
# those docs stay on the reducible track only.
_SUBJECTS = {
    "settlement_teligent_sawyer.txt": ("Timothy B. Sawyer", "Teligent"),
    "settlement_tarantella_keddy.txt": ("Caroline Keddy", "Tarantella"),
    "employment_ppd_hill.txt": ("Raymond H. Hill", "PPD"),
    "employment_euramax_brown.txt": ("Richard Brown", "Euramax International"),
    "consulting_walmart_simon.txt": ("William S. Simon", "Walmart"),
    "consulting_nda_hg_holdings.txt": ("Brad G. Garner", "HG Holdings"),
    "transition_societypass_nguyen.txt": ("Dennis Nguyen", "Society Pass"),
    "nda_lightwave_logic.txt": None,
    "separation_crypto_co_gilbert.txt": ("James Gilbert", "The Crypto Company"),
    "noncompete_cdi_stuart.txt": ("Jay G. Stuart", "CDI Corp"),
    "lease_mostofi_b4mc.txt": None,
    "loan_vpr_brands_frija.txt": ("Kevin Frija", "VPR Brands"),
    "license_implant_sciences_ibt.txt": None,
    "purchase_bizright_bzrth.txt": ("Allan Huang", "BizRight LLC"),
}


_EDGAR_HEADER_LINE = re.compile(
    r"^(EX-\S+|\d{1,3}|\S+\.(?:htm|html|txt))$", re.IGNORECASE
)


def _strip_edgar_header(text: str) -> str:
    """Remove the SEC EDGAR exhibit preamble the HTML wrapper carries above the
    document body: the exhibit label ("EX-10.1"), the sequence number, and the
    source filename ("vprb072318ex10-1.htm"). These are submission plumbing, not
    legal content, and the filename embeds the ticker and filing date — handing
    a re-identifier to any downstream reader. Strip the leading run of header
    lines (capped so it cannot eat into the body), then drop any stray
    exhibit-filename token that survives elsewhere.
    """
    lines = text.split("\n")
    i = 0
    while i < min(len(lines), 6) and (
        lines[i].strip() == "" or _EDGAR_HEADER_LINE.match(lines[i].strip())
    ):
        i += 1
    body = "\n".join(lines[i:])
    body = re.sub(r"\b\S+\.(?:htm|html)\b", "", body)
    return body.strip()


def _identity_prompt(name: str, org: str) -> str:
    """A prompt whose answer genuinely requires the real named individual.

    Asking about the real person's external history cannot be served by a
    PERSON_01 placeholder, so masking destroys the task: NAMED_ENTITY_ESSENTIAL.
    """
    return (
        f"I'm conducting background diligence on {name}, the individual named in "
        f"this agreement involving {org}. Using this document together with your "
        f"own knowledge of this specific person, summarize their public track "
        f"record, other roles and board memberships, and any reputational or "
        f"litigation history relevant to this engagement."
    )


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

    Each document yields one REDUCIBLE case (a "review this agreement" prompt,
    which masking fully satisfies) and, where the document names a specific
    individual (see _SUBJECTS), one IRREDUCIBLE case (an identity-essential
    prompt about that real person, which masking cannot serve). Reducibility is
    a property of the request, so the same source document appears on both
    tracks under different prompts.

    Returns dicts with keys: name, content, prompt, expected_class,
    irreducible_reason (only on the irreducible track), pii_notes. The caller
    wraps `prompt` + `content` into a ChatMessage.
    """
    available = ensure_downloaded()
    cases = []

    for filename, path in available.items():
        info = DOCUMENTS[filename]
        text = _strip_edgar_header(path.read_text(encoding="utf-8"))

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
            "prompt": prefix,
            "expected_class": "reducible",
            "pii_notes": info["pii_notes"],
        })

        subject = _SUBJECTS.get(filename)
        if subject:
            name, org = subject
            cases.append({
                "name": f"[EDGAR·ID] {name} background diligence ({org})",
                "content": text,
                "prompt": _identity_prompt(name, org),
                "expected_class": "irreducible",
                "irreducible_reason": "NAMED_ENTITY_ESSENTIAL",
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
