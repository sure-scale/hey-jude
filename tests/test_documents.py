import base64
from email.message import EmailMessage
from io import BytesIO

import pytest
from docx import Document

from hey_jude.config import Settings
from hey_jude.services.documents import (
    DocumentProcessingError,
    content_to_text,
    extract_document_text,
)


def _data_url(media_type: str, raw: bytes) -> str:
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _minimal_pdf(text: str) -> bytes:
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
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


def _docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def test_extracts_text_layer_pdf():
    text = extract_document_text(
        _minimal_pdf("John Smith signed the NDA."),
        media_type="application/pdf",
        filename="nda.pdf",
    )

    assert "John Smith signed the NDA." in text


def test_extracts_docx_text():
    text = extract_document_text(
        _docx_bytes("Jane Roe reviewed the merger agreement."),
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        filename="agreement.docx",
    )

    assert "Jane Roe reviewed the merger agreement." in text


def test_extracts_html_text_from_base64_file_part():
    content = [
        {"type": "text", "text": "Summarize this filing."},
        {
            "type": "input_file",
            "filename": "filing.html",
            "file_data": _data_url(
                "text/html",
                b"<html><body><h1>Acme merger</h1><p>John Smith advised the board.</p></body></html>",
            ),
        },
    ]

    result = content_to_text(content, Settings())

    assert "Summarize this filing." in result.text
    assert "Acme merger" in result.text
    assert "John Smith advised the board." in result.text
    assert result.warnings == []


def test_extracts_email_plain_text_body():
    message = EmailMessage()
    message["Subject"] = "Privilege review"
    message["From"] = "lawyer@example.com"
    message["To"] = "client@example.com"
    message.set_content("Please review the NDA for Jane Roe.")

    text = extract_document_text(
        message.as_bytes(),
        media_type="message/rfc822",
        filename="privileged.eml",
    )

    assert "Subject: Privilege review" in text
    assert "Please review the NDA for Jane Roe." in text


def test_unreadable_image_rejects_by_default():
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url("image/png", b"not-real-image-bytes"),
            },
        }
    ]

    with pytest.raises(DocumentProcessingError, match="not readable as text"):
        content_to_text(content, Settings(document_unreadable_action="reject"))


def test_invalid_base64_rejects_as_unreadable_attachment():
    content = [{"type": "input_file", "filename": "memo.pdf", "file_data": "not base64"}]

    with pytest.raises(DocumentProcessingError, match="invalid base64"):
        content_to_text(content, Settings(document_unreadable_action="reject"))


def test_unreadable_image_warn_mode_omits_bytes_and_records_reason():
    content = [
        {"type": "text", "text": "Analyze the attachment."},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url("image/png", b"not-real-image-bytes"),
            },
        },
    ]

    result = content_to_text(content, Settings(document_unreadable_action="warn"))

    assert "Analyze the attachment." in result.text
    assert "not-real-image-bytes" not in result.text
    assert "omitted" in result.text
    assert result.warnings[0].action == "warn"
    assert "not readable as text" in result.warnings[0].reason


def test_unreadable_image_skip_mode_omits_bytes_and_records_reason():
    content = [
        {"type": "text", "text": "Analyze the attachment."},
        {
            "type": "image_url",
            "image_url": {
                "url": _data_url("image/png", b"not-real-image-bytes"),
            },
        },
    ]

    result = content_to_text(content, Settings(document_unreadable_action="skip"))

    assert result.text == "Analyze the attachment."
    assert result.warnings[0].action == "skip"
    assert "not readable as text" in result.warnings[0].reason
