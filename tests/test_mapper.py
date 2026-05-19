import pytest
from hey_jude.services.mapper import reverse_map_text


def test_simple_replacement():
    text = "Pinnacle Systems released a new product."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Microsoft released a new product."


def test_multiple_replacements():
    text = "Vertex Holdings sued Pinnacle Systems over Photogram."
    reverse_mapping = {
        "Vertex Holdings": "Meta",
        "Pinnacle Systems": "Microsoft",
        "Photogram": "Instagram",
    }
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Meta sued Microsoft over Instagram."


def test_case_insensitive_matching():
    text = "pinnacle systems announced earnings."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Microsoft announced earnings."


def test_longest_match_first():
    text = "Vertex Holdings International is a big company. Vertex Holdings is too."
    reverse_mapping = {
        "Vertex Holdings International": "Meta Platforms Inc",
        "Vertex Holdings": "Meta",
    }
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Meta Platforms Inc is a big company. Meta is too."


def test_multiple_occurrences():
    text = "Pinnacle Systems vs Pinnacle Systems in court."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Microsoft vs Microsoft in court."


def test_no_mapping_passthrough():
    text = "No entities here."
    result = reverse_map_text(text, {})
    assert result == "No entities here."


def test_no_match_passthrough():
    text = "Some other company did a thing."
    reverse_mapping = {"Pinnacle Systems": "Microsoft"}
    result = reverse_map_text(text, reverse_mapping)
    assert result == "Some other company did a thing."
