"""Tests for TextProcessor service."""

import pytest
from app.services.text_processor import TextProcessor


class TestTextProcessor:
    def test_preprocess_normalizes_newlines(self):
        text = "Line one\r\nLine two\rLine three\n"
        result = TextProcessor.preprocess_text(text)
        assert "\r" not in result
        assert "Line one\nLine two\nLine three" == result

    def test_preprocess_collapses_blank_lines(self):
        text = "Para one.\n\n\n\n\nPara two."
        result = TextProcessor.preprocess_text(text)
        assert "\n\n\n" not in result
        assert "Para one.\n\nPara two." == result

    def test_preprocess_strips_line_whitespace(self):
        text = "  indented  \n  also indented  "
        result = TextProcessor.preprocess_text(text)
        lines = result.split("\n")
        for line in lines:
            assert line == line.strip()

    def test_get_text_stats(self):
        text = "Hello world.\nSecond line.\nThird line."
        stats = TextProcessor.get_text_stats(text)
        assert stats["total_chars"] == len(text)
        assert stats["total_lines"] == 3
        assert stats["total_words"] == 6

    def test_split_text_delegates(self):
        text = "A" * 1000
        chunks = TextProcessor.split_text(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1

    def test_extract_from_files(self, tmp_text_file):
        result = TextProcessor.extract_from_files([tmp_text_file])
        assert "Hello World" in result
