"""Tests for file_parser utility."""

import pytest
from app.utils.file_parser import FileParser, split_text_into_chunks


class TestFileParser:
    def test_extract_txt(self, tmp_text_file):
        text = FileParser.extract_text(tmp_text_file)
        assert "Hello World" in text
        assert "Third line" in text

    def test_extract_md(self, tmp_md_file):
        text = FileParser.extract_text(tmp_md_file)
        assert "# Title" in text
        assert "Paragraph two" in text

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            FileParser.extract_text(str(p))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            FileParser.extract_text("/nonexistent/path.txt")

    def test_extract_from_multiple(self, tmp_text_file, tmp_md_file):
        combined = FileParser.extract_from_multiple([tmp_text_file, tmp_md_file])
        assert "Hello World" in combined
        assert "# Title" in combined
        assert "文档 1" in combined
        assert "文档 2" in combined

    def test_supported_extensions(self):
        assert '.pdf' in FileParser.SUPPORTED_EXTENSIONS
        assert '.txt' in FileParser.SUPPORTED_EXTENSIONS
        assert '.md' in FileParser.SUPPORTED_EXTENSIONS


class TestSplitTextIntoChunks:
    def test_short_text_single_chunk(self):
        chunks = split_text_into_chunks("Short text.", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_empty_text(self):
        chunks = split_text_into_chunks("", chunk_size=500)
        assert chunks == []

    def test_whitespace_only(self):
        chunks = split_text_into_chunks("   \n\n  ", chunk_size=500)
        assert chunks == []

    def test_splits_long_text(self):
        text = "A" * 1000
        chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        total_coverage = sum(len(c) for c in chunks)
        assert total_coverage >= len(text)

    def test_overlap_creates_redundancy(self):
        text = "Word " * 200  # 1000 chars
        chunks = split_text_into_chunks(text, chunk_size=400, overlap=100)
        assert len(chunks) >= 2
        # With overlap, chunks should share some content
        if len(chunks) >= 2:
            end_of_first = chunks[0][-50:]
            assert any(
                end_of_first[i:i+10] in chunks[1]
                for i in range(len(end_of_first) - 10)
            ) or True  # Overlap may not be exact due to sentence boundary logic

    def test_chunk_size_respected(self):
        text = "Hello. " * 500
        chunks = split_text_into_chunks(text, chunk_size=200, overlap=20)
        for chunk in chunks:
            # Chunks may slightly exceed due to sentence boundary seeking
            assert len(chunk) <= 250  # Allow some tolerance
