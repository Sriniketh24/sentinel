from src.ingestion.chunker import MultimodalChunker
from src.models.document import DocumentChunk, Modality


def test_short_text_not_split():
    chunker = MultimodalChunker()
    chunk = DocumentChunk(
        id="test_1",
        document_id="doc_1",
        content="Short text that fits in one chunk.",
        modality=Modality.TEXT,
    )
    result = chunker.chunk([chunk])
    assert len(result) == 1
    assert result[0].content == chunk.content


def test_long_text_split_into_multiple():
    chunker = MultimodalChunker()
    words = ["word"] * 1500
    chunk = DocumentChunk(
        id="test_2",
        document_id="doc_2",
        content=" ".join(words),
        modality=Modality.TEXT,
    )
    result = chunker.chunk([chunk])
    assert len(result) > 1
    for c in result:
        word_count = len(c.content.split())
        assert word_count <= 512 + 10  # allow small tolerance


def test_table_chunks_not_split():
    chunker = MultimodalChunker()
    chunk = DocumentChunk(
        id="test_3",
        document_id="doc_3",
        content="| Col1 | Col2 |\n| --- | --- |\n| A | B |",
        modality=Modality.TABLE,
    )
    result = chunker.chunk([chunk])
    assert len(result) == 1


def test_chunk_indexes_sequential():
    chunker = MultimodalChunker()
    chunks = [
        DocumentChunk(id=f"t{i}", document_id="d1", content=f"Content {i}", modality=Modality.TEXT)
        for i in range(5)
    ]
    result = chunker.chunk(chunks)
    for i, c in enumerate(result):
        assert c.chunk_index == i
