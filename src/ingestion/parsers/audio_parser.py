from pathlib import Path

import structlog

from src.models.document import DocumentChunk, Modality

log = structlog.get_logger()


class AudioParser:
    def __init__(self, model_name: str = "base") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self._model_name)
        return self._model

    def parse(self, file_path: str, document_id: str) -> list[DocumentChunk]:
        path = Path(file_path)
        if not path.exists():
            log.error("audio file not found", path=file_path)
            return []

        model = self._load_model()
        result = model.transcribe(str(path), language="en", verbose=False)

        chunks: list[DocumentChunk] = []
        segments = result.get("segments", [])

        current_text = ""
        current_start = 0.0
        segment_count = 0

        for seg in segments:
            current_text += seg["text"]
            segment_count += 1

            if segment_count >= 10 or seg.get("id", 0) == len(segments) - 1:
                chunks.append(
                    DocumentChunk(
                        id=f"{document_id}_audio_{len(chunks)}",
                        document_id=document_id,
                        content=current_text.strip(),
                        modality=Modality.AUDIO,
                        chunk_index=len(chunks),
                        metadata={
                            "source": "audio_transcript",
                            "start_time": current_start,
                            "end_time": seg["end"],
                        },
                    )
                )
                current_text = ""
                current_start = seg["end"]
                segment_count = 0

        log.info("parsed audio", path=file_path, chunks=len(chunks))
        return chunks
