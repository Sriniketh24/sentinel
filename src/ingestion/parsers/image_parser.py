from pathlib import Path

import structlog

from src.models.document import DocumentChunk, Modality

log = structlog.get_logger()


class ImageParser:
    def __init__(self) -> None:
        self._processor = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoProcessor

            model_id = "microsoft/Florence-2-base"
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
        return self._model, self._processor

    def parse(self, file_path: str, document_id: str) -> list[DocumentChunk]:
        path = Path(file_path)
        if not path.exists():
            log.error("image not found", path=file_path)
            return []

        from PIL import Image

        image = Image.open(path).convert("RGB")
        caption = self._generate_caption(image)

        chunk = DocumentChunk(
            id=f"{document_id}_img_0",
            document_id=document_id,
            content=caption,
            modality=Modality.IMAGE,
            chunk_index=0,
            metadata={"source": "image_caption", "original_path": str(path)},
        )

        log.info("parsed image", path=file_path, caption_len=len(caption))
        return [chunk]

    def _generate_caption(self, image) -> str:
        import torch

        model, processor = self._load_model()
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )

        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return result.strip()
