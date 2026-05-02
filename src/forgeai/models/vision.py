"""
Full multimodal pipeline (Phase 2).

Image preprocessing, vision encoder routing, and multi-modal input handling
for vision-language models like LLaVA, Qwen2-VL, InternVL, and Phi-3-Vision.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class ImageInput:
    """Processed image input for vision models."""
    raw_bytes: bytes
    width: int = 0
    height: int = 0
    format: str = ""
    source: str = ""  # file path or URL


@dataclass
class MultimodalInput:
    """Combined text + image input for vision-language models."""
    text: str
    images: list[ImageInput] = field(default_factory=list)
    max_image_size: int = 1344  # Max dimension for resizing
    image_detail: str = "auto"  # auto, low, high


class VisionPipeline:
    """
    Handles image preprocessing and vision encoder routing.

    Supports loading images from files, URLs, and base64 strings.
    Automatically resizes and formats images for the target model.
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

    def __init__(self, max_image_size: int = 1344) -> None:
        self.max_image_size = max_image_size
        self._pil_available = self._check_pil()

    def _check_pil(self) -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("PIL") is not None
        except (ImportError, ValueError):
            return False

    def load_image(self, source: str | bytes) -> ImageInput:
        """Load an image from a file path, URL, or raw bytes."""
        if isinstance(source, bytes):
            return self._load_from_bytes(source)
        if source.startswith(("http://", "https://")):
            return self._load_from_url(source)
        return self._load_from_file(source)

    def _load_from_file(self, path: str) -> ImageInput:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if p.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {p.suffix}")
        raw = p.read_bytes()
        img = ImageInput(raw_bytes=raw, format=p.suffix.lstrip("."), source=path)
        if self._pil_available:
            img = self._get_dimensions(img)
        return img

    def _load_from_url(self, url: str) -> ImageInput:
        try:
            import httpx
            resp = httpx.get(url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            img = ImageInput(raw_bytes=resp.content, source=url)
            if self._pil_available:
                img = self._get_dimensions(img)
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to download image: {e}") from e

    def _load_from_bytes(self, data: bytes) -> ImageInput:
        img = ImageInput(raw_bytes=data, source="bytes")
        if self._pil_available:
            img = self._get_dimensions(img)
        return img

    def _get_dimensions(self, img: ImageInput) -> ImageInput:
        from PIL import Image
        pil_img = Image.open(io.BytesIO(img.raw_bytes))
        img.width, img.height = pil_img.size
        img.format = pil_img.format or ""
        return img

    def preprocess(self, img: ImageInput) -> ImageInput:
        """Resize image if needed for the target model."""
        if not self._pil_available:
            return img
        from PIL import Image
        pil_img = Image.open(io.BytesIO(img.raw_bytes))
        if max(pil_img.size) > self.max_image_size:
            pil_img.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)
            buf = io.BytesIO()
            fmt = img.format.upper() if img.format else "PNG"
            if fmt == "JPG":
                fmt = "JPEG"
            pil_img.save(buf, format=fmt)
            img.raw_bytes = buf.getvalue()
            img.width, img.height = pil_img.size
        return img

    def to_base64(self, img: ImageInput) -> str:
        """Convert image to base64 string."""
        return base64.b64encode(img.raw_bytes).decode("utf-8")

    def prepare_multimodal_input(
        self,
        text: str,
        image_sources: list[str],
    ) -> MultimodalInput:
        """
        Prepare a combined text + images input for vision models.

        Args:
            text: The text prompt.
            image_sources: List of image paths/URLs.

        Returns:
            MultimodalInput ready for the engine.
        """
        images = []
        for src in image_sources:
            img = self.load_image(src)
            img = self.preprocess(img)
            images.append(img)

        return MultimodalInput(
            text=text,
            images=images,
            max_image_size=self.max_image_size,
        )


def detect_vision_model(model_type: str) -> str | None:
    """Map model type to vision encoder configuration."""
    vision_configs = {
        "llava": "clip-vit-large-patch14",
        "internvl": "internvl-vit",
        "qwen2_vl": "qwen2-vl-vit",
        "phi3_v": "phi3-vision-encoder",
        "glm4v": "glm4v-encoder",
    }
    return vision_configs.get(model_type.lower())
