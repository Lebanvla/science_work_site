from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from app.config import MODEL_NAME, ROAD_CLASS_INDEX, ROAD_CLASS_NAME

logger = logging.getLogger(__name__)

_MODEL: "SegmentationModel | None" = None


class SegmentationModel(ABC):
    @abstractmethod
    def predict(self, image_path: str) -> np.ndarray:
        """Return a binary uint8 road mask with shape (height, width)."""
        raise NotImplementedError


class SegFormerCityscapesModel(SegmentationModel):
    def __init__(self, model_name: str = MODEL_NAME, road_class_name: str = ROAD_CLASS_NAME) -> None:
        self.model_name = model_name
        self.road_class_name = road_class_name.lower().strip()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading image processor: %s", self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        logger.info("Loading segmentation model: %s", self.model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.road_class_id = self._resolve_road_class_id()
        logger.info(
            "Model loaded on %s. Road class '%s' resolved to index %s",
            self.device,
            self.road_class_name,
            self.road_class_id,
        )

    def _resolve_road_class_id(self) -> int:
        if ROAD_CLASS_INDEX is not None and ROAD_CLASS_INDEX != "":
            return int(ROAD_CLASS_INDEX)

        id2label = getattr(self.model.config, "id2label", None) or {}
        normalized = {
            int(class_id): str(label).lower().strip()
            for class_id, label in id2label.items()
        }

        for class_id, label in normalized.items():
            if label == self.road_class_name:
                return class_id

        for class_id, label in normalized.items():
            if self.road_class_name in label:
                return class_id

        if self.road_class_name == "road":
            logger.warning(
                "Could not find 'road' in model config id2label=%s. Falling back to Cityscapes road index 0.",
                id2label,
            )
            return 0

        raise ValueError(
            f"Could not resolve road class '{self.road_class_name}'. "
            "Set ROAD_CLASS_INDEX explicitly if your model config does not contain id2label."
        )

    def predict(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            predicted = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        mask = (predicted == self.road_class_id).astype(np.uint8) * 255
        return mask


def load_model() -> SegmentationModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = SegFormerCityscapesModel()
    return _MODEL


def predict_road_mask(image_path: str) -> np.ndarray:
    model = load_model()
    return model.predict(image_path)


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb).astype(np.float32)

    if mask.shape[:2] != image_np.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    road_mask = mask > 0
    overlay_color = np.array([255, 128, 0], dtype=np.float32)
    alpha = 0.45

    blended = image_np.copy()
    blended[road_mask] = (1.0 - alpha) * blended[road_mask] + alpha * overlay_color

    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def process_image(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input image not found: {input_file}")

    image = Image.open(input_file).convert("RGB")
    mask = predict_road_mask(str(input_file))
    result = overlay_mask(image, mask)
    result.save(output_file, format="PNG")
