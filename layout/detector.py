"""
YOLOv8 DocLayNet layout detector.

Wraps ultralytics YOLO to detect document layout regions (title,
heading, body text, footnotes, tables, etc.) from rendered page images.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .models import DOCLAYNET_INDEX_TO_LABEL, LayoutLabel, LayoutRegion

# Default model path relative to project root
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "yolov8x_doclaynet.pt"
)


class LayoutDetector:
    """
    Detects document layout regions using a YOLOv8 model trained on
    DocLayNet.

    Usage::

        detector = LayoutDetector("models/yolov8x_doclaynet.pt")
        regions = detector.detect(pil_image)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Load the YOLO model.

        Args:
            model_path: Path to the ``.pt`` weights file.  Falls back to
                        ``models/yolov8x_doclaynet.pt`` in the project root.
            device:     Force a device (``"cpu"``, ``"cuda:0"``, …).
                        ``None`` lets ultralytics pick automatically.
        """
        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {path}. "
                "Download a DocLayNet model and place it there."
            )

        self.model = YOLO(str(path), task="detect")
        self.device = device

        # Verify the model has the expected DocLayNet classes
        self._class_map = self.model.names  # {0: 'Caption', 1: 'Footnote', …}
        self._validate_classes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 1024,
    ) -> List[LayoutRegion]:
        """
        Run layout detection on a rendered page image.

        Args:
            image:          PIL Image (RGB) of the rendered PDF page.
            confidence:     Minimum confidence to keep a detection.
            iou_threshold:  IoU threshold for NMS (non-max suppression).
            image_size:     Inference resolution (longer side). Larger
                            values are slower but catch small elements.

        Returns:
            List of ``LayoutRegion`` sorted top-to-bottom by bbox y0.
        """
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            imgsz=image_size,
            device=self.device,
            verbose=False,
        )

        regions: List[LayoutRegion] = []

        if not results or len(results) == 0:
            return regions

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return regions

        # boxes.xyxy  → Tensor[N, 4]  (pixel coords)
        # boxes.conf  → Tensor[N]
        # boxes.cls   → Tensor[N]
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(coords, confs, classes):
            label = self._resolve_label(cls_id)
            regions.append(
                LayoutRegion(
                    label=label,
                    confidence=float(conf),
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                )
            )

        # Sort top-to-bottom, then left-to-right
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    # ------------------------------------------------------------------
    # Debug / visualisation
    # ------------------------------------------------------------------

    def detect_and_annotate(
        self,
        image: Image.Image,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 1024,
    ) -> Image.Image:
        """
        Run detection and return the image with bounding boxes drawn by
        ultralytics (handy for quick visual checks).
        """
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            imgsz=image_size,
            device=self.device,
            verbose=False,
        )
        if results:
            annotated = results[0].plot(pil=True)
            return Image.fromarray(annotated[..., ::-1])  # BGR → RGB
        return image

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_label(self, cls_id: int) -> LayoutLabel:
        """Map a YOLO class index to a LayoutLabel."""
        if cls_id in DOCLAYNET_INDEX_TO_LABEL:
            return DOCLAYNET_INDEX_TO_LABEL[cls_id]

        # Fallback: try matching the name string from the model
        name = self._class_map.get(cls_id, "").lower().replace("-", "_")
        for label in LayoutLabel:
            if label.name.lower() == name:
                return label

        return LayoutLabel.UNKNOWN

    def _validate_classes(self) -> None:
        """
        Warn (but don't crash) if the model's class names don't look
        like DocLayNet labels.
        """
        expected = {
            "caption",
            "footnote",
            "formula",
            "list-item",
            "page-footer",
            "page-header",
            "picture",
            "section-header",
            "table",
            "text",
            "title",
        }
        model_names = {v.lower() for v in self._class_map.values()}
        missing = expected - model_names
        if missing:
            print(
                f"[LayoutDetector] Warning: model is missing expected "
                f"DocLayNet classes: {missing}"
            )

    def __repr__(self) -> str:
        n = len(self._class_map)
        return f"LayoutDetector({n} classes, device={self.device})"
