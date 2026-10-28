"""YOLOv9 minion and health-bar detector wrapper."""

from pathlib import Path
import sys

import numpy as np

from my_utils.data_structures.labels import normalize_label


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINION_LABELS = {
    "red": frozenset({"red caster minion", "red melee minion", "red siege minion"}),
    "blue": frozenset({"blue caster minion", "blue melee minion", "blue siege minion"}),
}
HEALTH_BAR_LABELS = {
    "red": "red health bar",
    "blue": "blue health bar",
}
KNOWN_LABELS = (
    MINION_LABELS["red"]
    | MINION_LABELS["blue"]
    | frozenset(HEALTH_BAR_LABELS.values())
)


def select_default_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda:0"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _load_yolov9_symbols():
    yolov9_dir = PROJECT_ROOT / "yolov9"
    if yolov9_dir.is_dir() and str(yolov9_dir) not in sys.path:
        sys.path.insert(0, str(yolov9_dir))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from yolov9.models.common import DetectMultiBackend
        from yolov9.utils.augmentations import letterbox
        from yolov9.utils.general import non_max_suppression, scale_boxes
    except ImportError:
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

    return DetectMultiBackend, letterbox, non_max_suppression, scale_boxes


def _unwrap_predictions(raw_pred):
    """Extract the primary detection tensor from single- or dual-head YOLOv9 outputs."""
    pred = raw_pred[0] if isinstance(raw_pred, (list, tuple)) else raw_pred
    if isinstance(pred, (list, tuple)):
        pred = pred[1] if len(pred) > 1 else pred[0]
    return pred


class Detector:
    def __init__(
        self,
        weight_path: str | Path = PROJECT_ROOT / "weights/best.pt",
        device: str | None = None,
        dnn: bool = False,
        half: bool = False,
        imgsz: tuple[int, int] = (960, 960),
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        backend=None,
    ):
        if not 0 < float(conf_thres) <= 1:
            raise ValueError("confidence threshold must be in (0, 1]")
        if not 0 < float(iou_thres) <= 1:
            raise ValueError("IoU threshold must be in (0, 1]")
        self.imgsz = imgsz
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        if backend is not None:
            self.model = backend
            self._letterbox = None
            self._nms = None
            self._scale_boxes = None
            self._torch = None
        else:
            import torch

            weight_path = Path(weight_path)
            if not weight_path.is_file():
                raise FileNotFoundError(f"model weights not found: {weight_path}")
            resolved_device = device or select_default_device(torch)
            DetectMultiBackend, letterbox, non_max_suppression, scale_boxes = _load_yolov9_symbols()
            self.model = DetectMultiBackend(
                str(weight_path), device=resolved_device, dnn=dnn, fp16=half
            )
            self._letterbox = letterbox
            self._nms = non_max_suppression
            self._scale_boxes = scale_boxes
            self._torch = torch

        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, list):
            raw_names = dict(enumerate(raw_names))
        self.names = {int(k): normalize_label(v) for k, v in raw_names.items()}
        recognized = set(self.names.values()) & KNOWN_LABELS
        if not recognized:
            raise ValueError("model class names do not contain recognized minion or health-bar labels")

        self.stride = int(getattr(self.model, "stride", 32))
        self.pt = bool(getattr(self.model, "pt", True))
        self.object_counts_template = {label: 0 for label in KNOWN_LABELS}

    def __call__(self, img: np.ndarray) -> dict:
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("input image must be a 3-channel BGR array")

        red_minions, blue_minions, red_health_bars, blue_health_bars, cur_object_counts = (
            self.detect_objects(img)
        )
        return {
            "red_minions": red_minions,
            "blue_minions": blue_minions,
            "red_health_bars": red_health_bars,
            "blue_health_bars": blue_health_bars,
            "cur_object_counts": cur_object_counts,
        }

    def detect_objects(self, img: np.ndarray):
        torch = self._torch
        im0 = np.ascontiguousarray(img)
        im = self._letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC BGR -> CHW RGB
        im = np.ascontiguousarray(im)

        with torch.no_grad():
            tensor = torch.from_numpy(im).to(self.model.device)
            tensor = tensor.half() if getattr(self.model, "fp16", False) else tensor.float()
            tensor /= 255.0
            if tensor.ndim == 3:
                tensor = tensor[None]

            raw_pred = self.model(tensor, augment=False, visualize=False)
            pred = _unwrap_predictions(raw_pred)
            pred = self._nms(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        return self._classify_detections(pred, tensor.shape[2:], im0.shape)

    def _classify_detections(self, pred, input_hw, orig_shape):
        red_minions = []
        blue_minions = []
        red_health_bars = []
        blue_health_bars = []
        cur_object_counts = self.object_counts_template.copy()

        for det in pred:
            if det is None or len(det) == 0:
                continue
            det = det.clone() if hasattr(det, "clone") else np.array(det, copy=True)
            if self._scale_boxes is not None:
                det[:, :4] = self._scale_boxes(input_hw, det[:, :4], orig_shape)

            for *xyxy, conf, cls in reversed(det):
                box = tuple(float(v.item() if hasattr(v, "item") else v) for v in xyxy)
                confidence = float(conf.item() if hasattr(conf, "item") else conf)
                class_id = int(cls.item() if hasattr(cls, "item") else cls)
                label = self.names.get(class_id)
                if label not in KNOWN_LABELS:
                    continue

                cur_object_counts[label] += 1
                if label in MINION_LABELS["red"]:
                    red_minions.append((box, label, confidence))
                elif label in MINION_LABELS["blue"]:
                    blue_minions.append((box, label, confidence))
                elif label == HEALTH_BAR_LABELS["red"]:
                    red_health_bars.append((box, confidence))
                elif label == HEALTH_BAR_LABELS["blue"]:
                    blue_health_bars.append((box, confidence))

        return red_minions, blue_minions, red_health_bars, blue_health_bars, cur_object_counts
