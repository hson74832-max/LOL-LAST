"""YOLOv9 minion and health-bar detector wrapper."""

from contextlib import contextmanager
import os
import pathlib
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

from my_utils.data_structures.labels import normalize_label


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "best.pt"
GDRIVE_WEIGHTS_ID = "135Bt-YX-Aqa_oaBxN220YSe9uGWiuljo"
GDRIVE_WEIGHTS_URL = (
    f"https://drive.google.com/file/d/{GDRIVE_WEIGHTS_ID}/view?usp=share_link"
)
MIN_VALID_WEIGHTS_BYTES = 1_000_000  # Valid YOLOv9 .pt checkpoints are multiple MBs

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


def _is_valid_checkpoint_file(path: Path) -> bool:
    """Return True if path exists, is non-empty, and is not an HTML error page."""
    try:
        if not path.is_file() or path.stat().st_size < 1024:
            return False
        with path.open("rb") as fh:
            header = fh.read(64).lstrip()
        if header.lower().startswith((b"<!doctype html", b"<html")):
            return False
        return True
    except OSError:
        return False


def _is_default_weight_request(weight_path: Path) -> bool:
    normalized = weight_path.as_posix().lstrip("./")
    return normalized in {"weights/best.pt", "best.pt"} or (
        weight_path.resolve(strict=False) == DEFAULT_WEIGHTS_PATH.resolve(strict=False)
    )


def _find_local_weight_candidates(weight_path: Path) -> Path | None:
    """Search explicit path, repo-relative path, and common Windows download locations."""
    direct_candidates = [weight_path]
    if not weight_path.is_absolute():
        direct_candidates.append(PROJECT_ROOT / weight_path)

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    if not _is_default_weight_request(weight_path):
        return None

    fallback_candidates = [
        DEFAULT_WEIGHTS_PATH,
        PROJECT_ROOT / "weights" / "best.pt.pt",
        PROJECT_ROOT / "best.pt",
        PROJECT_ROOT / "best.pt.pt",
    ]
    weights_dir = PROJECT_ROOT / "weights"
    if weights_dir.is_dir():
        pt_files = sorted(p for p in weights_dir.glob("*.pt") if p.is_file())
        if len(pt_files) == 1:
            fallback_candidates.append(pt_files[0])

    for candidate in fallback_candidates:
        if _is_valid_checkpoint_file(candidate):
            if candidate != DEFAULT_WEIGHTS_PATH:
                DEFAULT_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(candidate, DEFAULT_WEIGHTS_PATH)
                    return DEFAULT_WEIGHTS_PATH
                except OSError:
                    return candidate
            return candidate

    downloads_dir = Path.home() / "Downloads"
    for name in ("best.pt", "best.pt.pt"):
        downloaded = downloads_dir / name
        if _is_valid_checkpoint_file(downloaded):
            DEFAULT_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(downloaded, DEFAULT_WEIGHTS_PATH)
                return DEFAULT_WEIGHTS_PATH
            except OSError:
                return downloaded

    return None


def _download_from_gdrive_requests(file_id: str, destination: Path) -> bool:
    """Fallback downloader using requests when gdown is not installed."""
    import requests

    session = requests.Session()
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    response = session.get(url, params=params, stream=True, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        html = response.text
        action_match = re.search(r'<form[^>]+action="([^"]+)"', html)
        form_url = action_match.group(1) if action_match else url
        form_params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html))
        if not form_params:
            form_params = params
        response = session.get(form_url, params=form_params, stream=True, timeout=60)
        response.raise_for_status()

    tmp_path = destination.with_suffix(".pt.part")
    try:
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        if not _is_valid_checkpoint_file(tmp_path) or tmp_path.stat().st_size < MIN_VALID_WEIGHTS_BYTES:
            tmp_path.unlink(missing_ok=True)
            return False
        tmp_path.replace(destination)
        return True
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def ensure_default_weights(destination: Path = DEFAULT_WEIGHTS_PATH) -> Path:
    """Ensure the default YOLOv9 weights exist locally, downloading them if needed."""
    destination = Path(destination)
    existing = _find_local_weight_candidates(destination)
    if existing is not None:
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".pt.part")
    print(f"[INFO] Model weights not found at {destination}. Downloading weights/best.pt ...")

    try:
        import gdown  # type: ignore[import-not-found]

        downloaded = gdown.download(
            id=GDRIVE_WEIGHTS_ID,
            output=str(tmp_path),
            quiet=False,
            fuzzy=True,
        )
        if downloaded and _is_valid_checkpoint_file(tmp_path) and tmp_path.stat().st_size >= MIN_VALID_WEIGHTS_BYTES:
            tmp_path.replace(destination)
            return destination
        tmp_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)

    try:
        if _download_from_gdrive_requests(GDRIVE_WEIGHTS_ID, destination):
            return destination
    except Exception as exc:
        raise FileNotFoundError(
            f"model weights not found at {destination} and automatic download failed ({exc}).\n"
            f"Please download best.pt from:\n  {GDRIVE_WEIGHTS_URL}\n"
            f"and save it to:\n  {destination}"
        ) from exc

    raise FileNotFoundError(
        f"model weights not found at {destination} and automatic download did not return a valid checkpoint.\n"
        f"Please download best.pt from:\n  {GDRIVE_WEIGHTS_URL}\n"
        f"and save it to:\n  {destination}"
    )


def resolve_weight_path(
    weight_path: str | Path = DEFAULT_WEIGHTS_PATH,
    auto_download: bool = True,
) -> Path:
    candidate_path = Path(weight_path)
    found = _find_local_weight_candidates(candidate_path)
    if found is not None:
        return found

    if auto_download and _is_default_weight_request(candidate_path):
        return ensure_default_weights(DEFAULT_WEIGHTS_PATH)

    expected = (
        candidate_path
        if candidate_path.is_absolute()
        else (PROJECT_ROOT / candidate_path)
    )
    raise FileNotFoundError(
        f"model weights not found: {expected}\n"
        f"Download the trained YOLOv9 weights from:\n  {GDRIVE_WEIGHTS_URL}\n"
        f"and place them at:\n  {DEFAULT_WEIGHTS_PATH}"
    )


@contextmanager
def _torch_load_compat(torch_module):
    """Ensure cross-platform PosixPath unpickling on Windows and weights_only=False on PyTorch 2.6+."""
    orig_torch_load = torch_module.load
    orig_posix_path = getattr(pathlib, "PosixPath", None)

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    torch_module.load = _compat_load
    if os.name == "nt" and orig_posix_path is not None:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        torch_module.load = orig_torch_load
        if os.name == "nt" and orig_posix_path is not None:
            pathlib.PosixPath = orig_posix_path  # type: ignore[misc,assignment]


def _load_yolov9_symbols():
    yolov9_dir = PROJECT_ROOT / "yolov9"
    if not (yolov9_dir / "models" / "common.py").is_file() and shutil.which("git"):
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/WongKinYiu/yolov9.git",
                    str(yolov9_dir),
                ],
                check=True,
            )
        except Exception:
            pass

    if yolov9_dir.is_dir() and str(yolov9_dir) not in sys.path:
        sys.path.insert(0, str(yolov9_dir))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    try:
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes
    except ImportError:
        from yolov9.models.common import DetectMultiBackend
        from yolov9.utils.augmentations import letterbox
        from yolov9.utils.general import non_max_suppression, scale_boxes

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
        weight_path: str | Path = DEFAULT_WEIGHTS_PATH,
        device: str | None = None,
        dnn: bool = False,
        half: bool | None = None,
        imgsz: tuple[int, int] | int = (640, 640),
        conf_thres: float = 0.15,
        iou_thres: float = 0.45,
        backend=None,
        auto_download: bool = True,
    ):
        if not 0 < float(conf_thres) <= 1:
            raise ValueError("confidence threshold must be in (0, 1]")
        if not 0 < float(iou_thres) <= 1:
            raise ValueError("IoU threshold must be in (0, 1]")
        if isinstance(imgsz, int):
            if imgsz <= 0:
                raise ValueError("imgsz must be positive")
            self.imgsz = (int(imgsz), int(imgsz))
        else:
            if len(imgsz) != 2 or min(imgsz) <= 0:
                raise ValueError("imgsz must be a pair of positive integers")
            self.imgsz = (int(imgsz[0]), int(imgsz[1]))

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

            resolved_weight_path = resolve_weight_path(
                weight_path, auto_download=auto_download
            )
            resolved_device_str = device or select_default_device(torch)
            resolved_device = torch.device(resolved_device_str)
            use_half = (resolved_device.type == "cuda") if half is None else bool(half)
            if resolved_device.type == "cuda" and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True

            DetectMultiBackend, letterbox, non_max_suppression, scale_boxes = _load_yolov9_symbols()
            with _torch_load_compat(torch):
                self.model = DetectMultiBackend(
                    str(resolved_weight_path),
                    device=resolved_device,
                    dnn=dnn,
                    fp16=use_half,
                )
            if hasattr(self.model, "warmup") and resolved_device.type == "cuda":
                try:
                    self.model.warmup(imgsz=(1, 3, *self.imgsz))
                except Exception:
                    pass
            print(
                f"[INFO] Detector initialized on device={resolved_device} "
                f"(fp16={getattr(self.model, 'fp16', use_half)}, imgsz={self.imgsz})"
            )
            if resolved_device.type == "cpu":
                print(
                    "[NOTE] Running on CPU. If you have an NVIDIA GPU on Windows, "
                    "re-run run.bat or install_requirements.bat to enable CUDA acceleration."
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
        im = self._letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC BGR -> CHW RGB

        with torch.no_grad():
            tensor = torch.from_numpy(im).to(self.model.device, non_blocking=True)
            tensor = tensor.half() if getattr(self.model, "fp16", False) else tensor.float()
            tensor /= 255.0
            if tensor.ndim == 3:
                tensor = tensor[None]

            raw_pred = self.model(tensor, augment=False, visualize=False)
            pred = _unwrap_predictions(raw_pred)
            pred = self._nms(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

        return self._classify_detections(pred, tensor.shape[2:], img.shape)

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

            det_np = det.detach().cpu().numpy() if hasattr(det, "detach") else np.asarray(det)
            for row in det_np[::-1]:
                box = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                confidence = float(row[4])
                class_id = int(row[5])
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
