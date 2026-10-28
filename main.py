"""Read-only last-hit companion.

Captures desktop frames, detects minions and their health bars, filters down to
only the minions that can currently be last-hit, orders them by priority
(nearest to the player anchor first), and visualizes only those candidates.
No game process memory access or mouse/keyboard input injection is performed.
"""

import argparse
import threading
import cv2
import numpy as np

from my_utils.loaders.frame_loader import CapturedFrame, FrameLoader
from my_utils.modules.algorithms import ObjectAssociator
from my_utils.modules.detector import DEFAULT_WEIGHTS_PATH, Detector
from my_utils.modules.last_hitter import LastHitAdvisor, LastHitCandidate, LastHitSettings
from my_utils.visualizers.annotator import AIAnnotator


WINDOW_TITLE = "Last Hit Companion (Read-Only)"
DEFAULT_MAX_HEALTH = {
    "caster": 300.0,
    "melee": 500.0,
    "siege": 1500.0,
}


class _AsyncDetectionWorker:
    """Runs YOLOv9 inference on the most recent frame without building a queue backlog."""

    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending_frame: CapturedFrame | None = None
        self._latest: tuple[CapturedFrame, dict] | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="LastHitDetectorWorker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: CapturedFrame) -> None:
        with self._lock:
            self._pending_frame = frame
            self._event.set()

    def get_latest(self) -> tuple[CapturedFrame, dict] | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            self._event.wait(timeout=0.1)
            with self._lock:
                if self._stopped:
                    return
                frame = self._pending_frame
                self._pending_frame = None
                self._event.clear()

            if frame is None:
                continue

            detection_result = self.detector(frame.image)
            with self._lock:
                self._latest = (frame, detection_result)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only League of Legends last-hit companion (no input or memory injection)."
    )
    parser.add_argument(
        "--monitor_idx",
        type=int,
        default=1,
        help="Monitor index to capture screenshots from (use --list-monitors to inspect).",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="List available monitors and exit.",
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Enable or disable the companion preview window (True/False).",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Disable the companion preview window.",
    )
    parser.add_argument(
        "--minion_color",
        choices=("red", "blue"),
        default="red",
        help="Color of enemy minions to evaluate for last hits.",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=720,
        help="Target frame height in pixels (default 720 for low-latency processing).",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=1280,
        help="Target frame width in pixels (default 1280 for low-latency processing).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLOv9 inference resolution (default 640 for lower latency; use 960 for max range).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run capture and inference synchronously instead of using the low-latency async pipeline.",
    )
    parser.add_argument(
        "--preview_W",
        type=int,
        default=None,
        help="Initial width of the resizable preview window (defaults to min(W, 1280)).",
    )
    parser.add_argument(
        "--preview_H",
        type=int,
        default=None,
        help="Initial height of the resizable preview window (defaults to min(H, 720)).",
    )
    parser.add_argument(
        "--mask_preview",
        action="store_true",
        help="Mask out the preview window's desktop rectangle from captured frames.",
    )
    parser.add_argument(
        "--effective_damage",
        "--effective-damage",
        "-effective_damage",
        dest="effective_damage",
        type=float,
        default=80.0,
        help="Conservative lower bound on post-mitigation basic-attack damage.",
    )
    parser.add_argument(
        "--caster_max_hp",
        type=float,
        default=DEFAULT_MAX_HEALTH["caster"],
        help="Conservative upper bound on caster minion max HP.",
    )
    parser.add_argument(
        "--melee_max_hp",
        type=float,
        default=DEFAULT_MAX_HEALTH["melee"],
        help="Conservative upper bound on melee minion max HP.",
    )
    parser.add_argument(
        "--siege_max_hp",
        type=float,
        default=DEFAULT_MAX_HEALTH["siege"],
        help="Conservative upper bound on siege/cannon minion max HP.",
    )
    parser.add_argument(
        "--health_margin",
        type=float,
        default=0.0,
        help="Safety margin added to measured health fraction (0 <= margin < 1).",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.25,
        help="Minimum detector confidence for minions and health bars.",
    )
    parser.add_argument(
        "--player_anchor",
        type=float,
        nargs=2,
        default=(0.5, 0.5),
        metavar=("X", "Y"),
        help="Normalized (x, y) player anchor in [0, 1] used to prioritize nearest last-hittable minions.",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=None,
        help="Optional maximum pixel distance from the player anchor.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to YOLOv9 weights file (auto-downloaded if missing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (e.g. cpu, cuda:0, mps). Auto-detected if omitted.",
    )
    return parser


def process_frame(
    captured_frame: CapturedFrame,
    detector: Detector,
    object_associator: ObjectAssociator,
    advisor: LastHitAdvisor,
    annotator: AIAnnotator | None = None,
    minion_color: str = "red",
) -> tuple[list[LastHitCandidate], np.ndarray | None]:
    """Evaluate a single frame and return (prioritized_last_hit_candidates, visualized_frame)."""
    detection_result = detector(captured_frame.image)
    matched_minions = object_associator(
        detection_result,
        img=captured_frame.image,
        minion_type=minion_color,
        excluded_regions=captured_frame.excluded_regions,
    )
    candidates = advisor.rank(matched_minions, captured_frame.image.shape)
    visualized = (
        annotator(candidates=candidates, img=captured_frame.image)
        if annotator is not None
        else None
    )
    return candidates, visualized


def evaluate_with_latest_detections(
    captured_frame: CapturedFrame,
    latest_detection: tuple[CapturedFrame, dict] | None,
    object_associator: ObjectAssociator,
    advisor: LastHitAdvisor,
    annotator: AIAnnotator | None = None,
    minion_color: str = "red",
) -> tuple[list[LastHitCandidate], np.ndarray | None]:
    """Re-evaluate health bars on the live frame using the newest detector boxes."""
    if latest_detection is None:
        visualized = (
            annotator(candidates=[], img=captured_frame.image, copy=False)
            if annotator is not None
            else None
        )
        return [], visualized

    det_frame, detection_result = latest_detection
    matched_minions = object_associator(
        detection_result,
        img=captured_frame.image,
        minion_type=minion_color,
        excluded_regions=captured_frame.excluded_regions,
    )
    if (
        not matched_minions
        and det_frame is not captured_frame
        and (captured_frame.captured_at - det_frame.captured_at) <= 0.6
    ):
        matched_minions = object_associator(
            detection_result,
            img=det_frame.image,
            minion_type=minion_color,
            excluded_regions=det_frame.excluded_regions,
        )

    candidates = advisor.rank(matched_minions, captured_frame.image.shape)
    visualized = (
        annotator(candidates=candidates, img=captured_frame.image, copy=False)
        if annotator is not None
        else None
    )
    return candidates, visualized


def _get_preview_window_rect(window_title: str) -> tuple[int, int, int, int] | None:
    try:
        rect = cv2.getWindowImageRect(window_title)
    except cv2.error:
        return None
    if rect is None or len(rect) != 4:
        return None
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def _init_preview_window(
    window_title: str,
    frame_w: int,
    frame_h: int,
    preview_w: int | None,
    preview_h: int | None,
) -> None:
    """Create a resizable preview window that scales the full frame instead of cropping it."""
    target_w = preview_w if (preview_w is not None and preview_w > 0) else min(frame_w, 1280)
    target_h = preview_h if (preview_h is not None and preview_h > 0) else min(frame_h, 720)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, int(target_w), int(target_h))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_monitors:
        for idx, monitor in enumerate(FrameLoader.list_monitors()):
            print(f"[{idx}] {monitor}")
        return 0

    settings = LastHitSettings(
        effective_damage=args.effective_damage,
        max_health={
            "caster": args.caster_max_hp,
            "melee": args.melee_max_hp,
            "siege": args.siege_max_hp,
        },
        health_margin=args.health_margin,
        min_confidence=args.min_confidence,
        player_anchor=tuple(args.player_anchor),
        max_distance=args.max_distance,
    )

    detector = Detector(
        weight_path=args.weights,
        device=args.device,
        imgsz=(args.imgsz, args.imgsz),
    )
    object_associator = ObjectAssociator(min_confidence=args.min_confidence)
    advisor = LastHitAdvisor(settings)
    annotator = AIAnnotator() if args.visualize else None

    worker = None if args.sync else _AsyncDetectionWorker(detector)
    if worker is not None:
        worker.start()

    window_created = False
    warned_black_frame = False
    with FrameLoader(monitor_idx=args.monitor_idx, width=args.W, height=args.H) as frame_loader:
        try:
            while True:
                if args.visualize and window_created and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                window_rect = (
                    _get_preview_window_rect(WINDOW_TITLE)
                    if (args.mask_preview and window_created)
                    else None
                )
                captured = frame_loader.grab(window_rect=window_rect)
                if not warned_black_frame and not np.any(captured.image):
                    print(
                        "[WARNING] Captured frame is completely black. "
                        "If League of Legends is in Exclusive Fullscreen mode, switch in-game Video settings "
                        "to 'Borderless' or 'Windowed'."
                    )
                    warned_black_frame = True

                if worker is not None:
                    worker.submit(captured)
                    _, visualized = evaluate_with_latest_detections(
                        captured_frame=captured,
                        latest_detection=worker.get_latest(),
                        object_associator=object_associator,
                        advisor=advisor,
                        annotator=annotator,
                        minion_color=args.minion_color,
                    )
                else:
                    _, visualized = process_frame(
                        captured_frame=captured,
                        detector=detector,
                        object_associator=object_associator,
                        advisor=advisor,
                        annotator=annotator,
                        minion_color=args.minion_color,
                    )

                if args.visualize and visualized is not None:
                    if not window_created:
                        _init_preview_window(
                            WINDOW_TITLE,
                            frame_w=visualized.shape[1],
                            frame_h=visualized.shape[0],
                            preview_w=args.preview_W,
                            preview_h=args.preview_H,
                        )
                        window_created = True
                    cv2.imshow(WINDOW_TITLE, visualized)
        finally:
            if worker is not None:
                worker.stop()
            if window_created:
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
