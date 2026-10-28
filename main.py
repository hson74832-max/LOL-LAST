"""Read-only last-hit companion.

Captures desktop frames, detects minions and their health bars, filters down to
only the minions that can currently be last-hit, orders them by priority
(nearest to the player anchor first), and visualizes only those candidates.
No game process memory access or mouse/keyboard input injection is performed.
"""

import argparse
import threading
import time
import cv2
import numpy as np

from my_utils.loaders.frame_loader import CapturedFrame, FrameLoader
from my_utils.modules.algorithms import ObjectAssociator
from my_utils.modules.detector import DEFAULT_WEIGHTS_PATH, Detector, diagnose_devices
from my_utils.modules.diagnostics import DebugReporter, describe_thresholds, stats_from_frame
from my_utils.modules.performance import (
    FpsLimiter,
    StageTimer,
    apply_process_priority,
    capture_and_processing_tips,
    set_torch_threads,
    validate_fps,
)
from my_utils.modules.last_hitter import LastHitAdvisor, LastHitCandidate, LastHitSettings
from my_utils.modules.stability import CandidateGate
from my_utils.modules.tracking import MinionTracker
from my_utils.visualizers.annotator import AIAnnotator
from my_utils.visualizers.overlay import (
    DEFAULT_MARKER_GAP,
    DesktopOverlay,
    OverlayUnavailable,
)


WINDOW_TITLE = "Last Hit Companion (Read-Only)"
# Detections older than this are never drawn on a live frame: a stale bounding box
# makes the companion point at minions that are already dead.
DEFAULT_MAX_DETECTION_AGE = 0.35
# Pacing defaults: inference is the expensive stage, so it runs well below the loop rate.
DEFAULT_INFER_FPS = 12.0
DEFAULT_LOOP_FPS = 30.0
DEFAULT_OVERLAY_FPS = 30.0
# Confirmed markers are held this long while a minion's health bar flickers.
DEFAULT_HOLD_MS = 250.0
DEFAULT_CONFIRM_FRAMES = 2
DEFAULT_BAR_MIN_CONFIDENCE = 0.20

DEFAULT_MAX_HEALTH = {
    "caster": 300.0,
    "melee": 500.0,
    "siege": 1500.0,
}


class _AsyncDetectionWorker:
    """Runs YOLOv9 inference on the most recent frame without building a queue backlog."""

    def __init__(self, detector: Detector, max_fps: float = 0.0) -> None:
        self.detector = detector
        self.last_inference_seconds: float | None = None
        self.inference_count = 0
        # Detection is the expensive stage: pacing it is what stops the game stuttering.
        self.max_fps = validate_fps(max_fps, "inference fps")
        self._min_interval = 1.0 / self.max_fps if self.max_fps else 0.0
        self._next_inference = 0.0
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending_frame: CapturedFrame | None = None
        self._latest: tuple[CapturedFrame, dict] | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="LastHitDetectorWorker", daemon=True)

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

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
            self._event.wait(timeout=0.05)
            if self.stopped:
                return

            if self._min_interval:
                now = time.perf_counter()
                if now < self._next_inference:
                    # Keep the newest pending frame and wait for the next slot instead
                    # of queueing work that would keep the GPU busy back-to-back.
                    time.sleep(min(0.05, self._next_inference - now))
                    continue

            with self._lock:
                if self._stopped:
                    return
                frame = self._pending_frame
                self._pending_frame = None
                self._event.clear()

            if frame is None:
                continue

            if self._min_interval:
                self._next_inference = time.perf_counter() + self._min_interval
            started = time.perf_counter()
            detection_result = self.detector(frame.image)
            elapsed = time.perf_counter() - started
            with self._lock:
                self._latest = (frame, detection_result)
                self.last_inference_seconds = elapsed
                self.inference_count += 1


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _fps(value: str) -> float:
    try:
        return validate_fps(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
    return number


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
        "--overlay",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help=(
            "Draw the last-hit markers as a click-through, always-on-top overlay on the "
            "game screen itself (True/False, default True). Use --no-overlay to disable."
        ),
    )
    parser.add_argument(
        "--no-overlay",
        dest="overlay",
        action="store_false",
        help="Disable the desktop overlay and only draw markers in the preview window.",
    )
    parser.add_argument(
        "--overlay_gap",
        type=_non_negative_float,
        default=DEFAULT_MARKER_GAP,
        help=(
            "Distance in frame pixels between a minion's bounding box top and its overlay "
            "marker, kept above the health bar (default 16)."
        ),
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
        help=(
            "Minimum detector confidence for minion bodies (default 0.25). "
            "Raise it (0.4-0.5) to suppress phantom boxes."
        ),
    )
    parser.add_argument(
        "--min_bar_confidence",
        type=float,
        default=DEFAULT_BAR_MIN_CONFIDENCE,
        help=(
            "Minimum detector confidence for health bars (default 0.20; bars are "
            "separate because low-HP bars score lower). Raise it (0.35-0.5) to reject "
            "corpse flashes and spell-effect artifacts."
        ),
    )
    parser.add_argument(
        "--max_detection_age",
        type=_non_negative_float,
        default=DEFAULT_MAX_DETECTION_AGE,
        help=(
            "Maximum age in seconds of the YOLO boxes reused on a live frame "
            f"(default {DEFAULT_MAX_DETECTION_AGE}; 0 disables the limit). Lower it "
            "(0.15-0.25) for stricter markers; raise it if your GPU/CPU cannot infer "
            "that fast."
        ),
    )
    parser.add_argument(
        "--confirm_frames",
        type=int,
        default=DEFAULT_CONFIRM_FRAMES,
        help=(
            "Number of consecutive frames a candidate must be seen in before it is "
            f"drawn (default {DEFAULT_CONFIRM_FRAMES}; 1 disables confirmation). "
            "Higher values remove flicker and one-frame misreads but react slower."
        ),
    )
    parser.add_argument(
        "--hold_ms",
        type=_non_negative_float,
        default=DEFAULT_HOLD_MS,
        help=(
            "Keep a confirmed marker for this many milliseconds after its minion stops "
            f"being detected (default {DEFAULT_HOLD_MS:.0f}; 0 disables holding). This stops "
            "a nearly-dead minion's marker from blinking off while its 2px health bar is "
            "unreadable. Held markers keep the last HP reading and are drawn dimmed."
        ),
    )
    parser.add_argument(
        "--overlay_labels",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help=(
            "Show the estimated remaining HP number above each marker on the desktop "
            "overlay (default True). Disable with --no-overlay_labels."
        ),
    )
    parser.add_argument(
        "--no-overlay_labels",
        dest="overlay_labels",
        action="store_false",
        help="Hide the HP numbers on the desktop overlay.",
    )
    parser.add_argument(
        "--overlay_label_mode",
        choices=("hp", "hp_damage", "percent"),
        default="hp",
        help=(
            "'hp' shows the estimated remaining HP (default), 'hp_damage' shows "
            "'HP/your damage' so the kill margin is visible, 'percent' shows the bar "
            "percentage."
        ),
    )
    parser.add_argument(
        "--bar_locator",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help=(
            "Read a health bar geometrically around each detected minion body when YOLOv9 "
            "reports no bar for it (default True). Disable with --no-bar_locator to require "
            "a detected bar class."
        ),
    )
    parser.add_argument(
        "--no-bar_locator",
        dest="bar_locator",
        action="store_false",
        help="Require YOLOv9 health-bar detections; do not read bars from the pixels.",
    )
    parser.add_argument(
        "--stale_frame_fallback",
        action="store_true",
        help=(
            "Legacy behaviour: when the live frame yields no match, re-run detection on "
            "the previous frame. Off by default because it shows dead/outdated minions."
        ),
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
        help=(
            "Inference device: cpu, cuda:0 (NVIDIA), mps (Apple), directml (AMD/Intel GPU "
            "on Windows, needs torch-directml). Auto-detected if omitted."
        ),
    )
    parser.add_argument(
        "--strict_device",
        action="store_true",
        help=(
            "Fail with an error if the requested --device is unavailable instead of "
            "warning and falling back to CPU inference."
        ),
    )
    parser.add_argument(
        "--infer_fps",
        type=_fps,
        default=DEFAULT_INFER_FPS,
        help=(
            "Maximum YOLO inferences per second (default 12; 0 = unlimited). "
            "This is the main lever for how much the game is slowed down: detection at "
            "10-15 fps is plenty for last hitting, while faster health-bar re-reads keep "
            "the markers responsive."
        ),
    )
    parser.add_argument(
        "--max_fps",
        type=_fps,
        default=DEFAULT_LOOP_FPS,
        help=(
            "Maximum capture/processing loop rate in frames per second (default 30; "
            "0 = unlimited). Lowering it reduces CPU load and window drawing."
        ),
    )
    parser.add_argument(
        "--overlay_fps",
        type=_fps,
        default=DEFAULT_OVERLAY_FPS,
        help="Maximum overlay redraw rate (default 30).",
    )
    parser.add_argument(
        "--priority",
        choices=("normal", "below", "lowest"),
        default="below",
        help=(
            "Priority of this companion process (default 'below'). 'lowest' gives the game "
            "the most CPU time; 'normal' restores default scheduling. Windows: "
            "below=below-normal, lowest=idle priority class."
        ),
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=None,
        help=(
            "Limit PyTorch intra-op threads (matters on CPU inference; default: leave "
            "PyTorch's choice). On a 8-core CPU, e.g. --torch_threads 4 keeps cores free."
        ),
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Print per-frame diagnostics (brightness, detected bodies/bars per color, "
            "matched minions, HP readings, inference latency, box age) and explain why no "
            "marker is drawn. Also saves an annotated frame to --debug_frames_dir."
        ),
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable diagnostics output.",
    )
    parser.add_argument(
        "--debug_frames_dir",
        type=str,
        default="frames/debug",
        help="Where --debug writes the annotated 'latest.png' (default frames/debug).",
    )
    parser.add_argument(
        "--debug_interval",
        type=_non_negative_float,
        default=1.0,
        help="Seconds between diagnostics lines / debug frame writes (default 1.0).",
    )
    parser.add_argument(
        "--check-devices",
        action="store_true",
        help=(
            "Print which inference backends (CUDA / MPS / DirectML / CPU) are available, "
            "what is missing and how to install it, then exit."
        ),
    )
    return parser


def process_frame(
    captured_frame: CapturedFrame,
    detector: Detector,
    object_associator: ObjectAssociator,
    advisor: LastHitAdvisor,
    annotator: AIAnnotator | None = None,
    minion_color: str = "red",
    gate: CandidateGate | None = None,
    pipeline_stats: dict | None = None,
    tracker: MinionTracker | None = None,
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
    if pipeline_stats is not None:
        pipeline_stats.update(
            matched=len(matched_minions),
            eligible=len(candidates),
            health=[m.health_level for m in matched_minions],
            detection_result=detection_result,
            **getattr(object_associator, "last_stats", {}),
        )
    if gate is not None:
        candidates = gate.update(
            candidates, captured_frame.image.shape, generation=captured_frame
        )
    if tracker is not None:
        # Frame timestamps are monotonic, so hold timing survives a stalled loop.
        candidates = tracker.update(
            candidates, captured_frame.image.shape, now=captured_frame.captured_at
        )
    if pipeline_stats is not None:
        pipeline_stats["kept"] = len(candidates)
        pipeline_stats["held"] = sum(1 for c in candidates if c.held)
    visualized = (
        annotator(candidates=candidates, img=captured_frame.image)
        if annotator is not None
        else None
    )
    return candidates, visualized


def detection_age(
    captured_frame: CapturedFrame,
    latest_detection: tuple[CapturedFrame, dict] | None,
) -> float | None:
    """Age in seconds of the detector boxes relative to the live frame, if any."""
    if latest_detection is None:
        return None
    det_frame = latest_detection[0]
    if det_frame is captured_frame:
        return 0.0
    return max(0.0, captured_frame.captured_at - det_frame.captured_at)


def evaluate_with_latest_detections(
    captured_frame: CapturedFrame,
    latest_detection: tuple[CapturedFrame, dict] | None,
    object_associator: ObjectAssociator,
    advisor: LastHitAdvisor,
    annotator: AIAnnotator | None = None,
    minion_color: str = "red",
    gate: CandidateGate | None = None,
    max_detection_age: float | None = DEFAULT_MAX_DETECTION_AGE,
    allow_stale_frame_fallback: bool = False,
    pipeline_stats: dict | None = None,
    tracker: MinionTracker | None = None,
) -> tuple[list[LastHitCandidate], np.ndarray | None]:
    """Re-evaluate health bars on the live frame using the newest detector boxes.

    Boxes older than ``max_detection_age`` are discarded instead of being projected
    onto the live frame: a delayed box is how the companion ends up highlighting a
    minion that died, or another minion entirely, while the camera scrolls.
    """
    def _empty():
        visualized = (
            annotator(candidates=[], img=captured_frame.image, copy=False)
            if annotator is not None
            else None
        )
        if gate is not None:
            gate.reset()
        if tracker is not None:
            tracker.reset()
        if pipeline_stats is not None:
            pipeline_stats.update(matched=0, eligible=0, kept=0, health=[], stale=True)
        return [], visualized

    if latest_detection is None:
        return _empty()

    age = detection_age(captured_frame, latest_detection)
    # 0 (or None) disables the age limit, e.g. for very slow CPU-only machines.
    if max_detection_age and age is not None and age > max_detection_age:
        if pipeline_stats is not None:
            pipeline_stats["stale_age"] = age
            pipeline_stats["stale"] = True
        return _empty()

    det_frame, detection_result = latest_detection
    matched_minions = object_associator(
        detection_result,
        img=captured_frame.image,
        minion_type=minion_color,
        excluded_regions=captured_frame.excluded_regions,
    )
    # Opt-in only: re-running association on an old frame re-creates the exact stale
    # markers this companion is meant to avoid (dead minions, outdated health).
    if not matched_minions and allow_stale_frame_fallback and det_frame is not captured_frame:
        matched_minions = object_associator(
            detection_result,
            img=det_frame.image,
            minion_type=minion_color,
            excluded_regions=det_frame.excluded_regions,
        )

    candidates = advisor.rank(matched_minions, captured_frame.image.shape)
    if pipeline_stats is not None:
        pipeline_stats.update(
            matched=len(matched_minions),
            eligible=len(candidates),
            health=[m.health_level for m in matched_minions],
            stale=False,
            detection_result=detection_result,
            **getattr(object_associator, "last_stats", {}),
        )
    if gate is not None:
        # Sightings are counted per detector result, not per capture frame.
        candidates = gate.update(
            candidates, captured_frame.image.shape, generation=det_frame
        )
    if tracker is not None:
        # Held markers are re-read against the live frame, so their HP stays current.
        candidates = tracker.update(
            candidates, captured_frame.image.shape, now=captured_frame.captured_at
        )
    if pipeline_stats is not None:
        pipeline_stats["kept"] = len(candidates)
        pipeline_stats["held"] = sum(1 for c in candidates if c.held)
    # The live frame was just handed to the detection worker; annotating it in
    # place would let YOLOv9 read half-drawn boxes, so always render on a copy.
    visualized = (
        annotator(candidates=candidates, img=captured_frame.image, copy=True)
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

    if args.check_devices:
        print("[INFO] Inference backend diagnostics")
        for line in diagnose_devices():
            print(f"[INFO] {line}" if line else "")
        return 0

    if args.confirm_frames < 1:
        parser.error("--confirm_frames must be at least 1")
    if not 0 < args.min_bar_confidence <= 1:
        parser.error("--min_bar_confidence must be greater than 0 and at most 1")

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

    priority_status = apply_process_priority(args.priority)
    if priority_status:
        print(f"[INFO] {priority_status}.")
    if args.torch_threads is not None:
        threads_status = set_torch_threads(args.torch_threads)
        if threads_status:
            print(f"[INFO] {threads_status}.")

    detector = Detector(
        weight_path=args.weights,
        device=args.device,
        imgsz=(args.imgsz, args.imgsz),
        strict_device=args.strict_device,
    )
    object_associator = ObjectAssociator(
        min_confidence=args.min_confidence,
        bar_min_confidence=args.min_bar_confidence,
        fallback_to_locator=args.bar_locator,
    )
    advisor = LastHitAdvisor(settings)
    gate = CandidateGate(confirm_frames=args.confirm_frames)
    tracker = MinionTracker(hold_ms=args.hold_ms)
    print(
        f"[INFO] Markers require {args.confirm_frames} consecutive frame(s); detector boxes "
        f"older than {args.max_detection_age * 1000:.0f} ms are ignored; "
        f"min body/bar confidence {args.min_confidence:.2f}/{args.min_bar_confidence:.2f}; "
        f"geometric health-bar reading {'on' if args.bar_locator else 'off'}; "
        f"markers held {args.hold_ms:.0f} ms after a minion disappears."
    )
    print(
        f"[INFO] Pacing: capture loop {args.max_fps or 'unlimited'} fps, detection "
        f"{args.infer_fps or 'unlimited'} fps, overlay {args.overlay_fps or 'unlimited'} fps."
    )
    if args.sync and args.infer_fps:
        print(
            "[NOTE] --sync runs inference inside the capture loop, so --infer_fps does not "
            "apply; lower --max_fps to pace it (the default async pipeline is recommended)."
        )
    annotator = AIAnnotator() if args.visualize else None
    reporter = (
        DebugReporter(
            minion_color=args.minion_color,
            frames_dir=args.debug_frames_dir,
            interval=args.debug_interval,
        )
        if args.debug
        else None
    )
    if reporter is not None:
        print("[INFO] Diagnostics enabled (--debug).")
        for line in describe_thresholds(settings, args.minion_color):
            print(f"[INFO] {line}")
        print(f"[INFO] Annotated frames are written to {reporter.frames_dir}/latest.png")
        for tip in capture_and_processing_tips(args.imgsz, args.W, args.H):
            print(f"[INFO] speed tip: {tip}")
        if not args.visualize and not args.overlay:
            print(
                "[NOTE] Both the preview window and the overlay are disabled, so no markers "
                "can be displayed; --debug output is the only feedback."
            )

    worker = None if args.sync else _AsyncDetectionWorker(detector, max_fps=args.infer_fps)
    if worker is not None:
        worker.start()

    window_created = False
    warned_black_frame = False
    warned_stale_detections = False
    limiter = FpsLimiter(args.max_fps)
    timings = StageTimer()
    overlay: DesktopOverlay | None = None
    overlay_failed = not args.overlay
    if args.overlay:
        print(
            "[INFO] Desktop overlay enabled: last-hit markers are drawn on the game screen "
            "itself. Press Ctrl+C to stop."
        )
    with FrameLoader(monitor_idx=args.monitor_idx, width=args.W, height=args.H) as frame_loader:
        try:
            while True:
                loop_started = time.perf_counter()
                if args.visualize and window_created and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                window_rect = (
                    _get_preview_window_rect(WINDOW_TITLE)
                    if (args.mask_preview and window_created)
                    else None
                )
                with timings.measure("capture"):
                    captured = frame_loader.grab(window_rect=window_rect)
                if not warned_black_frame and not np.any(captured.image):
                    print(
                        "[WARNING] Captured frame is completely black. "
                        "If League of Legends is in Exclusive Fullscreen mode, switch in-game Video settings "
                        "to 'Borderless' or 'Windowed'."
                    )
                    warned_black_frame = True

                pipeline_stats: dict | None = {} if reporter is not None else None
                evaluate_started = time.perf_counter()
                if worker is not None:
                    worker.submit(captured)
                    latest = worker.get_latest()
                    age = detection_age(captured, latest)
                    if (
                        age is not None
                        and args.max_detection_age
                        and age > args.max_detection_age
                        and not warned_stale_detections
                    ):
                        inference = worker.last_inference_seconds
                        print(
                            f"[WARNING] Detector boxes are {age * 1000:.0f} ms old, which is "
                            f"above --max_detection_age {args.max_detection_age * 1000:.0f} ms, "
                            "so markers are suppressed instead of being drawn on stale boxes.\n"
                            f"          Last inference took "
                            f"{'n/a' if inference is None else f'{inference * 1000:.0f} ms'} "
                            f"on device {detector.device_name}.\n"
                            "          Either speed up inference (run.bat --check-devices shows "
                            "whether the GPU is used) or raise --max_detection_age "
                            "(0 disables the limit). --debug explains every frame."
                        )
                        warned_stale_detections = True
                    candidates, visualized = evaluate_with_latest_detections(
                        captured_frame=captured,
                        latest_detection=latest,
                        object_associator=object_associator,
                        advisor=advisor,
                        annotator=annotator,
                        minion_color=args.minion_color,
                        gate=gate,
                        max_detection_age=args.max_detection_age,
                        allow_stale_frame_fallback=args.stale_frame_fallback,
                        pipeline_stats=pipeline_stats,
                        tracker=tracker,
                    )
                else:
                    candidates, visualized = process_frame(
                        captured_frame=captured,
                        detector=detector,
                        object_associator=object_associator,
                        advisor=advisor,
                        annotator=annotator,
                        minion_color=args.minion_color,
                        gate=gate,
                        pipeline_stats=pipeline_stats,
                        tracker=tracker,
                    )

                timings.add("evaluate", time.perf_counter() - evaluate_started)

                if reporter is not None:
                    detection_result = (pipeline_stats or {}).get("detection_result") or {}
                    if worker is not None:
                        age_seconds = detection_age(captured, latest)
                        inference_seconds = worker.last_inference_seconds
                    else:
                        age_seconds = 0.0
                        inference_seconds = None
                    if pipeline_stats and pipeline_stats.get("stale_age") is not None:
                        age_seconds = pipeline_stats["stale_age"]
                    stats = stats_from_frame(
                        image=captured.image,
                        detection_result=detection_result,
                        minion_color=args.minion_color,
                        pipeline=pipeline_stats,
                        detection_age_seconds=age_seconds,
                        inference_seconds=inference_seconds,
                    )
                    reporter.report(stats, timings=timings.averages())
                    reporter.save(
                        image=captured.image,
                        detection_result=detection_result,
                        candidates=candidates,
                    )

                if not overlay_failed and overlay is None:
                    try:
                        overlay = DesktopOverlay(
                            monitor=frame_loader.monitor,
                            primary_width=frame_loader.primary_monitor["width"],
                            marker_gap=args.overlay_gap,
                            max_fps=args.overlay_fps or 60.0,
                            show_labels=args.overlay_labels,
                            label_mode=args.overlay_label_mode,
                            damage=settings.effective_damage,
                        )
                    except (OverlayUnavailable, ValueError) as exc:
                        overlay_failed = True
                        print(f"[WARNING] Desktop overlay disabled: {exc}")

                if overlay is not None:
                    with timings.measure("overlay"):
                        overlay.render(candidates, captured.image.shape)
                    if overlay.closed:
                        overlay_failed = True
                        overlay = None
                        print(
                            "[WARNING] Desktop overlay closed unexpectedly; "
                            "markers are now only drawn in the preview window."
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
                    with timings.measure("preview"):
                        cv2.imshow(WINDOW_TITLE, visualized)

                timings.add("loop", time.perf_counter() - loop_started)
                limiter.sleep()
        finally:
            if worker is not None:
                worker.stop()
            if overlay is not None:
                overlay.close()
            if reporter is not None:
                reporter.close()
            if window_created:
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
