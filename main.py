"""Read-only last-hit companion.

Captures desktop frames, detects minions and their health bars, filters down to
only the minions that can currently be last-hit, orders them by priority
(nearest to the player anchor first), and visualizes only those candidates.
No game process memory access or mouse/keyboard input injection is performed.
"""

import argparse
import cv2

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
        default=2234 // 2,
        help="Target frame height in pixels.",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=3456 // 2,
        help="Target frame width in pixels.",
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
        default=0.02,
        help="Safety margin added to measured health fraction (0 <= margin < 1).",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.6,
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
) -> tuple[list[LastHitCandidate], object]:
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

    detector = Detector(weight_path=args.weights, device=args.device)
    object_associator = ObjectAssociator(min_confidence=args.min_confidence)
    advisor = LastHitAdvisor(settings)
    annotator = AIAnnotator() if args.visualize else None

    window_created = False
    with FrameLoader(monitor_idx=args.monitor_idx, width=args.W, height=args.H) as frame_loader:
        try:
            while True:
                if args.visualize and window_created and (cv2.waitKey(1) & 0xFF == ord("q")):
                    break

                window_rect = _get_preview_window_rect(WINDOW_TITLE) if window_created else None
                captured = frame_loader.grab(window_rect=window_rect)
                _, visualized = process_frame(
                    captured_frame=captured,
                    detector=detector,
                    object_associator=object_associator,
                    advisor=advisor,
                    annotator=annotator,
                    minion_color=args.minion_color,
                )

                if args.visualize and visualized is not None:
                    cv2.imshow(WINDOW_TITLE, visualized)
                    window_created = True
        finally:
            if window_created:
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
