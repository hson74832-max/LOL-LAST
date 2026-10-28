import ast
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    _AsyncDetectionWorker,
    build_arg_parser,
    evaluate_with_latest_detections,
    process_frame,
)
from my_utils.data_structures.labels import normalize_label
from my_utils.data_structures.minion import Minion
from my_utils.loaders.frame_loader import (
    CapturedFrame,
    mask_window,
    resize_excluded_regions,
    resize_image,
)
from my_utils.modules.algorithms import ObjectAssociator
from my_utils.modules.detector import (
    Detector,
    _is_valid_checkpoint_file,
    _torch_load_compat,
    _unwrap_predictions,
    resolve_weight_path,
)
from my_utils.modules.last_hitter import LastHitAdvisor, LastHitCandidate, LastHitSettings
from my_utils.modules.utils.health_bar_utils import HealthBarUtils
from my_utils.visualizers.annotator import (
    AIAnnotator,
    PRIMARY_TARGET_COLOR,
    SECONDARY_TARGET_COLOR,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_synthetic_health_bar(width: int = 60, height: int = 8, fill_px: int = 12, color: str = "red") -> np.ndarray:
    """Construct a synthetic health bar with the exact template transition at fill_px."""
    utils = HealthBarUtils()
    template = utils.templates[color]
    th, tw = template.shape[:2]
    bar = np.zeros((height, width, 3), dtype=np.uint8)

    left_color = template[:, :2].mean(axis=(0, 1)).astype(np.uint8)
    right_color = template[:, 2:].mean(axis=(0, 1)).astype(np.uint8)
    bar[:, :fill_px] = left_color
    bar[:, fill_px:] = right_color

    y0 = (height - th) // 2
    x0 = max(0, min(width - tw, fill_px - tw // 2))
    bar[y0 : y0 + th, x0 : x0 + tw] = template
    return bar


def test_no_injection_dependencies_or_calls():
    """Verify that input/process injection libraries are completely absent from the codebase."""
    forbidden_modules = {"pynput", "pyautogui", "keyboard", "mouse", "ctypes", "win32api", "pymem"}
    python_files = [
        p
        for p in REPO_ROOT.rglob("*.py")
        if ".venv" not in p.parts and "tests" not in p.parts and "yolov9" not in p.parts
    ]
    assert python_files

    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in forbidden_modules, f"Forbidden import {root_mod} in {path}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod not in forbidden_modules, f"Forbidden import {root_mod} in {path}"

    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "pynput" not in requirements


def test_only_last_hittable_minions_are_returned_and_prioritized():
    settings = LastHitSettings(
        effective_damage=80.0,
        max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0},
        health_margin=0.02,
        min_confidence=0.6,
        player_anchor=(0.5, 0.5),
    )
    advisor = LastHitAdvisor(settings)
    frame_shape = (1000, 1000, 3)

    far_low_caster = Minion("red caster minion", 0.15, (100, 100, 160, 160), 0.92)
    near_low_caster = Minion("red caster minion", 0.20, (460, 460, 520, 520), 0.90)
    high_hp_caster = Minion("red caster minion", 0.50, (490, 490, 530, 530), 0.95)
    low_fraction_but_high_hp_siege = Minion("red siege minion", 0.10, (480, 480, 540, 540), 0.95)
    zero_health_artifact = Minion("red caster minion", 0.0, (495, 495, 515, 515), 0.95)

    ranked = advisor.rank(
        [
            far_low_caster,
            near_low_caster,
            high_hp_caster,
            low_fraction_but_high_hp_siege,
            zero_health_artifact,
        ],
        frame_shape,
    )

    assert [c.minion for c in ranked] == [near_low_caster, far_low_caster]
    assert ranked[0].distance < ranked[1].distance
    assert all(c.health_upper_bound <= settings.effective_damage for c in ranked)


def test_priority_tie_breaker_is_deterministic():
    settings = LastHitSettings(
        effective_damage=100.0,
        max_health={"caster": 300.0},
        health_margin=0.0,
        min_confidence=0.6,
        player_anchor=(0.5, 0.5),
    )
    advisor = LastHitAdvisor(settings)
    # Equidistant from (500, 500): centers at (400, 500) and (600, 500)
    higher_hp = Minion("red caster minion", 0.25, (380, 480, 420, 520), 0.9)
    lower_hp = Minion("red caster minion", 0.10, (580, 480, 620, 520), 0.9)

    ranked = advisor.rank([higher_hp, lower_hp], (1000, 1000, 3))
    assert [c.minion for c in ranked] == [lower_hp, higher_hp]


def test_annotator_only_shows_last_hittable_candidates():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    annotator = AIAnnotator()

    # Empty candidate list -> completely untouched image (no object count text or stray boxes)
    empty_render = annotator(candidates=[], img=frame)
    assert np.array_equal(empty_render, frame)

    # Raw Minion passed directly must be rejected
    raw_minion = Minion("red caster minion", 0.1, (50, 50, 100, 100), 0.9)
    with pytest.raises(TypeError):
        annotator(candidates=[raw_minion], img=frame)  # type: ignore[list-item]

    primary = LastHitCandidate(
        minion=Minion("red caster minion", 0.15, (180, 180, 230, 230), 0.92),
        health_upper_bound=51.0,
        distance=10.0,
    )
    secondary = LastHitCandidate(
        minion=Minion("red melee minion", 0.10, (40, 40, 90, 90), 0.88),
        health_upper_bound=60.0,
        distance=180.0,
    )
    rendered = annotator(candidates=[primary, secondary], img=frame)

    # Original frame is not mutated
    assert not np.shares_memory(rendered, frame)
    assert np.all(frame == 0)

    # Primary (#1) box uses PRIMARY_TARGET_COLOR and secondary (#2) uses SECONDARY_TARGET_COLOR
    assert tuple(rendered[180, 200]) == PRIMARY_TARGET_COLOR
    assert tuple(rendered[40, 60]) == SECONDARY_TARGET_COLOR
    # Distant regions where no last-hittable minion or label exists remain untouched
    assert np.all(rendered[0:15, 300:400] == 0)
    assert np.all(rendered[300:400, 300:400] == 0)


def test_last_hit_settings_loophole_validation():
    source_hp = {"caster": 300.0, "melee": 500.0}
    settings = LastHitSettings(effective_damage=75.0, max_health=source_hp)
    # Mutating caller's dict or attempting to mutate settings.max_health cannot alter settings
    source_hp["caster"] = 10.0
    assert settings.max_health["caster"] == 300.0
    with pytest.raises(TypeError):
        settings.max_health["caster"] = 10.0  # type: ignore[index]

    # Booleans, NaN, Inf, non-positive values, and unknown minion types are rejected
    with pytest.raises(ValueError):
        LastHitSettings(effective_damage=True, max_health={"caster": 300.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LastHitSettings(effective_damage=float("nan"), max_health={"caster": 300.0})
    with pytest.raises(ValueError):
        LastHitSettings(effective_damage=80.0, max_health={"super": 1500.0})
    with pytest.raises(ValueError):
        LastHitSettings(effective_damage=80.0, max_health={"caster": 300.0}, player_anchor=(1.5, 0.5))


def test_health_bar_utils_rejects_uniform_or_wrong_color_bars_and_uses_rightmost_edge():
    utils = HealthBarUtils()

    # Uniform dark or bright bars have no valid transition and must return None (not 0.0 or 1.0)
    uniform_bar = np.full((8, 60, 3), 40, dtype=np.uint8)
    assert utils.get_health_level(uniform_bar, bar_color="red") is None
    assert utils.get_health_level(uniform_bar, bar_color="blue") is None

    # Synthetic red bar with transition at ~12px out of 60px
    red_bar = make_synthetic_health_bar(width=60, height=8, fill_px=12, color="red")
    level = utils.get_health_level(red_bar, bar_color="red")
    assert level is not None
    assert 0.15 < level < 0.35

    # Using a duplicate transition later in the bar must report the rightmost (conservative upper bound)
    double_edge_bar = red_bar.copy()
    template = utils.templates["red"]
    th, tw = template.shape[:2]
    y0 = (double_edge_bar.shape[0] - th) // 2
    double_edge_bar[y0 : y0 + th, 36 : 36 + tw] = template
    later_level = utils.get_health_level(double_edge_bar, bar_color="red")
    assert later_level is not None
    assert later_level > level


def test_object_associator_filters_clipped_ambiguous_and_unreadable_detections():
    associator = ObjectAssociator(intersection_threshold=0.5, min_confidence=0.6)
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Place a valid red health bar inside (100, 90, 160, 98)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=10, color="red")
    # Place another valid red health bar at the left screen edge x1=0 (clipped -> must be rejected)
    img[50:58, 0:60] = make_synthetic_health_bar(width=60, height=8, fill_px=10, color="red")

    detection_result = {
        "red_minions": [
            ((95, 85, 170, 160), "red caster minion", 0.91),
            # Clipped at screen border x1=0
            ((0, 45, 75, 120), "red caster minion", 0.95),
        ],
        "red_health_bars": [
            ((100, 90, 160, 98), 0.93),
            ((0, 50, 60, 58), 0.95),
        ],
        "blue_minions": [],
        "blue_health_bars": [],
    }

    matched = associator(detection_result, img=img, minion_type="red")
    assert len(matched) == 1
    assert matched[0].minion_type == "red caster minion"
    assert matched[0].bbox == (95.0, 85.0, 170.0, 160.0)

    # If the valid minion overlaps an excluded region (e.g. masked preview window), it must be ignored
    matched_excluded = associator(
        detection_result,
        img=img,
        minion_type="red",
        excluded_regions=((90, 80, 180, 170),),
    )
    assert matched_excluded == []


def test_object_associator_rejects_conflicting_duplicate_classes_and_ambiguous_assignments():
    associator = ObjectAssociator(intersection_threshold=0.5, min_confidence=0.6)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=10, color="red")

    # Conflicting duplicate boxes on the same body (caster vs siege) -> discarded rather than guessing
    conflicting = {
        "red_minions": [
            ((95, 85, 170, 160), "red caster minion", 0.89),
            ((96, 86, 170, 160), "red siege minion", 0.88),
        ],
        "red_health_bars": [((100, 90, 160, 98), 0.93)],
    }
    assert associator(conflicting, img=img, minion_type="red") == []

    # Two distinct minion bodies fully covering the same health bar -> ambiguous assignment rejected
    ambiguous = {
        "red_minions": [
            ((80, 85, 165, 160), "red caster minion", 0.90),
            ((95, 85, 190, 160), "red caster minion", 0.90),
        ],
        "red_health_bars": [((100, 90, 160, 98), 0.93)],
    }
    assert associator(ambiguous, img=img, minion_type="red") == []


def test_end_to_end_process_frame_shows_only_prioritized_last_hittable_minions():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Minion 1 (near center, low HP -> eligible, rank #1)
    img[180:188, 170:230] = make_synthetic_health_bar(width=60, height=8, fill_px=8, color="red")
    # Minion 2 (farther from center, low HP -> eligible, rank #2)
    img[40:48, 40:100] = make_synthetic_health_bar(width=60, height=8, fill_px=8, color="red")
    # Minion 3 (near center, high HP -> NOT last-hittable, must be excluded)
    img[260:268, 170:230] = make_synthetic_health_bar(width=60, height=8, fill_px=45, color="red")

    class DummyDetector:
        def __call__(self, _frame):
            return {
                "red_minions": [
                    ((165, 175, 235, 240), "red caster minion", 0.95),
                    ((35, 35, 105, 100), "red caster minion", 0.92),
                    ((165, 255, 235, 320), "red caster minion", 0.96),
                ],
                "red_health_bars": [
                    ((170, 180, 230, 188), 0.95),
                    ((40, 40, 100, 48), 0.92),
                    ((170, 260, 230, 268), 0.96),
                ],
                "blue_minions": [],
                "blue_health_bars": [],
                "cur_object_counts": {"red caster minion": 3},
            }

    settings = LastHitSettings(
        effective_damage=85.0,
        max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0},
    )
    candidates, visualized = process_frame(
        captured_frame=CapturedFrame(image=img, captured_at=1.0),
        detector=DummyDetector(),  # type: ignore[arg-type]
        object_associator=ObjectAssociator(),
        advisor=LastHitAdvisor(settings),
        annotator=AIAnnotator(),
        minion_color="red",
    )

    assert len(candidates) == 2
    assert candidates[0].minion.bbox == (165.0, 175.0, 235.0, 240.0)
    assert candidates[1].minion.bbox == (35.0, 35.0, 105.0, 100.0)
    # High-HP minion at y=255..320 must not have a bounding box drawn on its bottom border y=320
    assert np.all(visualized[320, 165:235] == 0)


def test_detector_classification_and_cli_parsing():
    class DummyBackend:
        names = {
            0: "red_caster_minion",
            1: "red_health_bar",
            2: "unknown_object",
        }
        stride = 32
        pt = True

    detector = Detector(backend=DummyBackend())
    pred = [
        np.array(
            [
                [10.0, 20.0, 50.0, 60.0, 0.9, 0.0],
                [12.0, 22.0, 48.0, 28.0, 0.85, 1.0],
                [70.0, 80.0, 90.0, 100.0, 0.99, 2.0],
            ]
        )
    ]
    red_m, blue_m, red_hb, blue_hb, counts = detector._classify_detections(
        pred, (120, 120), (120, 120, 3)
    )
    assert len(red_m) == 1 and red_m[0][1] == "red caster minion"
    assert len(red_hb) == 1
    assert blue_m == [] and blue_hb == []
    assert counts["red caster minion"] == 1
    assert "unknown object" not in counts

    # Dual-head prediction unwrapping
    assert _unwrap_predictions((["aux", "main"], "proto")) == "main"
    assert normalize_label("Red-Caster_Minion") == "red caster minion"

    # CLI parser accepts --visualize True and --visualize False without error
    parser = build_arg_parser()
    assert parser.parse_args(["--visualize", "True"]).visualize is True
    assert parser.parse_args(["--visualize", "False"]).visualize is False
    assert parser.parse_args(["--no-visualize"]).visualize is False


def test_window_masking_and_resizing():
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    monitor = {"left": 100, "top": 50, "width": 300, "height": 200}
    masked, regions = mask_window(img, monitor, (150, 80, 60, 40), margin=10)
    assert len(regions) == 1
    x1, y1, x2, y2 = regions[0]
    assert np.all(masked[y1:y2, x1:x2] == 0)

    resized = resize_image(masked, width=150, height=100)
    scaled_regions = resize_excluded_regions(regions, masked.shape, resized.shape)
    assert resized.shape == (100, 150, 3)
    assert len(scaled_regions) == 1


def test_installer_scripts_and_requirements_file():
    from packaging.requirements import Requirement

    req_lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert req_lines
    for line in req_lines:
        # Must be valid PEP 508 requirement specifiers (no broken git+ yolov9 URL without setup.py)
        req = Requirement(line)
        assert req.name.lower() != "pynput"
        assert "yolov9" not in line.lower()

    install_bat = (REPO_ROOT / "install_requirements.bat").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in install_bat
    assert "https://github.com/WongKinYiu/yolov9.git" in install_bat

    run_bat = (REPO_ROOT / "run.bat").read_text(encoding="utf-8")
    assert "install_requirements.bat" in run_bat
    assert "main.py" in run_bat

    install_sh = (REPO_ROOT / "install_requirements.sh").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in install_sh
    assert "https://github.com/WongKinYiu/yolov9.git" in install_sh
    assert "ensure_default_weights" in install_bat
    assert "ensure_default_weights" in install_sh


def test_weight_resolution_and_windows_torch_compat(tmp_path, monkeypatch):
    import my_utils.modules.detector as detector_mod

    # 1. Valid vs invalid checkpoint header checks
    html_file = tmp_path / "error.pt"
    html_file.write_bytes(b"<!DOCTYPE html><html>Error</html>" + b"0" * 2048)
    assert not _is_valid_checkpoint_file(html_file)

    valid_file = tmp_path / "best.pt"
    valid_file.write_bytes(b"PK\x03\x04" + b"\x00" * 4096)
    assert _is_valid_checkpoint_file(valid_file)

    # 2. Fallback discovery when best.pt is placed in project root or double-extension best.pt.pt
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    fake_default = fake_root / "weights" / "best.pt"
    monkeypatch.setattr(detector_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(detector_mod, "DEFAULT_WEIGHTS_PATH", fake_default)

    root_weight = fake_root / "best.pt.pt"
    root_weight.write_bytes(b"PK\x03\x04" + b"\x00" * 4096)
    resolved = resolve_weight_path("weights/best.pt", auto_download=False)
    assert resolved == fake_default
    assert fake_default.is_file()

    # 3. _torch_load_compat sets weights_only=False by default
    recorded_kwargs = {}

    class DummyTorch:
        @staticmethod
        def load(*_args, **kwargs):
            recorded_kwargs.update(kwargs)
            return {"model": None}

    with _torch_load_compat(DummyTorch):
        DummyTorch.load("dummy.pt", map_location="cpu")
    assert recorded_kwargs.get("weights_only") is False

    # 4. Single-dash -effective_damage 77 works in CLI
    parser = build_arg_parser()
    parsed = parser.parse_args(["-effective_damage", "77", "--H", "1440", "--W", "2560"])
    assert parsed.effective_damage == 77.0
    assert parsed.mask_preview is False
    assert parsed.H == 1440 and parsed.W == 2560

    # 5. High-resolution (2x upscaled for 2560x1440) health bar still matches via multi-scale fallback
    utils = HealthBarUtils()
    base_bar = make_synthetic_health_bar(width=60, height=8, fill_px=12, color="red")
    upscaled_bar = cv2.resize(base_bar, (120, 16), interpolation=cv2.INTER_LINEAR)
    upscaled_level = utils.get_health_level(upscaled_bar, bar_color="red")
    assert upscaled_level is not None
    assert 0.15 < upscaled_level < 0.35


def test_real_world_low_hp_minions_in_wave_with_low_bar_confidence_and_overlap():
    """Verify low-HP minions (including melee/siege and low-confidence bars in overlapping waves) are detected."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Realistic bordered & slightly blurred red health bar at ~12% HP (melee minion: 0.12 * 500 = 60 HP <= 77)
    def make_realistic_red_bar(w: int, h: int, fill: int) -> np.ndarray:
        bar = np.zeros((h, w, 3), dtype=np.uint8)
        bar[:, :fill] = [55, 55, 185]
        bar[:, fill:] = [28, 22, 32]
        bar[0, :] = [15, 15, 15]
        bar[-1, :] = [15, 15, 15]
        bar[:, 0] = [15, 15, 15]
        bar[:, -1] = [15, 15, 15]
        return cv2.GaussianBlur(bar, (3, 1), 0.6)

    # Minion A (melee, low HP: 6/50 = 12% -> 60 HP <= 77) at top
    img[90:98, 100:150] = make_realistic_red_bar(50, 8, 6)
    # Minion B (caster, low HP: 10/50 = 20% -> 60 HP <= 77) slightly lower, overlapped by Minion A's body box!
    img[135:143, 100:150] = make_realistic_red_bar(50, 8, 10)

    class WaveDetector:
        def __call__(self, _frame):
            return {
                "red_minions": [
                    # Minion A's body box extends down to y=165, overlapping BOTH Bar A (y=90..98) and Bar B (y=135..143)
                    ((95, 85, 160, 165), "red melee minion", 0.84),
                    ((95, 130, 160, 205), "red caster minion", 0.81),
                ],
                "red_health_bars": [
                    # Low-HP bars often have confidence 0.30..0.55 in YOLOv9 (as seen in figures/futuristic.png)
                    ((100, 90, 150, 98), 0.37),
                    ((100, 135, 150, 143), 0.43),
                ],
                "blue_minions": [],
                "blue_health_bars": [],
                "cur_object_counts": {},
            }

    settings = LastHitSettings(
        effective_damage=77.0,
        max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0},
    )
    candidates, visualized = process_frame(
        captured_frame=CapturedFrame(image=img, captured_at=1.0),
        detector=WaveDetector(),  # type: ignore[arg-type]
        object_associator=ObjectAssociator(),
        advisor=LastHitAdvisor(settings),
        annotator=AIAnnotator(),
        minion_color="red",
    )

    assert len(candidates) == 2
    assert visualized is not None
    # Closest to center (200, 200) is Minion B at y=130..205, then Minion A at y=85..165
    assert candidates[0].minion.minion_type == "red caster minion"
    assert candidates[1].minion.minion_type == "red melee minion"


def test_async_detection_worker_and_live_frame_health_reevaluation():
    """Verify the async worker drops stale frames and live frames immediately update last-hit status."""
    import time

    # Frame 1: minion at 50% HP (not last-hittable)
    frame_high_hp = np.zeros((300, 400, 3), dtype=np.uint8)
    frame_high_hp[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=30, color="red")

    # Frame 2 (live frame): same minion position, but health bar just dropped to ~13% HP (last-hittable!)
    frame_low_hp = np.zeros((300, 400, 3), dtype=np.uint8)
    frame_low_hp[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=8, color="red")

    class BoxDetector:
        def __call__(self, _frame):
            return {
                "red_minions": [((95, 85, 170, 160), "red caster minion", 0.91)],
                "red_health_bars": [((100, 90, 160, 98), 0.89)],
                "blue_minions": [],
                "blue_health_bars": [],
                "cur_object_counts": {},
            }

    worker = _AsyncDetectionWorker(BoxDetector())  # type: ignore[arg-type]
    worker.start()
    try:
        cap1 = CapturedFrame(image=frame_high_hp, captured_at=1.0)
        worker.submit(cap1)
        for _ in range(50):
            if worker.get_latest() is not None:
                break
            time.sleep(0.01)
        assert worker.get_latest() is not None

        settings = LastHitSettings(
            effective_damage=77.0,
            max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0},
        )
        associator = ObjectAssociator()
        advisor = LastHitAdvisor(settings)
        annotator = AIAnnotator()

        # On cap1 (50% HP = 150 HP > 77), minion is NOT last-hittable
        candidates_1, _ = evaluate_with_latest_detections(
            captured_frame=cap1,
            latest_detection=worker.get_latest(),
            object_associator=associator,
            advisor=advisor,
            annotator=annotator,
            minion_color="red",
        )
        assert candidates_1 == []

        # On cap2 (13% HP = 40 HP <= 77), using the already-available detection boxes on the live frame
        # immediately identifies the minion as last-hittable without waiting for another YOLO pass!
        cap2 = CapturedFrame(image=frame_low_hp.copy(), captured_at=1.05)
        candidates_2, _ = evaluate_with_latest_detections(
            captured_frame=cap2,
            latest_detection=worker.get_latest(),
            object_associator=associator,
            advisor=advisor,
            annotator=annotator,
            minion_color="red",
        )
        assert len(candidates_2) == 1
        assert candidates_2[0].health_upper_bound <= 77.0
    finally:
        worker.stop()


def test_overlay_is_lazily_loaded_and_stays_headless_safe():
    """The overlay must import tkinter lazily so headless/CI imports never create a window."""
    overlay_path = REPO_ROOT / "my_utils" / "visualizers" / "overlay.py"
    tree = ast.parse(overlay_path.read_text(encoding="utf-8"), filename=str(overlay_path))
    top_level_imports = set()
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "tkinter" not in top_level_imports

    from my_utils.visualizers.overlay import DesktopOverlay, OverlayUnavailable

    # Monitor validation happens before any Tk object is created.
    with pytest.raises(ValueError):
        DesktopOverlay({"left": 0, "top": 0, "width": 0, "height": 1080})
    with pytest.raises(ValueError):
        DesktopOverlay({"left": 0, "top": 0})

    try:
        import tkinter  # noqa: F401

        has_tk = True
    except Exception:
        has_tk = False
    if not has_tk:
        with pytest.raises(OverlayUnavailable):
            DesktopOverlay({"left": 0, "top": 0, "width": 1920, "height": 1080})


def test_overlay_marker_geometry_and_monitor_mapping():
    from my_utils.visualizers.overlay import (
        DEFAULT_MARKER_GAP,
        PRIMARY_TARGET_COLOR,
        SECONDARY_TARGET_COLOR,
        DesktopOverlay,
        display_scale,
        marker_polygon,
    )

    # Markers float entirely above the bounding box, centred on it, above the health bar.
    points = marker_polygon((100, 200, 140, 260), gap=DEFAULT_MARKER_GAP)
    assert points[0][0] == 120.0
    assert points[0][1] == 200.0 - DEFAULT_MARKER_GAP
    assert all(y < 200.0 for _, y in points)
    assert len(points) == 3

    with pytest.raises(ValueError):
        marker_polygon((1, 2, 3))
    with pytest.raises(ValueError):
        marker_polygon((100, 200, 140, 260), gap=-5)
    with pytest.raises(ValueError):
        marker_polygon((100, float("nan"), 140, 260))

    # DPI virtualisation ratio between Tk logical units and mss physical pixels.
    assert display_scale(1920, 1920) == 1.0
    assert abs(display_scale(2400, 1920) - 1.25) < 1e-9
    assert display_scale(0, 1920) == 1.0
    assert display_scale(None, 1920) == 1.0

    class _FakeCanvas:
        def __init__(self):
            self.polygons = []
            self.texts = []

        def delete(self, *_args):
            self.polygons.clear()
            self.texts.clear()

        def create_polygon(self, *coords, **kwargs):
            self.polygons.append((coords, kwargs))

        def create_text(self, *coords, **kwargs):
            self.texts.append((coords, kwargs))

    class _FakeRoot:
        def update(self):
            pass

    overlay = DesktopOverlay.__new__(DesktopOverlay)
    overlay.monitor_left = 0
    overlay.monitor_top = 0
    overlay.monitor_width = 1920
    overlay.monitor_height = 1080
    overlay.marker_gap = 16.0
    overlay.show_labels = True
    overlay.label_font = ("Segoe UI", 9, "bold")
    overlay.label_offset = 6.0
    overlay._display_scale = 1.0
    overlay._max_interval = 0.0
    overlay._last_render = 0.0
    overlay._closed = False
    overlay._tkinter = None
    overlay._root = _FakeRoot()
    overlay._canvas = _FakeCanvas()

    primary = LastHitCandidate(
        minion=Minion("red caster minion", 0.10, (960, 500, 1000, 560), 0.9),
        health_upper_bound=30.0,
        distance=5.0,
    )
    secondary = LastHitCandidate(
        minion=Minion("red melee minion", 0.10, (480, 500, 520, 560), 0.9),
        health_upper_bound=50.0,
        distance=90.0,
    )
    overlay.render([primary, secondary], (1080, 1920, 3))

    assert len(overlay._canvas.polygons) == 2
    primary_coords, primary_kwargs = overlay._canvas.polygons[0]
    secondary_coords, secondary_kwargs = overlay._canvas.polygons[1]
    assert primary_kwargs["fill"] == PRIMARY_TARGET_COLOR
    assert secondary_kwargs["fill"] == SECONDARY_TARGET_COLOR
    # Frame pixels map 1:1 onto the monitor here, so the marker tip sits 16px above the box.
    assert primary_coords[0] == 980.0
    assert primary_coords[1] == 500.0 - 16.0
    assert secondary_coords[0] == 500.0

    # Closing is idempotent and silences further drawing.
    overlay.close()
    assert overlay.closed is True
    overlay.render([primary], (1080, 1920, 3))
    assert len(overlay._canvas.polygons) == 2


def test_overlay_cli_flags_and_defaults():
    parser = build_arg_parser()
    default_args = parser.parse_args([])
    assert default_args.overlay is True
    assert default_args.overlay_gap > 0

    assert parser.parse_args(["--overlay", "False"]).overlay is False
    assert parser.parse_args(["--no-overlay"]).overlay is False
    assert parser.parse_args(["--overlay"]).overlay is True
    assert parser.parse_args(["--overlay_gap", "24"]).overlay_gap == 24.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--overlay_gap", "-3"])


def test_main_loop_draws_candidates_on_the_desktop_overlay(monkeypatch, capsys):
    """The capture loop must feed last-hit candidates to the desktop overlay (not only the preview)."""
    import main as main_module

    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=8, color="red")

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, _frame):
            return {
                "red_minions": [((95, 85, 170, 160), "red caster minion", 0.91)],
                "red_health_bars": [((100, 90, 160, 98), 0.89)],
                "blue_minions": [],
                "blue_health_bars": [],
                "cur_object_counts": {},
            }

    remaining = [CapturedFrame(image=img.copy(), captured_at=1.0) for _ in range(3)]

    class FakeFrameLoader:
        monitor = {"left": 0, "top": 0, "width": 400, "height": 300}
        primary_monitor = {"left": 0, "top": 0, "width": 400, "height": 300}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def grab(self, window_rect=None):
            if not remaining:
                raise KeyboardInterrupt
            return remaining.pop(0)

    events = []

    class FakeOverlay:
        def __init__(self, monitor, primary_width=None, marker_gap=16.0, max_fps=30.0,
                     show_labels=True, label_mode="hp", damage=None):
            self.closed = False
            events.append(("init", monitor["width"], primary_width, marker_gap))

        def render(self, candidates, frame_shape):
            events.append(("render", len(candidates), tuple(frame_shape[:2])))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(main_module, "Detector", FakeDetector)
    monkeypatch.setattr(main_module, "FrameLoader", FakeFrameLoader)
    monkeypatch.setattr(main_module, "DesktopOverlay", FakeOverlay)

    with pytest.raises(KeyboardInterrupt):
        main_module.main(["--sync", "--no-visualize", "--overlay_gap", "20"])

    assert events[0] == ("init", 400, 400, 20.0)
    assert ("render", 1, (300, 400)) in events
    assert events[-1] == ("close",)

    # --no-overlay keeps the desktop clean.
    events.clear()
    remaining.extend(CapturedFrame(image=img.copy(), captured_at=1.0) for _ in range(3))
    with pytest.raises(KeyboardInterrupt):
        main_module.main(["--sync", "--no-visualize", "--no-overlay"])
    assert events == []


def test_device_selection_covers_cuda_mps_directml_and_cpu(monkeypatch):
    """AMD/Intel GPUs must be usable through torch-directml instead of silently using CPU."""
    import types

    from my_utils.modules import detector as detector_module
    from my_utils.modules.detector import (
        directml_available,
        is_directml_request,
        select_default_device,
    )

    def fake_torch(cuda=False, mps=False):
        backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda), backends=backends
        )

    monkeypatch.setattr(detector_module, "directml_available", lambda: False)
    assert select_default_device(fake_torch(cuda=True)) == "cuda:0"
    assert select_default_device(fake_torch(mps=True)) == "mps"
    assert select_default_device(fake_torch()) == "cpu"

    # With torch-directml present on an AMD/Intel machine, DirectML wins over CPU.
    monkeypatch.setattr(detector_module, "directml_available", lambda: True)
    assert select_default_device(fake_torch()) == "directml"
    assert select_default_device(fake_torch(cuda=True)) == "cuda:0"

    for alias in ("directml", "DirectML", "dml", "privateuseone:0", " privateuseone "):
        assert is_directml_request(alias) is True
    for value in ("cpu", "cuda:0", "mps", None):
        assert is_directml_request(value) is False

    # Not installed -> availability is False and the error message points at the fix.
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    assert directml_available() is False
    with pytest.raises(RuntimeError) as excinfo:
        detector_module.directml_device()
    assert "torch-directml" in str(excinfo.value)

    # A working fake backend is returned as-is.
    fake_dml = types.SimpleNamespace(
        is_available=lambda: True, device=lambda: "dml-device"
    )
    monkeypatch.setitem(sys.modules, "torch_directml", fake_dml)
    assert directml_available() is True
    assert detector_module.directml_device() == "dml-device"


def test_candidate_gate_requires_consecutive_sightings_and_drops_dead_minions():
    from my_utils.modules.stability import CandidateGate

    with pytest.raises(ValueError):
        CandidateGate(confirm_frames=0)
    with pytest.raises(ValueError):
        CandidateGate(confirm_frames=2, match_ratio=0)

    def candidate(x, y, kind="red caster minion", health=0.1):
        return LastHitCandidate(
            minion=Minion(kind, health, (x, y, x + 40, y + 60), 0.9),
            health_upper_bound=30.0,
            distance=1.0,
        )

    gate = CandidateGate(confirm_frames=2)
    first = candidate(100, 100)
    second = candidate(400, 100, kind="red melee minion")

    # The live frame is re-evaluated many times per detector result: one result can
    # never confirm a marker, no matter how many times it is re-read.
    gen = object()
    assert gate.update([first, second], (300, 500, 3), generation=gen) == []
    for _ in range(5):
        assert gate.update([first, second], (300, 500, 3), generation=gen) == []

    # A new detector result that still sees them confirms both markers.
    gen = object()
    confirmed = gate.update([first, second], (300, 500, 3), generation=gen)
    assert [c.minion.minion_type for c in confirmed] == ["red caster minion", "red melee minion"]

    # Small jitter between frames keeps the same track (no re-confirmation delay).
    jittered = candidate(104, 103)
    assert gate.update([jittered], (300, 500, 3), generation=object()) != []

    # Minion dies: it disappears from the very next evaluation, no hold, no memory.
    assert gate.update([], (300, 500, 3), generation=object()) == []
    assert gate.track_count == 0

    # After a gap the minion has to be seen twice again (no instant ghost marker).
    assert gate.update([first], (300, 500, 3), generation=object()) == []
    assert gate.update([first], (300, 500, 3), generation=object()) != []

    # Adjacent minions in a wave are separate tracks, never merged.
    gate = CandidateGate(confirm_frames=1)
    left = candidate(100, 100)
    right = candidate(175, 100)
    assert len(gate.update([left, right], (300, 500, 3), generation=object())) == 2
    assert gate.track_count == 2

    # confirm_frames=1 disables the confirmation delay entirely.
    gate = CandidateGate(confirm_frames=1)
    assert gate.update([first], (300, 500, 3), generation=object()) != []


def test_stale_detections_are_never_drawn_as_markers():
    """Boxes older than --max_detection_age must be discarded, not projected on the live frame."""
    from main import DEFAULT_MAX_DETECTION_AGE, detection_age

    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=8, color="red")

    detections = {
        "red_minions": [((95, 85, 170, 160), "red caster minion", 0.91)],
        "red_health_bars": [((100, 90, 160, 98), 0.89)],
        "blue_minions": [],
        "blue_health_bars": [],
        "cur_object_counts": {},
    }
    settings = LastHitSettings(
        effective_damage=77.0, max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0}
    )
    associator = ObjectAssociator()
    advisor = LastHitAdvisor(settings)

    det_frame = CapturedFrame(image=img.copy(), captured_at=1.0)
    live_frame = CapturedFrame(image=img.copy(), captured_at=1.05)
    assert detection_age(live_frame, (det_frame, detections)) == pytest.approx(0.05)
    assert detection_age(live_frame, None) is None

    # Fresh boxes: the minion is a valid last-hit candidate.
    candidates, _ = evaluate_with_latest_detections(
        captured_frame=live_frame,
        latest_detection=(det_frame, detections),
        object_associator=associator,
        advisor=advisor,
    )
    assert len(candidates) == 1

    # Same boxes, but now far too old (slow inference): nothing is drawn at all.
    stale_live = CapturedFrame(image=img.copy(), captured_at=1.0 + DEFAULT_MAX_DETECTION_AGE + 0.2)
    assert detection_age(stale_live, (det_frame, detections)) > DEFAULT_MAX_DETECTION_AGE
    candidates, _ = evaluate_with_latest_detections(
        captured_frame=stale_live,
        latest_detection=(det_frame, detections),
        object_associator=associator,
        advisor=advisor,
    )
    assert candidates == []

    # The legacy 0.6 s "re-run on the previous frame" fallback is opt-in only.
    empty_live = CapturedFrame(image=np.zeros((300, 400, 3), dtype=np.uint8), captured_at=1.2)
    off, _ = evaluate_with_latest_detections(
        captured_frame=empty_live,
        latest_detection=(det_frame, detections),
        object_associator=associator,
        advisor=advisor,
    )
    assert off == []
    on, _ = evaluate_with_latest_detections(
        captured_frame=empty_live,
        latest_detection=(det_frame, detections),
        object_associator=associator,
        advisor=advisor,
        allow_stale_frame_fallback=True,
    )
    assert len(on) == 1


def test_cli_exposes_strictness_knobs_and_bar_confidence_threshold():
    from main import DEFAULT_BAR_MIN_CONFIDENCE, DEFAULT_CONFIRM_FRAMES, DEFAULT_MAX_DETECTION_AGE

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.max_detection_age == DEFAULT_MAX_DETECTION_AGE
    assert args.confirm_frames == DEFAULT_CONFIRM_FRAMES
    assert args.min_bar_confidence == DEFAULT_BAR_MIN_CONFIDENCE
    assert args.stale_frame_fallback is False

    strict = parser.parse_args(
        ["--max_detection_age", "0.15", "--confirm_frames", "3", "--min_bar_confidence", "0.4"]
    )
    assert strict.max_detection_age == 0.15
    assert strict.confirm_frames == 3
    assert strict.min_bar_confidence == 0.4
    with pytest.raises(SystemExit):
        parser.parse_args(["--max_detection_age", "-1"])

    # Bar detections below the bar threshold are dropped (corpse / effect artifacts).
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=10, color="red")
    weak_bar = {
        "red_minions": [((95, 85, 170, 160), "red caster minion", 0.9)],
        "red_health_bars": [((100, 90, 160, 98), 0.30)],
        "blue_minions": [],
        "blue_health_bars": [],
    }
    assert len(ObjectAssociator(bar_min_confidence=0.20)(weak_bar, img=img, minion_type="red")) == 1
    # Raising the bar threshold rejects the YOLO bar detection itself ...
    assert (
        ObjectAssociator(bar_min_confidence=0.40, fallback_to_locator=False)(
            weak_bar, img=img, minion_type="red"
        )
        == []
    )
    # ... but the geometric locator can still read the bar from the pixels.
    fallback = ObjectAssociator(bar_min_confidence=0.40)(weak_bar, img=img, minion_type="red")
    assert len(fallback) == 1
    assert 0.1 < fallback[0].health_level < 0.3
    with pytest.raises(ValueError):
        ObjectAssociator(bar_min_confidence=0)
    with pytest.raises(ValueError):
        ObjectAssociator(bar_min_confidence=1.5)


def test_main_gates_markers_and_honours_confirm_frames(monkeypatch):
    """main() must route candidates through the confirmation gate and the bar threshold."""
    import main as main_module

    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=8, color="red")

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, _frame):
            return {
                "red_minions": [((95, 85, 170, 160), "red caster minion", 0.91)],
                "red_health_bars": [((100, 90, 160, 98), 0.89)],
                "blue_minions": [],
                "blue_health_bars": [],
                "cur_object_counts": {},
            }

    remaining = [CapturedFrame(image=img.copy(), captured_at=1.0 + i) for i in range(4)]

    class FakeFrameLoader:
        monitor = {"left": 0, "top": 0, "width": 400, "height": 300}
        primary_monitor = {"left": 0, "top": 0, "width": 400, "height": 300}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def grab(self, window_rect=None):
            if not remaining:
                raise KeyboardInterrupt
            return remaining.pop(0)

    rendered = []

    class FakeOverlay:
        def __init__(self, monitor, primary_width=None, marker_gap=16.0, max_fps=30.0,
                     show_labels=True, label_mode="hp", damage=None):
            self.closed = False
            assert label_mode in {"hp", "hp_damage", "percent"}
            assert damage is None or damage > 0

        def render(self, candidates, frame_shape):
            rendered.append(len(candidates))

        def close(self):
            pass

    monkeypatch.setattr(main_module, "Detector", FakeDetector)
    monkeypatch.setattr(main_module, "FrameLoader", FakeFrameLoader)
    monkeypatch.setattr(main_module, "DesktopOverlay", FakeOverlay)

    with pytest.raises(KeyboardInterrupt):
        main_module.main(["--sync", "--no-visualize", "--confirm_frames", "2"])
    # First inference only seeds the track; from the second one the marker is drawn.
    assert rendered == [0, 1, 1, 1]

    rendered.clear()
    remaining.extend(CapturedFrame(image=img.copy(), captured_at=5.0 + i) for i in range(3))
    with pytest.raises(KeyboardInterrupt):
        main_module.main(["--sync", "--no-visualize", "--confirm_frames", "1"])
    assert rendered == [1, 1, 1]


def test_directml_detector_builds_on_cpu_and_moves_model_to_dml(monkeypatch, tmp_path, capsys):
    """AMD/Intel path: YOLOv9 cannot build on 'privateuseone', so build on CPU then move."""
    import types as _types

    from my_utils.modules import detector as detector_module

    calls = []

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.ndim = self.array.ndim
            self.shape = self.array.shape

        def to(self, device, non_blocking=False):
            calls.append(("tensor.to", str(device)))
            return self

        def half(self):
            return self

        def float(self):
            return self

        def __itruediv__(self, value):
            self.array = self.array / value
            return self

        def __getitem__(self, item):
            return FakeTensor(self.array[item])

    class FakeNet:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = str(device)
            calls.append(("net.to", str(device)))
            return self

    class FakeModel:
        names = {0: "red_caster_minion", 1: "red_health_bar"}
        stride = 32
        pt = True
        fp16 = False

        def __init__(self):
            self.device = "cpu"
            self.model = FakeNet()

        def __call__(self, tensor, augment=False, visualize=False):
            calls.append(("forward", str(self.device)))

            class Pred:
                def cpu(self):
                    calls.append(("pred.cpu",))
                    return np.array([[10.0, 20.0, 50.0, 60.0, 0.9, 0.0]])

            return Pred()

    class FakeDevice:
        def __init__(self, name):
            self.name = str(name)
            self.type = self.name.split(":")[0]

        def __str__(self):
            return self.name

    fake_torch = _types.SimpleNamespace(
        device=FakeDevice,
        load=lambda *a, **k: {},
        no_grad=lambda: __import__("contextlib").nullcontext(),
        from_numpy=lambda array: FakeTensor(array),
        backends=_types.SimpleNamespace(cudnn=_types.SimpleNamespace(benchmark=False)),
    )
    fake_dml = _types.SimpleNamespace(
        is_available=lambda: True, device=lambda: "dml-device"
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_directml", fake_dml)
    monkeypatch.setattr(detector_module, "resolve_weight_path", lambda *a, **k: tmp_path / "best.pt")
    monkeypatch.setattr(
        detector_module,
        "_load_yolov9_symbols",
        lambda: (
            lambda *a, **k: FakeModel(),
            lambda img, size, stride=32, auto=True: (img, None, None),
            lambda pred, **kwargs: [pred],
            lambda *a, **k: a[1],
        ),
    )

    detector = Detector(weight_path=tmp_path / "best.pt", device="directml", half=True)
    assert detector.directml is True
    assert detector.device_name == "dml-device"
    assert ("net.to", "dml-device") in calls
    out = capsys.readouterr().out
    assert "DirectML" in out
    assert "fp16 is not supported by DirectML" in out

    # Inference: input goes to the DML device, predictions come back to CPU for NMS.
    result = detector(np.zeros((64, 64, 3), dtype=np.uint8))
    assert ("tensor.to", "dml-device") in calls
    assert ("forward", "dml-device") in calls
    assert ("pred.cpu",) in calls
    assert len(result["red_minions"]) == 1

    # A DirectML failure must degrade to CPU instead of crashing.
    calls.clear()

    class BrokenNet(FakeNet):
        def to(self, device):
            raise RuntimeError("dml init failed")

    class BrokenModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.model = BrokenNet()

    monkeypatch.setattr(
        detector_module,
        "_load_yolov9_symbols",
        lambda: (
            lambda *a, **k: BrokenModel(),
            lambda img, size, stride=32, auto=True: (img, None, None),
            lambda pred, **kwargs: [pred],
            lambda *a, **k: a[1],
        ),
    )
    detector = Detector(weight_path=tmp_path / "best.pt", device="dml")
    assert detector.directml is False
    assert detector.device_name == "cpu"
    assert "falling back to CPU" in capsys.readouterr().out


def test_directml_status_reports_actionable_reasons(monkeypatch):
    from my_utils.modules import detector as detector_module

    monkeypatch.setitem(sys.modules, "torch_directml", None)
    available, reason = detector_module.directml_status()
    assert available is False
    assert "not installed" in reason

    import types as _types

    monkeypatch.setitem(
        sys.modules,
        "torch_directml",
        _types.SimpleNamespace(is_available=lambda: False, device=lambda: "dml"),
    )
    available, reason = detector_module.directml_status()
    assert available is False and "no usable DirectML device" in reason

    monkeypatch.setitem(
        sys.modules,
        "torch_directml",
        _types.SimpleNamespace(is_available=lambda: True, device=lambda: "dml"),
    )
    available, reason = detector_module.directml_status()
    assert available is True and "available" in reason


def _fake_torch_for_device_tests():
    import types as _types

    class FakeDevice:
        def __init__(self, name):
            self.name = str(name)
            self.type = self.name.split(":")[0]

        def __str__(self):
            return self.name

    class FakeNet:
        def to(self, device):
            return self

    class FakeModel:
        names = {0: "red_caster_minion", 1: "red_health_bar"}
        stride = 32
        pt = True

        def __init__(self):
            self.device = "cpu"
            self.model = FakeNet()

    torch_module = _types.SimpleNamespace(
        device=FakeDevice,
        load=lambda *a, **k: {},
        backends=_types.SimpleNamespace(cudnn=_types.SimpleNamespace(benchmark=False)),
    )
    return torch_module, FakeModel


def _patch_detector_backend(monkeypatch, torch_module, FakeModel):
    from my_utils.modules import detector as detector_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(
        detector_module, "resolve_weight_path", lambda *a, **k: Path("best.pt")
    )
    monkeypatch.setattr(
        detector_module,
        "_load_yolov9_symbols",
        lambda: (
            lambda *a, **k: FakeModel(),
            lambda img, size, stride=32, auto=True: (img, None, None),
            lambda pred, **kwargs: [pred],
            lambda *a, **k: a[1],
        ),
    )
    return detector_module


def test_missing_directml_falls_back_to_cpu_instead_of_crashing(monkeypatch, capsys):
    """`--device directml` without torch-directml must warn and keep running on CPU."""
    from my_utils.modules.detector import Detector

    torch_module, FakeModel = _fake_torch_for_device_tests()
    _patch_detector_backend(monkeypatch, torch_module, FakeModel)
    monkeypatch.setitem(sys.modules, "torch_directml", None)

    detector = Detector(device="directml")

    assert detector.directml is False
    assert detector.device_name == "cpu"
    assert detector.device_fallback_reason
    out = capsys.readouterr().out
    assert "[WARNING] DirectML was requested but is unavailable" in out
    assert "torch-directml is not installed" in out
    # The hint must name the exact interpreter that is running the companion.
    assert ".venv\\Scripts\\python.exe -m pip install torch-directml" in out
    assert "separate packages" in out
    assert "CPU (DirectML fallback)" in out

    # --strict_device keeps the old hard failure for users who want it.
    with pytest.raises(RuntimeError) as excinfo:
        Detector(device="directml", strict_device=True)
    assert "torch-directml" in str(excinfo.value)


def test_check_devices_reports_backends_and_how_to_fix_them(monkeypatch, capsys):
    import types as _types

    from my_utils.modules import detector as detector_module

    parser = build_arg_parser()
    assert parser.parse_args(["--check-devices"]).check_devices is True
    assert parser.parse_args([]).check_devices is False
    assert parser.parse_args(["--strict_device"]).strict_device is True

    fake_torch = _types.SimpleNamespace(
        __version__="2.9.0+cpu",
        cuda=_types.SimpleNamespace(is_available=lambda: False),
        backends=_types.SimpleNamespace(mps=_types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    lines = detector_module.diagnose_devices(fake_torch)
    report = "\n".join(lines)
    assert "torch               : 2.9.0+cpu" in report
    assert "CUDA (NVIDIA)       : not available" in report
    assert "DirectML (AMD/Intel): not available" in report
    assert "not installed" in report
    assert "torch-directml" in report
    assert "Selected backend    : cpu" in report

    # Working DirectML is reported as available and becomes the selected backend.
    monkeypatch.setitem(
        sys.modules,
        "torch_directml",
        _types.SimpleNamespace(is_available=lambda: True, device=lambda: "dml"),
    )
    monkeypatch.setattr(detector_module, "directml_available", lambda: True)
    lines = detector_module.diagnose_devices(fake_torch)
    report = "\n".join(lines)
    assert "DirectML (AMD/Intel): available" in report
    assert "Selected backend    : directml" in report

    # The CLI shortcut prints the report and exits without loading any model.
    import main as main_module

    monkeypatch.setattr(
        main_module, "diagnose_devices", lambda: ["torch : test-version", "", "Selected backend    : cpu"]
    )
    assert main_module.main(["--check-devices"]) == 0
    out = capsys.readouterr().out
    assert "Inference backend diagnostics" in out
    assert "[INFO] torch : test-version" in out


def test_debug_stats_and_hints_explain_every_empty_pipeline_stage():
    from my_utils.modules.diagnostics import no_marker_hint, stats_from_frame

    img = np.full((300, 400, 3), 60, dtype=np.uint8)
    black = np.zeros((300, 400, 3), dtype=np.uint8)

    def hint(det, pipeline=None, image=img, age=0.05, infer=0.04):
        return no_marker_hint(
            stats_from_frame(image, det, "red", pipeline=pipeline,
                             detection_age_seconds=age, inference_seconds=infer)
        )

    # Nothing detected
    assert "no minions and no health bars" in hint({"red_minions": [], "red_health_bars": []})
    # Wrong minion colour
    assert "--minion_color blue" in hint({"red_minions": [], "blue_minions": [1, 2]})
    # Bodies without any readable bar (YOLO and locator both failed)
    unreadable = hint({"red_minions": [1, 2, 3], "red_health_bars": []})
    assert "--min_bar_confidence" in unreadable and "--imgsz 960" in unreadable
    assert "3 minion bodies detected" in unreadable
    # Bodies with a YOLO bar that still could not be read
    assert "1 YOLOv9 bar" in hint(
        {"red_minions": [1, 2], "red_health_bars": [1]}, {"matched": 0}
    )
    # Health filter rejects everything
    assert "--effective_damage" in hint(
        {"red_minions": [1], "red_health_bars": [1]},
        {"matched": 1, "eligible": 0, "health": [0.85]},
    )
    # Confirmation gate hides an eligible candidate
    assert "--confirm_frames" in hint(
        {"red_minions": [1], "red_health_bars": [1]},
        {"matched": 1, "eligible": 1, "kept": 0, "health": [0.1]},
    )
    # Stale boxes win over every other explanation (that is the actual cause)
    stale = hint(
        {"red_minions": [1], "red_health_bars": [1]},
        {"matched": 0, "eligible": 0, "kept": 0, "stale": True},
        age=0.62, infer=0.58,
    )
    assert "too old" in stale and "620ms" in stale and "--max_detection_age" in stale
    # Black capture
    assert "captured screen is black" in hint(
        {"red_minions": [1], "red_health_bars": [1]},
        {"matched": 1, "eligible": 1, "kept": 0},
        image=black,
    )
    # Working case -> no hint at all
    assert hint({"red_minions": [1], "red_health_bars": [1]},
                {"matched": 1, "eligible": 1, "kept": 1, "health": [0.1]}) is None


def test_debug_reporter_formats_prints_and_saves_annotated_frames(tmp_path, capsys):
    from my_utils.modules.diagnostics import DebugReporter, annotate_debug_frame, format_stats

    img = np.full((300, 400, 3), 60, dtype=np.uint8)
    detection_result = {
        "red_minions": [((100, 80, 140, 160), "red caster minion", 0.9)],
        "red_health_bars": [((105, 85, 140, 93), 0.9)],
    }
    reporter = DebugReporter(minion_color="red", frames_dir=tmp_path / "dbg", interval=0.0)
    stats = __import__("my_utils.modules.diagnostics", fromlist=["stats_from_frame"]).stats_from_frame(
        img, detection_result, "red",
        pipeline={"matched": 1, "eligible": 0, "kept": 0, "health": [0.55]},
        detection_age_seconds=0.11, inference_seconds=0.09,
    )
    line = format_stats(stats, "red")
    assert "bodies=1 bars=1" in line and "matched=1" in line
    assert "infer=90ms" in line and "box_age=110ms" in line

    hint = reporter.report(stats, force=True)
    out = capsys.readouterr().out
    assert "[DEBUG] brightness=60.0" in out
    assert "--effective_damage" in out and hint

    saved = reporter.save(img, detection_result, [], force=True)
    reporter.close()  # flush the background writer
    assert saved is not None and saved.exists() and saved.name == "latest.png"
    written = cv2.imread(str(saved))
    assert written is not None and written.shape == img.shape
    # Raw body and health-bar boxes are drawn even when nothing is last-hittable.
    from my_utils.modules.diagnostics import BAR_BOX_COLOR, BODY_BOX_COLOR

    assert tuple(written[80, 100]) == BODY_BOX_COLOR
    assert tuple(written[85, 105]) == BAR_BOX_COLOR

    # Without a directory nothing is written and no error is raised.
    assert DebugReporter().save(img, detection_result, [], force=True) is None
    with pytest.raises(ValueError):
        annotate_debug_frame(None, detection_result, [])


def test_main_debug_mode_reports_why_nothing_is_drawn(monkeypatch, capsys, tmp_path):
    import main as main_module

    img = np.full((300, 400, 3), 45, dtype=np.uint8)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=30, color="red")

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, _frame):
            return {
                "red_minions": [((95, 85, 170, 160), "red caster minion", 0.91)],
                "red_health_bars": [((100, 90, 160, 98), 0.89)],
                "blue_minions": [],
                "blue_health_bars": [],
                "cur_object_counts": {},
            }

    remaining = [CapturedFrame(image=img.copy(), captured_at=1.0 + i) for i in range(2)]

    class FakeFrameLoader:
        monitor = {"left": 0, "top": 0, "width": 400, "height": 300}
        primary_monitor = {"left": 0, "top": 0, "width": 400, "height": 300}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def grab(self, window_rect=None):
            if not remaining:
                raise KeyboardInterrupt
            return remaining.pop(0)

    monkeypatch.setattr(main_module, "Detector", FakeDetector)
    monkeypatch.setattr(main_module, "FrameLoader", FakeFrameLoader)

    frames_dir = tmp_path / "debug-frames"
    with pytest.raises(KeyboardInterrupt):
        main_module.main(
            [
                "--sync", "--no-overlay", "--no-visualize", "--debug",
                "--debug_interval", "0", "--debug_frames_dir", str(frames_dir),
                # 50% HP on a caster (300 HP) = 150 HP > 80 damage -> not last-hittable
            ]
        )

    out = capsys.readouterr().out
    assert "Diagnostics enabled" in out
    assert "[DEBUG] brightness=" in out
    assert "matched=1" in out and "eligible=0" in out
    assert "--effective_damage" in out
    assert "Both the preview window and the overlay are disabled" in out
    assert (frames_dir / "latest.png").exists()

    # Debug output is opt-in: no [DEBUG] lines without the flag.
    remaining.clear()
    remaining.extend(CapturedFrame(image=img.copy(), captured_at=2.0 + i) for i in range(2))
    with pytest.raises(KeyboardInterrupt):
        main_module.main(["--sync", "--no-overlay", "--no-visualize"])
    assert "[DEBUG]" not in capsys.readouterr().out


def test_describe_thresholds_turns_damage_into_hp_percentages():
    from my_utils.modules.diagnostics import describe_thresholds

    settings = LastHitSettings(
        effective_damage=80.0,
        max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0},
        health_margin=0.0,
    )
    lines = describe_thresholds(settings, "red")
    report = "\n".join(lines)
    assert "effective_damage=80" in report
    assert "red caster        : only while the bar reads <= 27% HP" in report
    assert "red melee         : only while the bar reads <= 16% HP" in report
    assert "red siege         : only while the bar reads <= 5% HP" in report

    weak = LastHitSettings(effective_damage=10.0, max_health={"caster": 300.0}, health_margin=0.1)
    assert "never last-hittable" in "\n".join(describe_thresholds(weak))

    huge = LastHitSettings(effective_damage=900.0, max_health={"caster": 300.0})
    assert "any HP" in "\n".join(describe_thresholds(huge))


def make_wave_bar(width=30, height=5, fill=8):
    """A League-like bar: saturated fill, dark depleted part, 1px border."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:, :fill] = [55, 55, 185]
    bar[:, fill:] = [28, 22, 32]
    bar[0, :] = [15, 15, 15]
    bar[-1, :] = [15, 15, 15]
    bar[:, 0] = [15, 15, 15]
    bar[:, -1] = [15, 15, 15]
    return cv2.GaussianBlur(bar, (3, 1), 0.6)


def test_health_bar_locator_reads_bars_from_pixels_around_bodies():
    from my_utils.modules.bar_locator import HealthBarLocator

    locator = HealthBarLocator()
    for background in ([45, 45, 45], [40, 90, 60], [120, 120, 118], [170, 170, 170]):
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[:] = background
        truth = []
        for i in range(4):
            x = 80 + i * 120
            fill = 4 + i * 6
            img[264:269, x : x + 30] = make_wave_bar(fill=fill)
            truth.append((x, fill / 30.0))

        for x, expected in truth:
            body = (x - 5, 280, x + 35, 340)
            reading = locator.locate(img, body, bar_color="red")
            assert reading is not None, f"no bar found on background {background}"
            # Accurate and slightly conservative (never an under-estimate of remaining HP).
            assert reading.level >= expected - 0.02
            assert reading.level <= expected + 0.06
            assert reading.box[0] == pytest.approx(x, abs=2)

    # Empty terrain -> no bar, no false positive.
    empty = np.full((400, 600, 3), 70, dtype=np.uint8)
    assert locator.locate(empty, (100, 100, 140, 160), "red") is None
    # A body that is far too small to own a bar is rejected.
    assert locator.locate(np.full((50, 50, 3), 45, dtype=np.uint8), (10, 10, 12, 12), "red") is None
    with pytest.raises(ValueError):
        locator.locate(np.zeros((10, 10), dtype=np.uint8), (1, 1, 5, 5), "red")
    with pytest.raises(ValueError):
        HealthBarLocator(min_span_ratio=2.0, max_span_ratio=1.0)


def test_blue_bars_are_found_too():
    from my_utils.modules.bar_locator import HealthBarLocator

    img = np.full((300, 400, 3), 50, dtype=np.uint8)
    bar = np.zeros((5, 30, 3), dtype=np.uint8)
    bar[:, :12] = [200, 150, 40]      # blue team fill (BGR)
    bar[:, 12:] = [25, 20, 30]
    bar[0, :] = [15, 15, 15]
    bar[-1, :] = [15, 15, 15]
    bar[:, 0] = [15, 15, 15]
    bar[:, -1] = [15, 15, 15]
    img[280:285, 100:130] = bar

    reading = HealthBarLocator().locate(img, (96, 296, 136, 356), bar_color="blue")
    assert reading is not None
    assert 0.3 < reading.level < 0.55
    # A red request on a blue bar finds nothing.
    assert HealthBarLocator().locate(img, (96, 296, 136, 356), bar_color="red") is None


def test_wave_without_yolo_bars_still_produces_last_hit_candidates():
    """Regression for the reported case: 8-9 bodies, 0-2 YOLO bars -> no markers at all."""
    img = np.full((720, 1280, 3), 45, dtype=np.uint8)
    bodies = []
    expected_levels = []
    for i in range(8):
        x = 120 + i * 130
        y = 320 + (i % 2) * 70
        fill = 3 + i * 3
        img[y - 16 : y - 11, x : x + 30] = make_wave_bar(fill=fill)
        bodies.append(((x - 4, y, x + 36, y + 60), "red caster minion", 0.88))
        expected_levels.append(fill / 30.0)

    detection_result = {
        "red_minions": bodies,
        "red_health_bars": [],          # YOLOv9 found no bars: this is the bug scenario
        "blue_minions": [],
        "blue_health_bars": [],
    }

    associator = ObjectAssociator()
    matched = associator(detection_result, img=img, minion_type="red")
    assert len(matched) == 8
    assert associator.last_stats == {"yolo_matches": 0, "fallback_matches": 8}
    for minion, expected in zip(matched, expected_levels):
        assert minion.health_level == pytest.approx(expected, abs=0.06)
        assert minion.minion_type == "red caster minion"

    # The lowest-HP minions are now last-hittable and reach the advisor...
    settings = LastHitSettings(
        effective_damage=80.0,
        max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0},
    )
    ranked = LastHitAdvisor(settings).rank(matched, img.shape)
    assert ranked, "geometric bar reading must make low-HP minions actionable"
    assert all(c.health_upper_bound <= 80.0 for c in ranked)

    # ... and --no-bar_locator restores the old (broken) behaviour.
    legacy = ObjectAssociator(fallback_to_locator=False)
    assert legacy(detection_result, img=img, minion_type="red") == []


def test_cli_exposes_bar_locator_switch():
    parser = build_arg_parser()
    assert parser.parse_args([]).bar_locator is True
    assert parser.parse_args(["--no-bar_locator"]).bar_locator is False
    assert parser.parse_args(["--bar_locator", "False"]).bar_locator is False


def test_process_frame_reports_geometric_bars_in_debug_stats(capsys):
    """End-to-end regression of the reported session: 9 bodies, no YOLO bars, no markers."""
    from main import process_frame
    from my_utils.modules.diagnostics import DebugReporter, format_stats, stats_from_frame

    img = np.full((720, 1280, 3), 53, dtype=np.uint8)
    bodies = []
    for i in range(9):
        x = 110 + i * 120
        y = 330 + (i % 2) * 60
        img[y - 16 : y - 11, x : x + 30] = make_wave_bar(fill=3 + i * 3)
        bodies.append(((x - 4, y, x + 36, y + 60), "red caster minion", 0.85))

    class WaveDetector:
        def __call__(self, _frame):
            return {
                "red_minions": bodies,
                "red_health_bars": [],       # <- the reported situation
                "blue_minions": [],
                "blue_health_bars": [],
            }

    settings = LastHitSettings(
        effective_damage=80.0,
        max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0},
    )
    pipeline_stats = {}
    candidates, visualized = process_frame(
        captured_frame=CapturedFrame(image=img, captured_at=1.0),
        detector=WaveDetector(),  # type: ignore[arg-type]
        object_associator=ObjectAssociator(),
        advisor=LastHitAdvisor(settings),
        annotator=AIAnnotator(),
        minion_color="red",
        pipeline_stats=pipeline_stats,
    )

    assert pipeline_stats["matched"] == 9
    assert pipeline_stats["yolo_matches"] == 0
    assert pipeline_stats["fallback_matches"] == 9
    assert candidates, "low-HP minions must be marked even without YOLO bar detections"
    assert visualized is not None

    stats = stats_from_frame(img, {"red_minions": bodies, "red_health_bars": []},
                             "red", pipeline=pipeline_stats)
    assert "geo bars=9" in format_stats(stats, "red")
    reporter = DebugReporter(minion_color="red", interval=0.0)
    assert reporter.report(stats, force=True) is None  # markers exist -> no hint
    assert "geo bars=9" in capsys.readouterr().out


def test_fps_limiter_and_stage_timer():
    import time

    from my_utils.modules.performance import FpsLimiter, StageTimer

    assert FpsLimiter(0).sleep() == 0.0          # unlimited never sleeps
    limiter = FpsLimiter(50)                     # 20 ms cadence
    started = time.perf_counter()
    for _ in range(6):
        limiter.sleep()
    elapsed = time.perf_counter() - started
    assert 0.07 <= elapsed <= 0.35, elapsed      # ~5 intervals, never faster

    with pytest.raises(ValueError):
        FpsLimiter(-1)
    with pytest.raises(ValueError):
        FpsLimiter(float("nan"))

    timer = StageTimer(window=2)
    with timer.measure("capture"):
        pass
    timer.add("evaluate", 0.002)
    timer.add("evaluate", 0.004)
    timer.add("evaluate", 0.006)                 # window keeps the two newest
    averages = timer.averages()
    assert set(averages) == {"capture", "evaluate"}
    assert averages["evaluate"] == pytest.approx(0.005)
    assert "evaluate=5.0ms" in timer.format()
    timer.add("evaluate", float("nan"))          # ignored
    assert timer.averages()["evaluate"] == pytest.approx(0.005)
    timer.reset()
    assert timer.averages() == {} and timer.format() == ""


def test_inference_pacing_limits_gpu_load():
    """Detection must not run back-to-back: that is what makes the game stutter."""
    import time

    from main import _AsyncDetectionWorker

    class CountingDetector:
        def __init__(self):
            self.calls = 0

        def __call__(self, _frame):
            self.calls += 1
            return {"red_minions": [], "red_health_bars": [],
                    "blue_minions": [], "blue_health_bars": []}

    frame = CapturedFrame(image=np.zeros((300, 400, 3), dtype=np.uint8), captured_at=1.0)

    # Capped at 20 fps: ~0.35 s of submissions must yield well under 10 inferences.
    detector = CountingDetector()
    worker = _AsyncDetectionWorker(detector, max_fps=20)
    worker.start()
    try:
        deadline = time.perf_counter() + 0.35
        while time.perf_counter() < deadline:
            worker.submit(frame)
            time.sleep(0.002)
        assert 3 <= detector.calls <= 9, detector.calls
    finally:
        worker.stop()

    # Unlimited (default 0) keeps the old behaviour: many inferences in the same window.
    detector = CountingDetector()
    worker = _AsyncDetectionWorker(detector)
    worker.start()
    try:
        deadline = time.perf_counter() + 0.2
        while time.perf_counter() < deadline:
            worker.submit(frame)
            time.sleep(0.002)
        assert detector.calls >= 10, detector.calls
    finally:
        worker.stop()

    with pytest.raises(ValueError):
        _AsyncDetectionWorker(CountingDetector(), max_fps=-5)


def test_process_priority_and_torch_threads_are_optional_and_safe(monkeypatch):
    import os
    import types as _types

    from my_utils.modules import performance

    recorded = {}

    class FakeProcess:
        def nice(self, value):
            recorded["nice"] = value

    fake_psutil = _types.SimpleNamespace(
        Process=lambda pid=None: FakeProcess(),
        NORMAL_PRIORITY_CLASS=32,
        BELOW_NORMAL_PRIORITY_CLASS=16384,
        IDLE_PRIORITY_CLASS=64,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert "below" in performance.apply_process_priority("below")
    assert recorded["nice"] == (16384 if os.name == "nt" else 10)
    assert "lowest" in performance.apply_process_priority("lowest")
    with pytest.raises(ValueError):
        performance.apply_process_priority("realtime")

    # psutil missing -> simply no priority change, no crash
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert performance.apply_process_priority("below") is None

    # torch thread limiting (CPU inference)
    monkeypatch.setitem(
        sys.modules, "torch", _types.SimpleNamespace(set_num_threads=lambda n: recorded.update(threads=n))
    )
    assert "4" in performance.set_torch_threads(4)
    assert recorded["threads"] == 4
    with pytest.raises(ValueError):
        performance.set_torch_threads(0)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert performance.set_torch_threads(2) is None


def test_capture_tips_are_specific_to_the_configuration():
    from my_utils.modules.performance import capture_and_processing_tips

    big = "\n".join(capture_and_processing_tips(960, 2560, 1440))
    assert "--imgsz 512" in big
    assert "--H 720 --W 1280" in big
    assert "--infer_fps 10" in big

    small = "\n".join(capture_and_processing_tips(480, 960, 540))
    assert "--imgsz 512" not in small          # already cheaper than that
    assert "--H 720 --W 1280" not in small     # frame is small enough


def test_cli_exposes_performance_flags():
    from main import DEFAULT_INFER_FPS, DEFAULT_LOOP_FPS, DEFAULT_OVERLAY_FPS

    parser = build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.infer_fps == DEFAULT_INFER_FPS
    assert defaults.max_fps == DEFAULT_LOOP_FPS
    assert defaults.overlay_fps == DEFAULT_OVERLAY_FPS
    assert defaults.priority == "below"
    assert defaults.torch_threads is None

    tuned = parser.parse_args(
        ["--infer_fps", "8", "--max_fps", "0", "--overlay_fps", "15",
         "--priority", "lowest", "--torch_threads", "4"]
    )
    assert tuned.infer_fps == 8.0
    assert tuned.max_fps == 0.0                # 0 = unlimited
    assert tuned.overlay_fps == 15.0
    assert tuned.priority == "lowest"
    assert tuned.torch_threads == 4

    with pytest.raises(SystemExit):
        parser.parse_args(["--infer_fps", "-3"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--priority", "realtime"])


def test_debug_reporter_writes_off_the_main_thread(monkeypatch):
    import time

    from my_utils.modules.diagnostics import DebugReporter

    img = np.full((200, 300, 3), 60, dtype=np.uint8)
    detection_result = {"red_minions": [((10, 10, 40, 60), "red caster minion", 0.9)],
                        "red_health_bars": []}
    reporter = DebugReporter(frames_dir=None, interval=0.0)
    assert reporter.save(img, detection_result, [], force=True) is None   # no dir -> no-op

    reporter = DebugReporter(frames_dir="/tmp/lol-debug-async", interval=0.0, async_writes=True)
    start = time.perf_counter()
    path = reporter.save(img, detection_result, [], force=True)
    elapsed = time.perf_counter() - start
    assert path is not None
    assert elapsed < 0.05, f"PNG encoding must not block the capture loop ({elapsed * 1000:.1f} ms)"
    reporter.close()
    assert path.exists()
    # Closing twice is safe and later saves are ignored.
    reporter.close()
    assert reporter.save(img, detection_result, [], force=True) is None

    # Synchronous mode is still available (useful for tests/one-shot dumps).
    sync = DebugReporter(frames_dir="/tmp/lol-debug-sync", interval=0.0, async_writes=False)
    path = sync.save(img, detection_result, [], force=True)
    assert path is not None and path.exists()


def test_main_loop_respects_max_fps(monkeypatch):
    """--max_fps must actually throttle the capture loop (protects the game's CPU time)."""
    import time

    import main as main_module

    iterations = []
    frame = CapturedFrame(image=np.full((120, 160, 3), 40, dtype=np.uint8), captured_at=1.0)

    class FakeFrameLoader:
        monitor = {"left": 0, "top": 0, "width": 160, "height": 120}
        primary_monitor = {"left": 0, "top": 0, "width": 160, "height": 120}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def grab(self, window_rect=None):
            iterations.append(time.perf_counter())
            if len(iterations) > 8:
                raise KeyboardInterrupt
            return frame

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, _img):
            return {"red_minions": [], "red_health_bars": [],
                    "blue_minions": [], "blue_health_bars": []}

    monkeypatch.setattr(main_module, "FrameLoader", FakeFrameLoader)
    monkeypatch.setattr(main_module, "Detector", FakeDetector)

    started = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        main_module.main(["--sync", "--no-overlay", "--no-visualize",
                          "--max_fps", "25", "--infer_fps", "5"])
    elapsed = time.perf_counter() - started

    assert len(iterations) == 9
    # 8 intervals at 25 fps ~= 0.32 s; the unlimited loop would finish in microseconds.
    assert elapsed >= 0.2, elapsed


def _candidate(x, y, hp=40.0, kind="red caster minion", health=0.13, held=False):
    return LastHitCandidate(
        minion=Minion(kind, health, (x, y, x + 40, y + 60), 0.9),
        health_upper_bound=hp,
        distance=1.0,
        held=held,
    )


def test_minion_tracker_holds_and_then_drops_markers():
    from my_utils.modules.tracking import MinionTracker

    with pytest.raises(ValueError):
        MinionTracker(hold_ms=-1)
    with pytest.raises(ValueError):
        MinionTracker(hold_ms=float("inf"))

    tracker = MinionTracker(hold_ms=250)
    first = _candidate(100, 100, hp=39)
    other = _candidate(400, 100, hp=52, kind="red melee minion")
    shape = (300, 800, 3)

    # Visible candidates are passed straight through, not flagged as held.
    now = 10.0
    out = tracker.update([first, other], shape, now=now)
    assert [c.held for c in out] == [False, False]
    assert tracker.track_count == 2

    # One frame where the (nearly dead) minion's bar is unreadable: the marker is held
    # with its LAST reading so the player can still hit it.
    out = tracker.update([other], shape, now=now + 0.1)
    assert sorted(c.held for c in out) == [False, True]
    held = [c for c in out if c.held][0]
    assert held.health_upper_bound == pytest.approx(39)
    assert held.minion.bbox == first.minion.bbox

    # Still held just before the deadline ...
    out = tracker.update([], shape, now=now + 0.24)
    assert len(out) == 2 and all(c.held for c in out)

    # ... and gone once the grace period expired (no corpse markers).
    out = tracker.update([], shape, now=now + 0.40)
    assert out == []
    assert tracker.track_count == 0

    # hold_ms=0 disables holding entirely.
    instant = MinionTracker(hold_ms=0)
    instant.update([first], shape, now=1.0)
    assert instant.update([], shape, now=1.001) == []


def test_minion_tracker_keeps_neighbours_apart():
    from my_utils.modules.tracking import MinionTracker

    tracker = MinionTracker(hold_ms=250)
    left = _candidate(100, 100, hp=30)
    right = _candidate(175, 100, hp=80, health=0.27)
    tracker.update([left, right], (300, 400, 3), now=1.0)

    # Only the left minion survives; its held marker must keep the LEFT HP reading.
    out = tracker.update([right], (300, 400, 3), now=1.05)
    held = [c for c in out if c.held]
    assert len(held) == 1
    assert held[0].minion.bbox == left.minion.bbox
    assert held[0].health_upper_bound == pytest.approx(30)


def test_overlay_draws_hp_labels_and_dashes_held_markers():
    from my_utils.visualizers.overlay import DesktopOverlay

    class _Canvas:
        def __init__(self):
            self.polygons, self.texts = [], []

        def delete(self, *_a):
            self.polygons.clear()
            self.texts.clear()

        def create_polygon(self, *coords, **kwargs):
            self.polygons.append((coords, kwargs))

        def create_text(self, *coords, **kwargs):
            self.texts.append((coords, kwargs))

    overlay = DesktopOverlay.__new__(DesktopOverlay)
    overlay.monitor_left = overlay.monitor_top = 0
    overlay.monitor_width, overlay.monitor_height = 1920, 1080
    overlay.marker_gap = 16.0
    overlay.show_labels = True
    overlay.label_font = ("Segoe UI", 9, "bold")
    overlay.label_offset = 6.0
    overlay._display_scale = 1.0
    overlay._max_interval = 0.0
    overlay._last_render = 0.0
    overlay._closed = False
    overlay._tkinter = None
    overlay._root = type("R", (), {"update": lambda self: None})()
    overlay._canvas = _Canvas()

    live = _candidate(960, 500, hp=57)
    held = _candidate(480, 500, hp=33, held=True)
    overlay.render([live, held], (1080, 1920, 3))

    labels = [kwargs["text"] for _, kwargs in overlay._canvas.texts]
    assert labels == ["57 HP", "33 HP?"]          # '?' marks an expired reading
    polygons = [kwargs for _, kwargs in overlay._canvas.polygons]
    assert polygons[0]["dash"] == ()
    assert polygons[1]["dash"] == (2, 2)
    from my_utils.visualizers.overlay import HELD_SECONDARY_COLOR

    assert polygons[1]["fill"] == HELD_SECONDARY_COLOR

    # Labels can be switched off entirely.
    overlay.show_labels = False
    overlay._max_interval = 0.0
    overlay.render([live], (1080, 1920, 3))
    assert overlay._canvas.texts == []


def test_annotator_marks_held_candidates(capsys=None):
    from my_utils.visualizers.annotator import HELD_PRIMARY_COLOR

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    held = _candidate(180, 180, hp=33, held=True)
    rendered = AIAnnotator()(candidates=[held], img=frame)
    assert tuple(rendered[180, 200]) == HELD_PRIMARY_COLOR
    assert not np.array_equal(rendered, frame)


def test_cli_exposes_hold_and_label_flags():
    parser = build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.hold_ms == 250.0
    assert defaults.overlay_labels is True

    tuned = parser.parse_args(["--hold_ms", "400", "--no-overlay_labels"])
    assert tuned.hold_ms == 400.0
    assert tuned.overlay_labels is False
    assert parser.parse_args(["--overlay_labels", "False"]).overlay_labels is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--hold_ms", "-5"])


def test_hold_survives_a_dropped_detection_in_the_pipeline():
    """A minion seen, then missed for one frame, must keep its marker (then drop)."""
    from my_utils.modules.tracking import MinionTracker

    settings = LastHitSettings(
        effective_damage=80.0, max_health={"caster": 300.0, "melee": 500.0, "siege": 1500.0}
    )
    img = np.full((300, 400, 3), 45, dtype=np.uint8)
    img[90:98, 100:160] = make_synthetic_health_bar(width=60, height=8, fill_px=8, color="red")

    detections = {
        "red_minions": [((95, 85, 170, 160), "red caster minion", 0.91)],
        "red_health_bars": [((100, 90, 160, 98), 0.89)],
        "blue_minions": [],
        "blue_health_bars": [],
    }
    empty = {"red_minions": [], "red_health_bars": [],
             "blue_minions": [], "blue_health_bars": []}

    tracker = MinionTracker(hold_ms=250)
    associator = ObjectAssociator()
    advisor = LastHitAdvisor(settings)

    frame = CapturedFrame(image=img.copy(), captured_at=10.0)
    candidates, _ = evaluate_with_latest_detections(
        captured_frame=frame,
        latest_detection=(frame, detections),
        object_associator=associator,
        advisor=advisor,
        tracker=tracker,
        gate=None,
    )
    assert len(candidates) == 1 and candidates[0].held is False
    hp = candidates[0].health_upper_bound
    assert 0 < hp <= 80

    # Next detector result shows nothing (bar unreadable for a moment): marker held.
    frame2 = CapturedFrame(image=img.copy(), captured_at=10.1)
    held, _ = evaluate_with_latest_detections(
        captured_frame=frame2,
        latest_detection=(frame2, empty),
        object_associator=associator,
        advisor=advisor,
        tracker=tracker,
        gate=None,
    )
    assert len(held) == 1 and held[0].held is True
    assert held[0].health_upper_bound == pytest.approx(hp)

    # Minion actually died / vanished for longer -> marker disappears.
    frame3 = CapturedFrame(image=img.copy(), captured_at=10.5)
    gone, _ = evaluate_with_latest_detections(
        captured_frame=frame3,
        latest_detection=(frame3, empty),
        object_associator=associator,
        advisor=advisor,
        tracker=tracker,
        gate=None,
    )
    assert gone == []


def test_overlay_label_modes_show_hp_damage_margin():
    from my_utils.visualizers.overlay import DesktopOverlay

    class _Canvas:
        def __init__(self):
            self.texts = []

        def delete(self, *_a):
            self.texts.clear()

        def create_polygon(self, *a, **k):
            pass

        def create_text(self, *a, **k):
            self.texts.append(k["text"])

    def make(mode, damage=None):
        overlay = DesktopOverlay.__new__(DesktopOverlay)
        overlay.monitor_left = overlay.monitor_top = 0
        overlay.monitor_width, overlay.monitor_height = 1920, 1080
        overlay.marker_gap = 16.0
        overlay.show_labels = True
        overlay.label_font = ("Segoe UI", 9, "bold")
        overlay.label_offset = 6.0
        overlay.label_mode = mode
        overlay.damage = damage
        overlay._display_scale = 1.0
        overlay._max_interval = 0.0
        overlay._last_render = 0.0
        overlay._closed = False
        overlay._tkinter = None
        overlay._root = type("R", (), {"update": lambda self: None})()
        overlay._canvas = _Canvas()
        return overlay

    candidate = _candidate(960, 500, hp=58, health=0.193)
    overlay = make("hp")
    overlay.render([candidate], (1080, 1920, 3))
    assert overlay._canvas.texts == ["58 HP"]

    overlay = make("hp_damage", damage=80.0)
    overlay.render([candidate], (1080, 1920, 3))
    assert overlay._canvas.texts == ["58/80"]      # 22 HP of margin left

    overlay = make("percent")
    overlay.render([candidate], (1080, 1920, 3))
    assert overlay._canvas.texts == ["19%"]

    with pytest.raises(ValueError):
        DesktopOverlay(
            {"left": 0, "top": 0, "width": 100, "height": 100}, label_mode="nonsense"
        )

    parser = build_arg_parser()
    assert parser.parse_args([]).overlay_label_mode == "hp"
    assert parser.parse_args(["--overlay_label_mode", "hp_damage"]).overlay_label_mode == "hp_damage"
    with pytest.raises(SystemExit):
        parser.parse_args(["--overlay_label_mode", "bogus"])
