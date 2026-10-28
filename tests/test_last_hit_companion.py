import ast
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_arg_parser, process_frame
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
