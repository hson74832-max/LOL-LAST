"""Regenerate ``data/champion_base_stats.json`` from Riot Data Dragon data.

The companion never talks to the network at runtime: the champion table is a
small, committed JSON file.  This script is only needed when a new patch
changes base stats.

Usage
-----
Download a Data Dragon ``championFull.json`` (or any mirror that keeps Data
Dragon's per-champion ``stats`` blocks, e.g. the ``champions.json`` of
https://github.com/megadoxs/CDN-Riot) and run::

    python tools/build_champion_stats.py champions.json --patch 15.15.1
    # or, from Data Dragon directly:
    python tools/build_champion_stats.py championFull.json --patch 15.15.1

Only base stats are copied.  Abilities, skins, lore, icons and every other
field are dropped, which keeps the committed file at a few dozen kilobytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "champion_base_stats.json"


def _flat_stat(stats: dict, key: str) -> tuple[float, float]:
    """Return ``(flat, per_level)`` for one Data Dragon stat block."""
    block = stats.get(key) or {}
    try:
        flat = float(block.get("flat", 0.0))
    except (TypeError, ValueError):
        flat = 0.0
    try:
        per_level = float(block.get("perLevel", 0.0))
    except (TypeError, ValueError):
        per_level = 0.0
    return flat, per_level


def _round(value: float, digits: int = 2) -> float:
    return float(round(value, digits))


def champion_record(champion: dict) -> dict:
    """Reduce one Data Dragon champion object to the fields the tool needs."""
    stats = champion.get("stats") or {}
    attack_damage, attack_damage_per_level = _flat_stat(stats, "attackDamage")
    armor, armor_per_level = _flat_stat(stats, "armor")
    health, health_per_level = _flat_stat(stats, "health")
    attack_range, _ = _flat_stat(stats, "attackRange")
    return {
        "name": champion.get("name") or champion.get("key") or "",
        "key": champion.get("key") or champion.get("id") or "",
        "attack_damage": _round(attack_damage),
        "attack_damage_per_level": _round(attack_damage_per_level),
        "armor": _round(armor),
        "armor_per_level": _round(armor_per_level),
        "health": _round(health),
        "health_per_level": _round(health_per_level),
        "attack_range": _round(attack_range),
        "attack_type": (champion.get("attackType") or "").upper() or None,
        "adaptive_type": (champion.get("adaptiveType") or "").upper() or None,
    }


def load_champions(path: Path) -> dict:
    """Load either a Data Dragon ``championFull.json`` or a keyed mirror."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        payload = payload["data"]  # Data Dragon championFull.json
    if isinstance(payload, list):
        return {record.get("key") or record.get("name"): record for record in payload}
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"unsupported champion data layout in {path}")


def build(source: Path, patch: str) -> dict:
    champions = {}
    for key, champion in load_champions(source).items():
        record = champion_record(champion)
        if not record["name"]:
            continue
        champions[record["name"]] = record
    if not champions:
        raise ValueError(f"no champions found in {source}")
    return {
        "source": "Riot Data Dragon champion stats",
        "patch": patch,
        "note": (
            "Base stats only. Base ability power is 0 for every champion in "
            "League of Legends, so AP comes exclusively from items, runes and buffs."
        ),
        "champions": dict(sorted(champions.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="championFull.json or a keyed champion mirror")
    parser.add_argument("--patch", default="unknown", help="Patch the data was taken from")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = build(args.source, args.patch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    count = len(payload["champions"])
    print(f"wrote {args.output} ({count} champions, patch {payload['patch']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
