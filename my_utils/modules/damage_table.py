"""Champion base damage table and last-hit damage estimates.

Everything here is static data plus arithmetic: **no game process, memory or
HUD reading** is involved, exactly like the rest of this companion.  The base
stats come from Riot's Data Dragon (see ``data/champion_base_stats.json`` and
``tools/build_champion_stats.py`` for the patch they were exported from).

Why a table at all?
-------------------
``--effective_damage`` is the one number the companion cannot read from the
screen.  It can, however, be *estimated* from things that are known before the
game starts: which champion you play, your level, and what you bought.  That is
what :func:`estimate_damage` does - and the result is what you pass (or let the
CLI pass) as ``--effective_damage``.

Two honest caveats, both reflected in the numbers this module returns:

* Damage is only an estimate.  Runes, buffs, dragon/baron buffs, item passives
  and armour on the target all move it, and abilities are ignored because
  last-hit timing is about your **basic attack**.
* Every champion has **0 base ability power**.  AP exists only on items, runes
  and buffs, so the AP column of the table is your item's AP, not a base stat.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "champion_base_stats.json"
DEFAULT_LEVELS = (1, 2, 3)

# League of Legends levels a stat with per-level growth like this:
#   value(n) = base + growth * (n - 1) * (0.7025 + 0.0175 * (n - 1))
# which makes level 2 worth 72 % of one growth step and level 3 147.5 %.
GROWTH_SLOPE = 0.0175
GROWTH_INTERCEPT = 0.7025


@dataclass(frozen=True)
class Item:
    """A purchasable (or starting) item that only matters for flat AD/AP."""

    key: str
    name: str
    attack_damage: float = 0.0
    ability_power: float = 0.0
    cost: int = 0

    @property
    def has_attack_damage(self) -> bool:
        return self.attack_damage != 0.0

    @property
    def has_ability_power(self) -> bool:
        return self.ability_power != 0.0


# Starting items, i.e. what you realistically have at levels 1-3.
ITEMS = {
    "none": Item("none", "no item", 0.0, 0.0, 0),
    "doran_blade": Item("doran_blade", "Doran's Blade", 8.0, 0.0, 450),
    "long_sword": Item("long_sword", "Long Sword", 10.0, 0.0, 350),
    "cull": Item("cull", "Cull", 7.0, 0.0, 450),
    "doran_ring": Item("doran_ring", "Doran's Ring", 0.0, 15.0, 400),
    "dark_seal": Item("dark_seal", "Dark Seal", 0.0, 15.0, 350),
    "amplifying_tome": Item("amplifying_tome", "Amplifying Tome", 0.0, 20.0, 435),
}
ITEM_ALIASES = {
    "dorans_blade": "doran_blade",
    "dorans_blade_": "doran_blade",
    "dorans_ring": "doran_ring",
    "dorans": "doran_blade",
    "blade": "doran_blade",
    "ring": "doran_ring",
    "sword": "long_sword",
    "tome": "amplifying_tome",
    "seal": "dark_seal",
    "": "none",
    "no_item": "none",
}

# Minion max HP / armour used for the "how many hits" column.  Kept in line with
# the companion defaults (caster 300) so the table and the markers agree.
DEFAULT_MINION_HP = 300.0
DEFAULT_MINION_ARMOR = 0.0

# Champion names whose Data Dragon spelling is easy to mistype.
CHAMPION_ALIASES = {
    "mundo": "Dr. Mundo",
    "drmundo": "Dr. Mundo",
    "wukong": "Wukong",
    "monkeyking": "Wukong",
    "nunu": "Nunu & Willump",
    "nunuwillump": "Nunu & Willump",
    "renata": "Renata Glasc",
    "jarvan": "Jarvan IV",
    "jarvaniv": "Jarvan IV",
    "aurelion": "Aurelion Sol",
    "asol": "Aurelion Sol",
    "belveth": "Bel'Veth",
    "chogath": "Cho'Gath",
    "ksante": "K'Sante",
    "kaisa": "Kai'Sa",
    "khazix": "Kha'Zix",
    "kogmaw": "Kog'Maw",
    "leblanc": "LeBlanc",
    "leesin": "Lee Sin",
    "masteryi": "Master Yi",
    "missfortune": "Miss Fortune",
    "reksai": "Rek'Sai",
    "tahmkench": "Tahm Kench",
    "twistedfate": "Twisted Fate",
    "velkoz": "Vel'Koz",
    "nunuandwillump": "Nunu & Willump",
}


def level_growth(level: int) -> float:
    """Growth multiplier applied to a per-level stat at ``level`` (level 1 = 0)."""
    try:
        level = int(level)
    except (TypeError, ValueError) as exc:
        raise ValueError("level must be an integer") from exc
    if not 1 <= level <= 18:
        raise ValueError("level must be between 1 and 18")
    steps = level - 1
    return steps * (GROWTH_INTERCEPT + GROWTH_SLOPE * steps)


def mitigation(armor: float) -> float:
    """Fraction of physical damage that gets through ``armor`` points of armour."""
    try:
        armor = float(armor)
    except (TypeError, ValueError) as exc:
        raise ValueError("armor must be a number") from exc
    if not math.isfinite(armor):
        raise ValueError("armor must be a finite number")
    if armor <= 0:
        return 1.0  # negative/zero armour never amplifies damage in this model
    return 100.0 / (100.0 + armor)


def _clean_name(name: str) -> str:
    return "".join(ch for ch in str(name).casefold() if ch.isalnum())


@dataclass(frozen=True)
class ChampionStats:
    """Base stats of one champion (Data Dragon, no items and no runes)."""

    name: str
    key: str
    attack_damage: float
    attack_damage_per_level: float
    armor: float = 0.0
    armor_per_level: float = 0.0
    health: float = 0.0
    health_per_level: float = 0.0
    attack_range: float = 0.0
    attack_type: str | None = None
    adaptive_type: str | None = None

    @property
    def is_ranged(self) -> bool:
        return (self.attack_type or "").upper() == "RANGED"

    def attack_damage_at(self, level: int = 1) -> float:
        """Raw attack damage at ``level`` (base + growth, no items/runes)."""
        return self.attack_damage + self.attack_damage_per_level * level_growth(level)

    def armor_at(self, level: int = 1) -> float:
        return self.armor + self.armor_per_level * level_growth(level)

    def ability_power_at(self, level: int = 1) -> float:
        """Base ability power: always 0 - AP only ever comes from items/runes."""
        level_growth(level)  # validates the level
        return 0.0

    def total_attack_damage(self, level: int = 1, bonus_ad: float = 0.0) -> float:
        return self.attack_damage_at(level) + float(bonus_ad)

    def total_ability_power(self, level: int = 1, bonus_ap: float = 0.0) -> float:
        return self.ability_power_at(level) + float(bonus_ap)


@dataclass(frozen=True)
class DamageEstimate:
    """What one basic attack and one (assumed) ability hit a minion for."""

    champion: str
    level: int
    item: Item
    attack_damage: float            # AD including item/runes
    ability_power: float            # AP including item/runes (no base AP)
    basic_attack_damage: float      # after minion armour
    ability_damage: float | None    # only when an AP ratio was supplied
    minion_armor: float
    hits_to_kill: int | None        # basic attacks for a full-HP caster minion

    @property
    def effective_damage(self) -> float:
        """Value to pass to ``--effective_damage``.

        Basic attacks are what last-hit timing depends on, so an assumed ability
        hit never lowers the number below one basic attack (the tool's damage
        value must be a damage the player can actually rely on).
        """
        if self.ability_damage is None:
            return self.basic_attack_damage
        return min(self.basic_attack_damage, self.ability_damage)

    def describe(self, minion_hp: float = DEFAULT_MINION_HP) -> str:
        parts = [
            f"{self.champion} level {self.level}",
            f"AD {self.attack_damage:.1f}",
            f"AP {self.ability_power:.0f}",
        ]
        line = (
            f"{parts[0]}: {parts[1]}, {parts[2]} "
            f"({self.item.name}) -> {self.basic_attack_damage:.1f} basic-attack damage "
            f"vs {self.minion_armor:.0f} armour"
        )
        if self.hits_to_kill:
            line += f" ({self.hits_to_kill} hit(s) for a {minion_hp:.0f} HP caster minion)"
        if self.ability_damage is not None:
            line += f"; assumed ability hit {self.ability_damage:.1f}"
        return line


@lru_cache(maxsize=1)
def _raw_data() -> dict:
    try:
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"patch": "unknown", "champions": {}}
    except (OSError, ValueError):
        return {"patch": "unknown", "champions": {}}
    if not isinstance(payload, dict):
        return {"patch": "unknown", "champions": {}}
    return payload


def data_patch() -> str:
    """Patch the committed champion table was exported from, if known."""
    return str(_raw_data().get("patch") or "unknown")


@lru_cache(maxsize=1)
def champion_table() -> dict[str, ChampionStats]:
    """All champions by display name (empty dict when the data file is missing)."""
    champions = {}
    for name, record in (_raw_data().get("champions") or {}).items():
        try:
            champions[name] = ChampionStats(
                name=record.get("name") or name,
                key=record.get("key") or name,
                attack_damage=float(record.get("attack_damage", 0.0)),
                attack_damage_per_level=float(record.get("attack_damage_per_level", 0.0)),
                armor=float(record.get("armor", 0.0)),
                armor_per_level=float(record.get("armor_per_level", 0.0)),
                health=float(record.get("health", 0.0)),
                health_per_level=float(record.get("health_per_level", 0.0)),
                attack_range=float(record.get("attack_range", 0.0)),
                attack_type=record.get("attack_type"),
                adaptive_type=record.get("adaptive_type"),
            )
        except (TypeError, ValueError):
            continue
    return dict(sorted(champions.items()))


def champion_names() -> list[str]:
    return list(champion_table())


def resolve_champion(name: str) -> ChampionStats:
    """Look up a champion by name, ignoring case, spaces, dots and apostrophes."""
    if name is None:
        raise ValueError("a champion name is required")
    table = champion_table()
    if not table:
        raise ValueError(
            "no champion data available (data/champion_base_stats.json is missing); "
            "run tools/build_champion_stats.py"
        )
    wanted = str(name).strip()
    if wanted in table:
        return table[wanted]
    cleaned = _clean_name(wanted)
    if not cleaned:
        raise ValueError("a champion name is required")
    alias = CHAMPION_ALIASES.get(cleaned)
    if alias and alias in table:
        return table[alias]
    for champion in table.values():
        if cleaned in {_clean_name(champion.name), _clean_name(champion.key)}:
            return champion
    # Last resort: a unique prefix/substring, so "ahri" and "mundo" both work.
    matches = [c for c in table.values() if cleaned in _clean_name(c.name)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        options = ", ".join(c.name for c in sorted(matches, key=lambda c: len(c.name))[:6])
        raise ValueError(f"champion name {name!r} is ambiguous ({options})")
    raise ValueError(f"unknown champion {name!r}")


def resolve_item(item: str | Item | None) -> Item:
    """Look up a starting item by key or by (loose) name."""
    if isinstance(item, Item):
        return item
    if item is None:
        return ITEMS["none"]
    wanted = str(item).strip()
    if not wanted:
        return ITEMS["none"]
    key = wanted.casefold().replace("-", "_").replace(" ", "_").replace("'", "")
    key = ITEM_ALIASES.get(key, key)
    if key in ITEMS:
        return ITEMS[key]
    for candidate in ITEMS.values():
        if _clean_name(candidate.name) == _clean_name(wanted):
            return candidate
    raise ValueError(
        "unknown item {!r} (choose from: {})".format(wanted, ", ".join(sorted(ITEMS)))
    )


def estimate_damage(
    champion: str | ChampionStats,
    level: int = 1,
    item: str | Item | None = None,
    bonus_ad: float = 0.0,
    bonus_ap: float = 0.0,
    minion_armor: float = DEFAULT_MINION_ARMOR,
    minion_hp: float = DEFAULT_MINION_HP,
    ap_ratio: float | None = None,
    ability_base: float | None = None,
) -> DamageEstimate:
    """Estimate one basic-attack hit (and optionally one ability hit) on a minion.

    An ability hit is only modelled when you supply its numbers: ``ability_base``
    (the spell's flat damage at that rank) and/or ``ap_ratio`` (its AP ratio,
    e.g. ``0.6`` for a 60 %-AP spell).  Both are opt-in because spell base
    damage is not part of the champion base stats.
    """
    stats = resolve_champion(champion) if not isinstance(champion, ChampionStats) else champion
    chosen_item = resolve_item(item)
    level = int(level)
    if not 1 <= level <= 18:
        raise ValueError("level must be between 1 and 18")
    minion_armor = float(minion_armor)
    minion_hp = float(minion_hp)

    attack_damage = stats.attack_damage_at(level) + chosen_item.attack_damage + float(bonus_ad)
    ability_power = chosen_item.ability_power + float(bonus_ap)
    basic = max(0.0, attack_damage * mitigation(minion_armor))

    ability_damage = None
    if ap_ratio is not None or ability_base is not None:
        ratio = 0.0 if ap_ratio is None else float(ap_ratio)
        if ratio < 0 or not math.isfinite(ratio):
            raise ValueError("AP ratio must be a finite, non-negative number")
        base = 0.0 if ability_base is None else float(ability_base)
        if base < 0 or not math.isfinite(base):
            raise ValueError("ability base damage must be a finite, non-negative number")
        ability_damage = max(0.0, (base + ratio * ability_power) * mitigation(minion_armor))

    hits = math.ceil(minion_hp / basic) if basic > 0 and minion_hp > 0 else None
    return DamageEstimate(
        champion=stats.name,
        level=level,
        item=chosen_item,
        attack_damage=attack_damage,
        ability_power=ability_power,
        basic_attack_damage=basic,
        ability_damage=ability_damage,
        minion_armor=minion_armor,
        hits_to_kill=hits,
    )


def damage_rows(
    champion: str | ChampionStats,
    levels=DEFAULT_LEVELS,
    item: str | Item | None = None,
    bonus_ad: float = 0.0,
    bonus_ap: float = 0.0,
    minion_armor: float = DEFAULT_MINION_ARMOR,
    minion_hp: float = DEFAULT_MINION_HP,
    ap_ratio: float | None = None,
    ability_base: float | None = None,
) -> list[DamageEstimate]:
    """One :class:`DamageEstimate` per level (used by the table renderer)."""
    return [
        estimate_damage(
            champion,
            level=level,
            item=item,
            bonus_ad=bonus_ad,
            bonus_ap=bonus_ap,
            minion_armor=minion_armor,
            minion_hp=minion_hp,
            ap_ratio=ap_ratio,
            ability_base=ability_base,
        )
        for level in levels
    ]


def _fmt(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}"


def format_table(
    names=None,
    levels=DEFAULT_LEVELS,
    item: str | Item | None = None,
    bonus_ad: float = 0.0,
    bonus_ap: float = 0.0,
    minion_armor: float = DEFAULT_MINION_ARMOR,
    minion_hp: float = DEFAULT_MINION_HP,
    ap_ratio: float | None = None,
    ability_base: float | None = None,
    digits: int = 1,
) -> str:
    """Render the base-damage table as fixed-width text (used by --damage_table)."""
    chosen_item = resolve_item(item)
    levels = tuple(int(level) for level in levels)
    if not levels:
        raise ValueError("provide at least one level")
    table = champion_table()
    if not table:
        return (
            "No champion data available: data/champion_base_stats.json is missing "
            "(regenerate it with tools/build_champion_stats.py)."
        )

    if names:
        selected = [resolve_champion(name) for name in names]
    else:
        selected = list(table.values())

    rows = []
    for stats in selected:
        estimates = damage_rows(
            stats,
            levels=levels,
            item=chosen_item,
            bonus_ad=bonus_ad,
            bonus_ap=bonus_ap,
            minion_armor=minion_armor,
            minion_hp=minion_hp,
            ap_ratio=ap_ratio,
            ability_base=ability_base,
        )
        rows.append((stats, estimates))

    header = ["Champion", "Type", "Base AD " + "/".join(str(l) for l in levels)]
    if chosen_item.has_attack_damage or bonus_ad:
        header.append(f"AD + {chosen_item.name} " + "/".join(str(l) for l in levels))
    if chosen_item.has_ability_power or bonus_ap:
        header.append("AP")
    if ap_ratio is not None or ability_base is not None:
        header.append(_ability_header(ap_ratio, ability_base))
    header.append(f"Basic hit ({minion_armor:.0f} armour)")
    header.append(f"Hits vs {minion_hp:.0f} HP")

    body = []
    for stats, estimates in rows:
        cells = [
            stats.name,
            "ranged" if stats.is_ranged else "melee",
            " / ".join(_fmt(stats.attack_damage_at(level), digits) for level in levels),
        ]
        if chosen_item.has_attack_damage or bonus_ad:
            cells.append(" / ".join(_fmt(e.attack_damage, digits) for e in estimates))
        if chosen_item.has_ability_power or bonus_ap:
            cells.append(_fmt(estimates[0].ability_power, digits))
        if ap_ratio is not None or ability_base is not None:
            cells.append(_fmt(estimates[0].ability_damage or 0.0, digits))
        first = estimates[0]
        cells.append(_fmt(first.basic_attack_damage, digits))
        cells.append("" if first.hits_to_kill is None else str(first.hits_to_kill))
        body.append(cells)

    widths = [len(cell) for cell in header]
    for cells in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells)]

    def render(cells):
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    lines = [render(header), render(["-" * width for width in widths])]
    lines.extend(render(cells) for cells in body)
    lines.append("")
    lines.append(
        "AD includes the item and any --bonus_ad; AP is item + --bonus_ap because every "
        "champion has 0 base AP."
    )
    lines.append(
        f"Basic hit = AD x 100/(100+{minion_armor:.0f}) minion armour. "
        f"Data: Data Dragon patch {data_patch()}."
    )
    if ap_ratio is not None or ability_base is not None:
        lines.append(
            "Ability column = (" + f"{0.0 if ability_base is None else float(ability_base):g}"
            " flat + " + f"{0.0 if ap_ratio is None else float(ap_ratio):g}"
            " x AP) x mitigation; the spell's own rank base damage is only included "
            "when you pass --ability_base."
        )
    return "\n".join(lines)


def _ability_header(ap_ratio: float | None, ability_base: float | None) -> str:
    parts = []
    if ability_base is not None:
        parts.append(f"{float(ability_base):g} flat")
    if ap_ratio is not None:
        parts.append(f"x{float(ap_ratio):g} AP")
    return "Ability (" + " + ".join(parts) + ")"


def describe_item(item: str | Item | None) -> str:
    chosen = resolve_item(item)
    if chosen.key == "none":
        return "no item"
    bonuses = []
    if chosen.attack_damage:
        bonuses.append(f"+{chosen.attack_damage:g} AD")
    if chosen.ability_power:
        bonuses.append(f"+{chosen.ability_power:g} AP")
    return f"{chosen.name} ({', '.join(bonuses)})"
