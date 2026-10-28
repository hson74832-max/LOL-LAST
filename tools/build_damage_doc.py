"""Regenerate ``docs/champion_damage_table.md`` from the committed champion stats.

Run after ``tools/build_champion_stats.py`` brought the data up to date::

    python tools/build_damage_doc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_utils.modules import damage_table as dt  # noqa: E402

DOC_PATH = REPO_ROOT / "docs" / "champion_damage_table.md"
LEVELS = (1, 2, 3)
ITEM = "doran_blade"
MINION_ARMOR = 0.0
MINION_HP = 300.0


def _row(cells) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def ad_table(names) -> list[str]:
    item = dt.resolve_item(ITEM)
    lines = [
        _row(
            ["Champion", "Type"]
            + [f"AD L{level}" for level in LEVELS]
            + [f"{item.name} AD L{level}" for level in LEVELS]
            + ["Hit vs minion (L1)", f"Autos for {MINION_HP:.0f} HP"]
        ),
        "| " + " | ".join(["---"] * (4 + 2 * len(LEVELS))) + " |",
    ]
    for name in names:
        stats = dt.resolve_champion(name)
        base = [stats.attack_damage_at(level) for level in LEVELS]
        with_item = [value + item.attack_damage for value in base]
        estimate = dt.estimate_damage(
            stats, level=1, item=item, minion_armor=MINION_ARMOR, minion_hp=MINION_HP
        )
        lines.append(
            _row(
                [stats.name, "ranged" if stats.is_ranged else "melee"]
                + [f"{value:.0f}" for value in base]
                + [f"{value:.0f}" for value in with_item]
                + [f"{estimate.basic_attack_damage:.0f}"]
                + ["" if estimate.hits_to_kill is None else estimate.hits_to_kill]
            )
        )
    return lines


def ap_table(names) -> list[str]:
    lines = [
        _row(
            ["Champion", "Type"]
            + [f"AD L{level}" for level in LEVELS]
            + ["AP: Doran's Ring", "AP: Dark Seal", "AP: Amplifying Tome"]
        ),
        "| " + " | ".join(["---"] * (3 + len(LEVELS) + 3)) + " |",
    ]
    for name in names:
        stats = dt.resolve_champion(name)
        base = [stats.attack_damage_at(level) for level in LEVELS]
        lines.append(
            _row(
                [stats.name, "ranged" if stats.is_ranged else "melee"]
                + [f"{value:.0f}" for value in base]
                + [f"{dt.ITEMS[key].ability_power:.0f}"
                   for key in ("doran_ring", "dark_seal", "amplifying_tome")]
            )
        )
    return lines


def build() -> str:
    table = dt.champion_table()
    names = list(table)
    ad_champions = [name for name in names if table[name].adaptive_type == "PHYSICAL_DAMAGE"]
    ap_champions = [name for name in names if table[name].adaptive_type != "PHYSICAL_DAMAGE"]
    patch = dt.data_patch()

    lines = [
        "# Champion base damage table (levels 1-3)",
        "",
        f"Generated from Riot Data Dragon base stats (patch **{patch}**) - all "
        f"{len(names)} champions. Regenerate with `python tools/build_damage_doc.py`.",
        "",
        "The last-hit companion cannot read your damage from the screen (that would need "
        "process/memory access, which this project never does), so the damage value is an "
        "input: `--effective_damage`. This table exists so you can pick that number from "
        "real stats instead of guessing.",
        "",
        "## How the numbers are computed",
        "",
        "- **Level scaling** (Riot's formula): "
        "`value(level) = base + growth x (level - 1) x (0.7025 + 0.0175 x (level - 1))`, "
        "so level 2 is worth 72 % of one growth step and level 3 is worth 147.5 %.",
        "- **Basic attack damage** = attack damage x `100 / (100 + minion armour)`. "
        f"The tables below use {MINION_ARMOR:.0f} armour (caster and melee minions). "
        "Siege/cannon minions are noticeably armoured, so pass `--minion_armor 30` (or "
        "higher) when you are last-hitting those.",
        "- **AP never adds to a basic attack.** Every champion has **0 base ability power**; "
        "AP comes only from items, runes and buffs, and only feeds abilities. That is why "
        "the AP table lists item AP instead of a base stat.",
        "- Abilities, on-hit effects (Sheen, Kraken Slayer, Doran's passive), runes, buffs "
        "and levelling order are all ignored. Treat every number as a starting point and "
        "then confirm it in game.",
        "",
        "## AD champions",
        "",
        f"Base AD plus the cheapest AD starting item (**Doran's Blade**, +8 AD, 450 g). "
        f"`Hit vs minion` is one level-1 basic attack against a {MINION_ARMOR:.0f}-armour "
        f"minion; the last column is how many of those hits a {MINION_HP:.0f} HP caster "
        "minion needs.",
        "",
    ]
    lines += ad_table(ad_champions)
    lines += [
        "",
        "## AP-scaling champions",
        "",
        "These champions scale their abilities with ability power, but they still *last hit* "
        "with basic attacks - so the AD columns are what you pass as `--effective_damage` "
        "unless you deliberately last hit with a spell. The AP columns show what each AP "
        "starting item gives you for that spell.",
        "",
    ]
    lines += ap_table(ap_champions)
    lines += [
        "",
        "## Picking `--effective_damage` in practice",
        "",
        "1. Look up your champion, level and starting item above.",
        "2. Use the `Hit vs minion` value (or the item column at your level).",
        "3. Round **down** - the companion only marks a minion when its conservative HP "
        "upper bound is at or below your damage, so overestimating damage is the one way to "
        "get an unwanted marker.",
        "4. Add runes/on-hit effects only if you are sure they apply on every hit "
        "(`--bonus_ad` does the arithmetic for you).",
        "",
        "Or let the tool do it:",
        "",
        "```bat",
        ":: print this table from the CLI",
        "run.bat --damage_table Ahri Lux Aatrox",
        "run.bat --damage_table --item long_sword --minion_armor 30",
        "",
        ":: use the estimate directly as the last-hit damage",
        "run.bat --champion Ahri --level 2 --item doran_ring --overlay_label_mode hp_damage",
        "```",
        "",
        "Example: `--champion Aatrox --level 1 --item doran_blade` resolves to "
        f"{dt.estimate_damage('Aatrox', level=1, item='doran_blade').effective_damage:.1f} "
        "damage per basic attack, so the companion passes exactly that as "
        "`--effective_damage`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(build(), encoding="utf-8")
    print(f"wrote {DOC_PATH} ({len(dt.champion_table())} champions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
