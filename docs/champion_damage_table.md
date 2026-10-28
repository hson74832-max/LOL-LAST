# Champion base damage table (levels 1-3)

Generated from Riot Data Dragon base stats (patch **15.15.1**) - all 171 champions. Regenerate with `python tools/build_damage_doc.py`.

The last-hit companion cannot read your damage from the screen (that would need process/memory access, which this project never does), so the damage value is an input: `--effective_damage`. This table exists so you can pick that number from real stats instead of guessing.

## How the numbers are computed

- **Level scaling** (Riot's formula): `value(level) = base + growth x (level - 1) x (0.7025 + 0.0175 x (level - 1))`, so level 2 is worth 72 % of one growth step and level 3 is worth 147.5 %.
- **Basic attack damage** = attack damage x `100 / (100 + minion armour)`. The tables below use 0 armour (caster and melee minions). Siege/cannon minions are noticeably armoured, so pass `--minion_armor 30` (or higher) when you are last-hitting those.
- **AP never adds to a basic attack.** Every champion has **0 base ability power**; AP comes only from items, runes and buffs, and only feeds abilities. That is why the AP table lists item AP instead of a base stat.
- Abilities, on-hit effects (Sheen, Kraken Slayer, Doran's passive), runes, buffs and levelling order are all ignored. Treat every number as a starting point and then confirm it in game.

## AD champions

Base AD plus the cheapest AD starting item (**Doran's Blade**, +8 AD, 450 g). `Hit vs minion` is one level-1 basic attack against a 0-armour minion; the last column is how many of those hits a 300 HP caster minion needs.

| Champion | Type | AD L1 | AD L2 | AD L3 | Doran's Blade AD L1 | Doran's Blade AD L2 | Doran's Blade AD L3 | Hit vs minion (L1) | Autos for 300 HP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aatrox | melee | 60 | 64 | 67 | 68 | 72 | 75 | 68 | 5 |
| Akali | melee | 62 | 64 | 67 | 70 | 72 | 75 | 70 | 5 |
| Akshan | ranged | 52 | 54 | 56 | 60 | 62 | 64 | 60 | 5 |
| Ambessa | melee | 63 | 65 | 67 | 71 | 73 | 75 | 71 | 5 |
| Aphelios | ranged | 55 | 57 | 58 | 63 | 65 | 66 | 63 | 5 |
| Ashe | ranged | 59 | 61 | 63 | 67 | 69 | 71 | 67 | 5 |
| Bel'Veth | melee | 60 | 61 | 62 | 68 | 69 | 70 | 68 | 5 |
| Blitzcrank | melee | 62 | 65 | 67 | 70 | 73 | 75 | 70 | 5 |
| Briar | melee | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Caitlyn | ranged | 60 | 63 | 66 | 68 | 71 | 74 | 68 | 5 |
| Camille | melee | 68 | 71 | 74 | 76 | 79 | 82 | 76 | 4 |
| Corki | ranged | 52 | 53 | 55 | 60 | 61 | 63 | 60 | 5 |
| Darius | melee | 64 | 68 | 71 | 72 | 76 | 79 | 72 | 5 |
| Dr. Mundo | melee | 61 | 63 | 65 | 69 | 71 | 73 | 69 | 5 |
| Draven | ranged | 62 | 65 | 67 | 70 | 73 | 75 | 70 | 5 |
| Ezreal | ranged | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Fiora | melee | 66 | 68 | 71 | 74 | 76 | 79 | 74 | 5 |
| Gangplank | melee | 64 | 67 | 70 | 72 | 75 | 78 | 72 | 5 |
| Garen | melee | 69 | 72 | 76 | 77 | 80 | 84 | 77 | 4 |
| Gnar | ranged | 60 | 62 | 65 | 68 | 70 | 73 | 68 | 5 |
| Graves | ranged | 68 | 71 | 74 | 76 | 79 | 82 | 76 | 4 |
| Hecarim | melee | 66 | 69 | 71 | 74 | 77 | 79 | 74 | 5 |
| Illaoi | melee | 65 | 69 | 72 | 73 | 77 | 80 | 73 | 5 |
| Irelia | melee | 65 | 68 | 70 | 73 | 76 | 78 | 73 | 5 |
| Jarvan IV | melee | 64 | 66 | 68 | 72 | 74 | 76 | 72 | 5 |
| Jax | melee | 68 | 71 | 74 | 76 | 79 | 82 | 76 | 4 |
| Jayce | ranged | 59 | 62 | 65 | 67 | 70 | 73 | 67 | 5 |
| Jhin | ranged | 59 | 62 | 65 | 67 | 70 | 73 | 67 | 5 |
| Jinx | ranged | 59 | 61 | 64 | 67 | 69 | 72 | 67 | 5 |
| K'Sante | melee | 64 | 67 | 69 | 72 | 75 | 77 | 72 | 5 |
| Kalista | ranged | 57 | 60 | 63 | 65 | 68 | 71 | 65 | 5 |
| Kayle | melee | 50 | 52 | 54 | 58 | 60 | 62 | 58 | 6 |
| Kayn | melee | 68 | 70 | 73 | 76 | 78 | 81 | 76 | 4 |
| Kha'Zix | melee | 60 | 62 | 65 | 68 | 70 | 73 | 68 | 5 |
| Kindred | ranged | 65 | 67 | 70 | 73 | 75 | 78 | 73 | 5 |
| Kled | melee | 65 | 68 | 70 | 73 | 76 | 78 | 73 | 5 |
| Kog'Maw | ranged | 61 | 63 | 66 | 69 | 71 | 74 | 69 | 5 |
| Lee Sin | melee | 66 | 69 | 71 | 74 | 77 | 79 | 74 | 5 |
| Leona | melee | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Lucian | ranged | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Master Yi | melee | 65 | 67 | 69 | 73 | 75 | 77 | 73 | 5 |
| Miss Fortune | ranged | 53 | 55 | 57 | 61 | 63 | 65 | 61 | 5 |
| Naafiri | melee | 55 | 56 | 58 | 63 | 64 | 66 | 63 | 5 |
| Nasus | melee | 67 | 70 | 73 | 75 | 78 | 81 | 75 | 4 |
| Nilah | melee | 60 | 61 | 63 | 68 | 69 | 71 | 68 | 5 |
| Nocturne | melee | 62 | 64 | 66 | 70 | 72 | 74 | 70 | 5 |
| Olaf | melee | 68 | 71 | 75 | 76 | 79 | 83 | 76 | 4 |
| Ornn | melee | 69 | 72 | 74 | 77 | 80 | 82 | 77 | 4 |
| Pantheon | melee | 64 | 66 | 69 | 72 | 74 | 77 | 72 | 5 |
| Poppy | melee | 60 | 63 | 66 | 68 | 71 | 74 | 68 | 5 |
| Pyke | melee | 62 | 63 | 65 | 70 | 71 | 73 | 70 | 5 |
| Qiyana | melee | 66 | 68 | 71 | 74 | 76 | 79 | 74 | 5 |
| Quinn | ranged | 59 | 61 | 63 | 67 | 69 | 71 | 67 | 5 |
| Rammus | melee | 65 | 67 | 69 | 73 | 75 | 77 | 73 | 5 |
| Rek'Sai | melee | 58 | 60 | 62 | 66 | 68 | 70 | 66 | 5 |
| Renekton | melee | 69 | 72 | 75 | 77 | 80 | 83 | 77 | 4 |
| Rengar | melee | 68 | 70 | 72 | 76 | 78 | 80 | 76 | 4 |
| Riven | melee | 64 | 66 | 68 | 72 | 74 | 76 | 72 | 5 |
| Samira | ranged | 57 | 59 | 61 | 65 | 67 | 69 | 65 | 5 |
| Senna | ranged | 50 | 50 | 50 | 58 | 58 | 58 | 58 | 6 |
| Sett | melee | 60 | 63 | 66 | 68 | 71 | 74 | 68 | 5 |
| Shaco | melee | 63 | 65 | 67 | 71 | 73 | 75 | 71 | 5 |
| Shen | melee | 64 | 66 | 68 | 72 | 74 | 76 | 72 | 5 |
| Shyvana | melee | 66 | 68 | 70 | 74 | 76 | 78 | 74 | 5 |
| Sion | melee | 68 | 71 | 74 | 76 | 79 | 82 | 76 | 4 |
| Sivir | ranged | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Skarner | melee | 63 | 65 | 67 | 71 | 73 | 75 | 71 | 5 |
| Smolder | ranged | 60 | 62 | 63 | 68 | 70 | 71 | 68 | 5 |
| Talon | melee | 68 | 70 | 73 | 76 | 78 | 81 | 76 | 4 |
| Taric | melee | 55 | 58 | 60 | 63 | 66 | 68 | 63 | 5 |
| Thresh | ranged | 56 | 58 | 59 | 64 | 66 | 67 | 64 | 5 |
| Tristana | ranged | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Trundle | melee | 68 | 71 | 74 | 76 | 79 | 82 | 76 | 4 |
| Tryndamere | melee | 66 | 69 | 72 | 74 | 77 | 80 | 74 | 5 |
| Twitch | ranged | 59 | 61 | 64 | 67 | 69 | 72 | 67 | 5 |
| Udyr | melee | 62 | 65 | 68 | 70 | 73 | 76 | 70 | 5 |
| Urgot | ranged | 63 | 66 | 69 | 71 | 74 | 77 | 71 | 5 |
| Varus | ranged | 59 | 61 | 64 | 67 | 69 | 72 | 67 | 5 |
| Vayne | ranged | 60 | 62 | 63 | 68 | 70 | 71 | 68 | 5 |
| Vi | melee | 63 | 65 | 67 | 71 | 73 | 75 | 71 | 5 |
| Viego | melee | 57 | 60 | 62 | 65 | 68 | 70 | 65 | 5 |
| Volibear | melee | 60 | 63 | 65 | 68 | 71 | 73 | 68 | 5 |
| Warwick | melee | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Wukong | melee | 66 | 69 | 71 | 74 | 77 | 79 | 74 | 5 |
| Xayah | ranged | 60 | 63 | 65 | 68 | 71 | 73 | 68 | 5 |
| Xin Zhao | melee | 63 | 65 | 67 | 71 | 73 | 75 | 71 | 5 |
| Yasuo | melee | 60 | 62 | 64 | 68 | 70 | 72 | 68 | 5 |
| Yone | melee | 60 | 61 | 63 | 68 | 69 | 71 | 68 | 5 |
| Yorick | melee | 62 | 66 | 69 | 70 | 74 | 77 | 70 | 5 |
| Yunara | ranged | 56 | 58 | 60 | 64 | 66 | 68 | 64 | 5 |
| Zed | melee | 63 | 65 | 68 | 71 | 73 | 76 | 71 | 5 |
| Zeri | ranged | 56 | 57 | 59 | 64 | 65 | 67 | 64 | 5 |

## AP-scaling champions

These champions scale their abilities with ability power, but they still *last hit* with basic attacks - so the AD columns are what you pass as `--effective_damage` unless you deliberately last hit with a spell. The AP columns show what each AP starting item gives you for that spell.

| Champion | Type | AD L1 | AD L2 | AD L3 | AP: Doran's Ring | AP: Dark Seal | AP: Amplifying Tome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ahri | ranged | 53 | 55 | 57 | 15 | 15 | 20 |
| Alistar | melee | 62 | 65 | 68 | 15 | 15 | 20 |
| Amumu | melee | 57 | 60 | 63 | 15 | 15 | 20 |
| Anivia | ranged | 51 | 53 | 56 | 15 | 15 | 20 |
| Annie | ranged | 50 | 52 | 54 | 15 | 15 | 20 |
| Aurelion Sol | ranged | 58 | 60 | 63 | 15 | 15 | 20 |
| Aurora | ranged | 53 | 55 | 57 | 15 | 15 | 20 |
| Azir | ranged | 56 | 59 | 61 | 15 | 15 | 20 |
| Bard | ranged | 52 | 54 | 56 | 15 | 15 | 20 |
| Brand | ranged | 57 | 59 | 61 | 15 | 15 | 20 |
| Braum | melee | 55 | 57 | 60 | 15 | 15 | 20 |
| Cassiopeia | ranged | 53 | 55 | 57 | 15 | 15 | 20 |
| Cho'Gath | melee | 69 | 72 | 75 | 15 | 15 | 20 |
| Diana | melee | 57 | 59 | 61 | 15 | 15 | 20 |
| Ekko | melee | 58 | 60 | 62 | 15 | 15 | 20 |
| Elise | ranged | 55 | 57 | 59 | 15 | 15 | 20 |
| Evelynn | melee | 61 | 63 | 65 | 15 | 15 | 20 |
| Fiddlesticks | ranged | 55 | 57 | 59 | 15 | 15 | 20 |
| Fizz | melee | 58 | 60 | 62 | 15 | 15 | 20 |
| Galio | melee | 59 | 62 | 64 | 15 | 15 | 20 |
| Gragas | melee | 64 | 67 | 69 | 15 | 15 | 20 |
| Gwen | melee | 63 | 65 | 67 | 15 | 15 | 20 |
| Heimerdinger | ranged | 56 | 58 | 60 | 15 | 15 | 20 |
| Hwei | ranged | 54 | 56 | 59 | 15 | 15 | 20 |
| Ivern | ranged | 50 | 52 | 54 | 15 | 15 | 20 |
| Janna | ranged | 47 | 49 | 51 | 15 | 15 | 20 |
| Kai'Sa | ranged | 59 | 61 | 63 | 15 | 15 | 20 |
| Karma | ranged | 51 | 53 | 56 | 15 | 15 | 20 |
| Karthus | ranged | 46 | 48 | 51 | 15 | 15 | 20 |
| Kassadin | melee | 59 | 62 | 65 | 15 | 15 | 20 |
| Katarina | melee | 58 | 60 | 63 | 15 | 15 | 20 |
| Kennen | ranged | 48 | 51 | 54 | 15 | 15 | 20 |
| LeBlanc | ranged | 55 | 57 | 58 | 15 | 15 | 20 |
| Lillia | melee | 61 | 63 | 66 | 15 | 15 | 20 |
| Lissandra | ranged | 55 | 57 | 59 | 15 | 15 | 20 |
| Lulu | ranged | 47 | 49 | 51 | 15 | 15 | 20 |
| Lux | ranged | 54 | 56 | 59 | 15 | 15 | 20 |
| Malphite | melee | 62 | 65 | 68 | 15 | 15 | 20 |
| Malzahar | ranged | 55 | 57 | 59 | 15 | 15 | 20 |
| Maokai | melee | 64 | 66 | 69 | 15 | 15 | 20 |
| Mel | ranged | 54 | 56 | 59 | 15 | 15 | 20 |
| Milio | ranged | 48 | 50 | 53 | 15 | 15 | 20 |
| Mordekaiser | melee | 61 | 64 | 67 | 15 | 15 | 20 |
| Morgana | ranged | 56 | 59 | 61 | 15 | 15 | 20 |
| Nami | ranged | 54 | 56 | 59 | 15 | 15 | 20 |
| Nautilus | melee | 61 | 63 | 66 | 15 | 15 | 20 |
| Neeko | ranged | 48 | 50 | 52 | 15 | 15 | 20 |
| Nidalee | ranged | 58 | 61 | 63 | 15 | 15 | 20 |
| Nunu & Willump | melee | 61 | 63 | 65 | 15 | 15 | 20 |
| Orianna | ranged | 44 | 46 | 48 | 15 | 15 | 20 |
| Rakan | melee | 62 | 65 | 67 | 15 | 15 | 20 |
| Rell | melee | 55 | 57 | 59 | 15 | 15 | 20 |
| Renata Glasc | ranged | 49 | 51 | 53 | 15 | 15 | 20 |
| Rumble | melee | 64 | 66 | 69 | 15 | 15 | 20 |
| Ryze | ranged | 58 | 60 | 62 | 15 | 15 | 20 |
| Sejuani | melee | 66 | 69 | 72 | 15 | 15 | 20 |
| Seraphine | ranged | 50 | 52 | 54 | 15 | 15 | 20 |
| Singed | melee | 63 | 65 | 68 | 15 | 15 | 20 |
| Sona | ranged | 49 | 51 | 53 | 15 | 15 | 20 |
| Soraka | ranged | 50 | 52 | 54 | 15 | 15 | 20 |
| Swain | ranged | 58 | 60 | 62 | 15 | 15 | 20 |
| Sylas | melee | 61 | 63 | 65 | 15 | 15 | 20 |
| Syndra | ranged | 54 | 56 | 58 | 15 | 15 | 20 |
| Tahm Kench | melee | 56 | 58 | 61 | 15 | 15 | 20 |
| Taliyah | ranged | 58 | 60 | 63 | 15 | 15 | 20 |
| Teemo | ranged | 54 | 56 | 58 | 15 | 15 | 20 |
| Twisted Fate | ranged | 52 | 54 | 56 | 15 | 15 | 20 |
| Veigar | ranged | 52 | 54 | 56 | 15 | 15 | 20 |
| Vel'Koz | ranged | 55 | 57 | 60 | 15 | 15 | 20 |
| Vex | ranged | 54 | 56 | 58 | 15 | 15 | 20 |
| Viktor | ranged | 53 | 55 | 57 | 15 | 15 | 20 |
| Vladimir | ranged | 55 | 57 | 59 | 15 | 15 | 20 |
| Xerath | ranged | 55 | 57 | 59 | 15 | 15 | 20 |
| Yuumi | ranged | 49 | 51 | 54 | 15 | 15 | 20 |
| Zac | melee | 60 | 62 | 65 | 15 | 15 | 20 |
| Ziggs | ranged | 55 | 57 | 60 | 15 | 15 | 20 |
| Zilean | ranged | 52 | 54 | 56 | 15 | 15 | 20 |
| Zoe | ranged | 58 | 60 | 63 | 15 | 15 | 20 |
| Zyra | ranged | 53 | 55 | 58 | 15 | 15 | 20 |

## Picking `--effective_damage` in practice

1. Look up your champion, level and starting item above.
2. Use the `Hit vs minion` value (or the item column at your level).
3. Round **down** - the companion only marks a minion when its conservative HP upper bound is at or below your damage, so overestimating damage is the one way to get an unwanted marker.
4. Add runes/on-hit effects only if you are sure they apply on every hit (`--bonus_ad` does the arithmetic for you).

Or let the tool do it:

```bat
:: print this table from the CLI
run.bat --damage_table Ahri Lux Aatrox
run.bat --damage_table --item long_sword --minion_armor 30

:: use the estimate directly as the last-hit damage
run.bat --champion Ahri --level 2 --item doran_ring --overlay_label_mode hp_damage
```

Example: `--champion Aatrox --level 1 --item doran_blade` resolves to 68.0 damage per basic attack, so the companion passes exactly that as `--effective_damage`.
