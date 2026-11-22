#!/usr/bin/env python3
"""
query_engine.py — a small, fully local natural-language query engine over
already-parsed GameChanger roster data.

No network calls, no LLM: every answer is computed directly from PlayerStats
objects with plain Python string matching and arithmetic. That's a deliberate
trade-off (see webapp.py's privacy notes) — it means the chat box can only
answer a specific set of question *shapes*, not literally anything. When it
can't confidently tell what's being asked, it says so and shows examples of
what does work, rather than guessing and returning a wrong number.

Understood question shapes:
  - Leaderboards:   "who has the most home runs?", "top 5 in OBP",
                     "who has the worst error rate?", "least strikeouts"
  - Player lookup:  "what's Luke's batting average?", "how many RBI does
                     Maya Chen have?"
  - Comparison:     "who has more walks, Luke or Owen?", "Luke vs Owen OPS"
  - Team aggregate:  "what's the team's total home runs?", "team average OBP"
  - Roster meta:     "how many players are on the roster?", "who are the
                     regulars?", "who's on the team?"
  - Pitching:        "who throws the most strikes?", "who has the most
                     strikeouts pitching?", "Maya's walks per inning",
                     "who throws the hardest?"

Pitching vs batting: words like "strikeouts", "walks" and "hits" mean
different things for a batter and a pitcher. A question that mentions
pitching (pitch, pitcher, threw, mound, allowed, batters...) maps those
words to the pitching stat; otherwise they mean the batting stat. A
pitching question that names no pitching stat gets a clarifying reply,
never a batting answer.
"""

from __future__ import annotations

import re
from typing import Optional

import generate_reports as gr
from generate_reports import PlayerStats

# ---------------------------------------------------------------------------
# Stat vocabulary
# ---------------------------------------------------------------------------

# alias phrase (lowercase) -> canonical PlayerStats attribute name. Longer,
# more specific phrases are tried before shorter ones (see match_stat()), so
# e.g. "batting average" is checked before the bare "average".
STAT_ALIASES: dict[str, str] = {
    # batting counting stats
    "plate appearances": "pa", "plate appearance": "pa", "pa": "pa",
    "at bats": "ab", "at-bats": "ab", "at bat": "ab", "ab": "ab", "abs": "ab",
    "hits": "h", "hit": "h",
    "singles": "b1", "single": "b1",
    "doubles": "b2", "double": "b2", "2b": "b2",
    "triples": "b3", "triple": "b3", "3b": "b3",
    "home runs": "hr", "home run": "hr", "homers": "hr", "homer": "hr",
    "dingers": "hr", "dinger": "hr", "hrs": "hr", "hr": "hr",
    "runs batted in": "rbi", "run batted in": "rbi", "rbis": "rbi", "rbi": "rbi",
    "runs scored": "r", "runs": "r", "run": "r",
    "walks": "bb", "walk": "bb", "base on balls": "bb", "bb": "bb",
    "strikeouts": "so", "strikeout": "so", "k's": "so", "ks": "so", "so": "so",
    "stolen bases": "sb", "stolen base": "sb", "sb": "sb",
    # batting rate stats
    # NOTE: bare "average"/"avg" is deliberately NOT mapped here — it would
    # out-rank shorter stat abbreviations (e.g. "obp", "slg") in questions
    # like "what's the team average OBP?" since it's matched by phrase
    # length. See match_stat()'s fallback for the bare-"average" case.
    "batting average": "avg",
    "on-base percentage": "obp", "on base percentage": "obp",
    "on-base %": "obp", "on base %": "obp", "on-base": "obp", "on base": "obp", "obp": "obp",
    "slugging percentage": "slg", "slugging %": "slg", "slugging": "slg", "slg": "slg",
    "ops": "ops",
    "isolated power": "iso", "iso": "iso",
    "strikeout rate": "k_pct", "strikeout percentage": "k_pct", "k%": "k_pct", "k rate": "k_pct",
    "walk rate": "bb_pct", "walk percentage": "bb_pct", "bb%": "bb_pct",
    "walks per strikeout": "bb_per_k", "bb/k": "bb_per_k",
    # defense
    "putouts": "putouts", "putout": "putouts", "po": "putouts",
    "assists": "assists", "assist": "assists",
    "errors": "errors", "error": "errors",
    "double plays": "double_plays", "double play": "double_plays", "dp": "double_plays",
    "fielding percentage": "fpct", "fielding %": "fpct", "fpct": "fpct",
    "error rate": "error_rate",
    "range factor": "range_factor",
    "fielding chances": "def_tc", "total chances": "def_tc", "chances": "def_tc", "tc": "def_tc",
    "defensive innings": "def_innings", "innings fielded": "def_innings",
    "primary position": "primary_position", "position": "primary_position",
    # everyday batting phrasings
    "strikes out": "k_pct", "strike out": "k_pct", "struck out": "k_pct", "striking out": "k_pct",
    "gets on base": "obp", "get on base": "obp", "reaches base": "obp", "reach base": "obp",
    "hits for power": "slg", "power hitter": "slg", "most power": "slg",
    "steals": "sb", "stolen": "sb",
    "innings played": "def_innings", "innings": "def_innings",
}

# Pitching-only stat words: safe to match in any question.
PITCHING_ONLY_ALIASES: dict[str, str] = {
    "innings pitched": "p_ip", "ip": "p_ip",
    "pitch count": "p_np", "pitches thrown": "p_np", "number of pitches": "p_np", "pitches": "p_np",
    "pitches per inning": "pitches_per_inn",
    "batters faced": "p_bf", "bf": "p_bf",
    "games pitched": "p_gp",
    "strike percentage": "strike_pct", "strike %": "strike_pct", "strike rate": "strike_pct",
    "s%": "strike_pct", "throws strikes": "strike_pct", "throw strikes": "strike_pct",
    "first pitch strikes": "fps_pct", "first-pitch strikes": "fps_pct", "first pitch strike": "fps_pct",
    "first-pitch strike": "fps_pct", "first-pitch strike %": "fps_pct", "fps%": "fps_pct", "fps": "fps_pct",
    "swing and miss": "swm_pct", "swing-and-miss": "swm_pct", "swings and misses": "swm_pct",
    "whiffs": "swm_pct", "whiff": "swm_pct", "sm%": "swm_pct",
    "walks per inning": "bb_per_inn", "bb per inning": "bb_per_inn", "bb/inn": "bb_per_inn",
    "strikeouts per walk": "k_per_bb", "k/bb": "k_per_bb",
    "opponents batting average": "opp_avg", "opponent batting average": "opp_avg",
    "batting average against": "opp_avg", "average against": "opp_avg", "baa": "opp_avg", "opp avg": "opp_avg",
    "earned run average": "era", "era": "era", "whip": "whip",
    "top speed": "top_mph", "fastest pitch": "top_mph", "velocity": "top_mph", "mph": "top_mph",
    "throws the hardest": "top_mph", "throws hardest": "top_mph", "throw the hardest": "top_mph",
    "hardest thrower": "top_mph", "pitch speed": "top_mph",
    "wins": "p_w", "losses": "p_l", "saves": "p_sv", "wild pitches": "p_wp", "wild pitch": "p_wp",
    "ground balls": "gb_pct", "ground ball": "gb_pct", "groundball": "gb_pct", "gb%": "gb_pct",
}

# Words that mean one thing for a batter and another for a pitcher. Only
# used when the question is about pitching (see is_pitching_question()).
PITCHING_CONTEXT_ALIASES: dict[str, str] = {
    "strikeouts": "p_so", "strikeout": "p_so", "k's": "p_so", "ks": "p_so", "so": "p_so",
    "strikes out": "p_so", "strike out": "p_so", "struck out": "p_so", "striking out": "p_so",
    "walks": "p_bb", "walk": "p_bb", "walked": "p_bb", "base on balls": "p_bb", "bb": "p_bb",
    "hits": "p_h", "hit": "p_h",
    "home runs": "p_hr", "home run": "p_hr", "homers": "p_hr", "hr": "p_hr",
    "runs": "p_r", "run": "p_r", "earned runs": "p_er", "earned run": "p_er",
    "hit by pitch": "p_hbp", "hit batters": "p_hbp", "hbp": "p_hbp",
    "strikes": "strike_pct", "walk rate": "bb_per_inn",
    "games": "p_gp", "appearances": "p_gp",
    "innings": "p_ip", "fastest": "top_mph", "speed": "top_mph", "hardest": "top_mph",
}

PITCHING_CONTEXT_RE = re.compile(
    r"\b(pitch\w*|threw|throw\w*|mound|allowed|gave up|give up|gives up"
    r"|opponents?|on the hill|hurler|bullpen|batters faced|hit batters"
    r"|(?:struck|strikes?|striking) out (?:the )?(?:most |more |fewest |least )?batters"
    r"|walk(?:ed|s)? (?:the )?(?:most |more |fewest |least )?batters)\b"
)

LABELS: dict[str, str] = {
    "pa": "plate appearances", "ab": "at-bats", "h": "hits", "b1": "singles",
    "b2": "doubles", "b3": "triples", "hr": "home runs", "rbi": "RBI", "r": "runs",
    "bb": "walks", "so": "strikeouts", "sb": "stolen bases",
    "avg": "batting average", "obp": "on-base %", "slg": "slugging %", "ops": "OPS",
    "iso": "isolated power", "k_pct": "strikeout rate", "bb_pct": "walk rate",
    "bb_per_k": "walks per strikeout",
    "putouts": "putouts", "assists": "assists", "errors": "errors",
    "double_plays": "double plays", "fpct": "fielding %", "error_rate": "error rate",
    "range_factor": "range factor", "def_tc": "fielding chances",
    "def_innings": "defensive innings", "primary_position": "primary position",
}

COUNTING_STATS = {
    "pa", "ab", "h", "b1", "b2", "b3", "hr", "rbi", "r", "bb", "so", "sb",
    "putouts", "assists", "errors", "double_plays", "def_tc",
}
DEFENSE_COUNTING_STATS = {"putouts", "assists", "errors", "double_plays", "def_tc"}
RATE_3F_STATS = {"avg", "obp", "slg", "ops", "iso", "fpct"}  # .XXX notation
PCT_1F_STATS = {"k_pct", "bb_pct", "error_rate"}  # NN.N%
PITCHING_COUNTING_STATS = {
    "p_np", "p_bf", "p_gp", "p_w", "p_l", "p_sv", "p_h", "p_r", "p_er", "p_bb", "p_so",
    "p_hbp", "p_wp", "p_hr",
}
PITCHING_RATE_STATS = {
    "strike_pct", "fps_pct", "swm_pct", "gb_pct", "bb_per_inn", "k_per_bb", "opp_avg",
    "era", "whip", "pitches_per_inn",
}
PITCHING_STATS = PITCHING_COUNTING_STATS | PITCHING_RATE_STATS | {"p_ip", "top_mph"}
COUNTING_STATS = COUNTING_STATS | PITCHING_COUNTING_STATS
RATE_3F_STATS = RATE_3F_STATS | {"opp_avg"}
PCT_1F_STATS = PCT_1F_STATS | {"strike_pct", "fps_pct", "swm_pct", "gb_pct"}
DEC_2F_STATS = {"bb_per_k", "bb_per_inn", "k_per_bb", "era", "whip"}

LABELS.update({
    "p_ip": "innings pitched", "p_np": "pitches thrown", "p_bf": "batters faced",
    "p_gp": "games pitched", "p_w": "pitching wins", "p_l": "pitching losses", "p_sv": "saves",
    "p_h": "hits allowed", "p_r": "runs allowed", "p_er": "earned runs allowed",
    "p_bb": "walks allowed (pitching)", "p_so": "strikeouts (pitching)", "p_hbp": "batters hit by a pitch",
    "p_wp": "wild pitches", "p_hr": "home runs allowed",
    "strike_pct": "strike %", "fps_pct": "first-pitch strike %", "swm_pct": "swing-and-miss %",
    "gb_pct": "ground-ball %", "bb_per_inn": "walks per inning", "k_per_bb": "strikeouts per walk",
    "opp_avg": "opponents' batting average", "era": "ERA", "whip": "WHIP",
    "pitches_per_inn": "pitches per inning", "top_mph": "top recorded speed",
})

LOWER_IS_BETTER = {
    "so", "k_pct", "errors", "error_rate",
    "p_h", "p_r", "p_er", "p_bb", "p_hbp", "p_wp", "p_hr", "p_l",
    "bb_per_inn", "opp_avg", "era", "whip", "pitches_per_inn",
}
NON_NUMERIC_STATS = {"primary_position"}

MOST_WORDS = {"most", "highest", "leads", "leader", "leading", "top", "greatest", "max", "maximum"}
LEAST_WORDS = {"least", "lowest", "fewest", "bottom", "min", "minimum"}
BEST_WORDS = {"best"}
WORST_WORDS = {"worst"}
TOTAL_WORDS = {"total", "sum", "combined", "altogether", "in all"}
MEAN_WORDS = {"average", "avg", "mean", "per player", "typical", "typically"}
TEAM_WORDS = {"team", "roster", "everyone", "squad", "whole team"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def is_pitching_question(question: str) -> bool:
    return bool(PITCHING_CONTEXT_RE.search(question.lower()))


def _normalize(question: str) -> str:
    # "opponents' average" / "Maya's" -> drop apostrophes that aren't part of a
    # stat alias like "k's", and turn "first-pitch"-style hyphens into spaces
    # only where an alias doesn't need them.
    q = question.lower().replace("’", "'")
    q = re.sub(r"(?<=s)'(?=\s|$)", "", q)
    return q


def match_stat(question: str) -> Optional[str]:
    q = _normalize(question)
    aliases = dict(STAT_ALIASES)
    aliases.update(PITCHING_ONLY_ALIASES)
    if is_pitching_question(q):
        aliases.update(PITCHING_CONTEXT_ALIASES)
    for alias, key in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"[^a-z0-9 ]", alias):
            # aliases with symbols (k%, bb/k, ...) can't use \b cleanly
            if alias in q:
                return key
        else:
            if re.search(r"\b" + re.escape(alias) + r"\b", q):
                return key
    # Bare "average"/"avg" isn't in STAT_ALIASES (see the comment there) so
    # it doesn't out-rank shorter stat abbreviations elsewhere in the same
    # question — but if nothing else matched, it does mean batting average.
    if re.search(r"\b(average|avg)\b", q):
        return "avg"
    return None


def _tokenize(s: str) -> list[str]:
    # Deliberately excludes apostrophes: "Luke's" should tokenize to "luke"
    # (matching the player's first name) plus a stray "s", not "luke's".
    return re.findall(r"[a-zA-Z]+", s.lower())


def find_players(question: str, players: list[PlayerStats]) -> list[PlayerStats]:
    """Finds which roster player(s) a question is talking about. Tries a
    full-name substring match first (unambiguous), then falls back to
    matching a bare first/last-name token — which can legitimately return
    more than one player if two teammates share a first name."""
    q = question.lower()
    full_matches = [p for p in players if p.name.lower() in q]
    if full_matches:
        full_matches.sort(key=lambda p: -len(p.name))
        return full_matches

    tokens = set(_tokenize(question))
    tok_matches = []
    for p in players:
        parts = p.name.lower().split()
        if not parts:
            continue
        if parts[0] in tokens or parts[-1] in tokens:
            tok_matches.append(p)
    return tok_matches


def _extract_top_n(q: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", q) or re.search(r"\bfirst\s+(\d+)\b", q)
    return int(m.group(1)) if m else None


def _wants_team_aggregate(q: str) -> bool:
    return any(w in q for w in TEAM_WORDS)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_stat_value(stat: str, value) -> str:
    if value is None:
        return "—"
    if stat in RATE_3F_STATS:
        return gr.fmt3(value)
    if stat in PCT_1F_STATS:
        return f"{value:.1f}%"
    if stat in DEC_2F_STATS:
        return f"{value:.2f}"
    if stat == "pitches_per_inn":
        return f"{value:.1f}"
    if stat == "p_ip":
        return gr.fmt_innings(value)
    if stat == "top_mph":
        return f"{int(round(value))} mph"
    if stat == "range_factor":
        return f"{value:.1f}"
    if stat == "def_innings":
        return gr.fmt_innings(value)
    if stat in COUNTING_STATS:
        return str(int(round(value)))
    return str(value)


def _pitched(p: PlayerStats) -> bool:
    return p.has_pitching_data and p.p_bf > 0


def _pool_for_stat(stat: str, all_players: list[PlayerStats], regulars: list[PlayerStats],
                    defensive_pool: list[PlayerStats], config: dict) -> tuple[list[PlayerStats], str]:
    if stat in PITCHING_RATE_STATS:
        pool = [p for p in all_players if _pitched(p) and not p.below_pitching_floor]
        note = (f" (Among the {len(pool)} pitchers who faced {gr.PITCHING_FLOOR_BF}+ batters, "
                f"so the numbers are meaningful.)")
        return pool, note
    if stat in PITCHING_STATS:
        return [p for p in all_players if _pitched(p)], ""
    if stat in DEFENSE_COUNTING_STATS:
        return [p for p in all_players if p.has_fielding_data], ""
    if stat in COUNTING_STATS:
        return all_players, ""
    if stat in {"fpct", "error_rate", "range_factor"}:
        note = f" (Among players with enough fielding chances to qualify — {gr.DEFENSE_RATE_STAT_FLOOR}+ total chances.)"
        return defensive_pool, note
    if stat in {"def_innings", "primary_position"}:
        return [p for p in all_players if p.has_fielding_data], ""
    min_pa = config.get("min_pa_regular", 15)
    note = f" (Among the {len(regulars)} regulars this stretch — {int(min_pa)}+ PA — so the numbers are meaningful.)"
    return regulars, note


# ---------------------------------------------------------------------------
# Answer builders
# ---------------------------------------------------------------------------

def _answer_lookup(stat: str, player: PlayerStats) -> str:
    value = getattr(player, stat, None)
    label = LABELS.get(stat, stat)
    if stat in PITCHING_STATS and not _pitched(player):
        return f"{player.name} hasn't pitched this stretch, so there's no {label}."
    if stat in PITCHING_RATE_STATS and player.below_pitching_floor and value is not None:
        return (f"{player.name}'s {label} is {format_stat_value(stat, value)}, but that's from only "
                f"{int(player.p_bf)} batters faced, too few to read much into.")
    if value is None:
        return f"{player.name} doesn't have a recorded {label}."
    return f"{player.name}'s {label} is {format_stat_value(stat, value)}."


def _answer_comparison(stat: str, players: list[PlayerStats]) -> str:
    a, b = players[0], players[1]
    va, vb = getattr(a, stat, None), getattr(b, stat, None)
    label = LABELS.get(stat, stat)
    if stat in PITCHING_STATS:
        missing = [p.name for p in (a, b) if not _pitched(p)]
        if missing:
            return f"{' and '.join(missing)} {'hasn' if len(missing) == 1 else 'haven'}'t pitched this stretch, so I can't compare {label}."
    if va is None or vb is None:
        return f"I don't have {label} for both {a.name} and {b.name}."
    fa, fb = format_stat_value(stat, va), format_stat_value(stat, vb)
    if va == vb:
        return f"{a.name} and {b.name} are tied in {label}, both at {fa}."
    lower_better = stat in LOWER_IS_BETTER
    a_ahead = (va < vb) if lower_better else (va > vb)
    leader = a if a_ahead else b
    comp_word = "lower" if lower_better else "higher"
    return f"{a.name}: {fa} {label} · {b.name}: {fb} {label}. {leader.name} has the {comp_word} number."


def _answer_leaderboard(stat: str, question: str, all_players: list[PlayerStats],
                         regulars: list[PlayerStats], defensive_pool: list[PlayerStats],
                         config: dict) -> str:
    q = question.lower()
    label = LABELS.get(stat, stat)
    n = _extract_top_n(q)
    pool, pool_note = _pool_for_stat(stat, all_players, regulars, defensive_pool, config)

    scored = [(p, getattr(p, stat, None)) for p in pool]
    scored = [(p, v) for p, v in scored if v is not None]
    if not scored:
        if stat in PITCHING_STATS:
            if not any(_pitched(p) for p in all_players):
                return "This export doesn't include any pitching stats."
            if stat in PITCHING_RATE_STATS:
                return (f"Nobody has faced {gr.PITCHING_FLOOR_BF}+ batters yet, so {label} would just be noise. "
                        f"Try a total instead, like strikeouts or innings pitched.")
        return f"No players in this roster have usable {label} data.{pool_note}"

    lower_better = stat in LOWER_IS_BETTER
    if any(w in q for w in LEAST_WORDS):
        descending = False
    elif any(w in q for w in WORST_WORDS):
        descending = lower_better
    elif any(w in q for w in BEST_WORDS):
        descending = not lower_better
    else:
        descending = True  # "most" / "highest" / "leads" / unspecified

    scored.sort(key=lambda t: t[1], reverse=descending)

    if stat in COUNTING_STATS:
        dword = "most" if descending else "fewest"
    else:
        dword = "highest" if descending else "lowest"

    if n is None:
        top_value = scored[0][1]
        tied = [p for p, v in scored if v == top_value]
        if len(tied) == 1:
            return f"{tied[0].name} has the {dword} {label}: {format_stat_value(stat, top_value)}.{pool_note}"
        names = ", ".join(p.name for p in tied)
        return f"It's a tie for {dword} {label} at {format_stat_value(stat, top_value)}: {names}.{pool_note}"

    n = max(1, min(n, len(scored)))
    lines = [f"{i + 1}. {p.name} — {format_stat_value(stat, v)}" for i, (p, v) in enumerate(scored[:n])]
    return f"Top {n} in {label}:\n" + "\n".join(lines) + pool_note


def _answer_team_aggregate(stat: str, question: str, all_players: list[PlayerStats],
                            regulars: list[PlayerStats], defensive_pool: list[PlayerStats],
                            config: dict) -> str:
    q = question.lower()
    label = LABELS.get(stat, stat)
    if stat in NON_NUMERIC_STATS:
        return "That's not a number I can total or average across the team."
    pool, note = _pool_for_stat(stat, all_players, regulars, defensive_pool, config)
    values = [getattr(p, stat, None) for p in pool]
    values = [v for v in values if v is not None]
    if not values:
        return f"No usable {label} data on this roster.{note}"

    wants_sum = any(w in q for w in TOTAL_WORDS) or (
        stat in COUNTING_STATS and not any(w in q for w in MEAN_WORDS)
    )
    if wants_sum and stat in COUNTING_STATS:
        total = sum(values)
        return f"The team's total {label} is {format_stat_value(stat, total)}, across {len(values)} players.{note}"

    mean_v = sum(values) / len(values)
    return f"The team average {label} is {format_stat_value(stat, mean_v)}, across {len(values)} players.{note}"


def _answer_roster_meta(question: str, all_players: list[PlayerStats], regulars: list[PlayerStats]) -> Optional[str]:
    q = question.lower()
    if re.search(r"how many (players|kids|people)", q) or "roster size" in q:
        return f"There are {len(all_players)} players on this roster."
    if "regular" in q:
        if "how many" in q:
            return f"{len(regulars)} of {len(all_players)} players are regulars this stretch."
        if any(w in q for w in ("who", "list", "which")):
            names = ", ".join(p.name for p in regulars) if regulars else "nobody yet — nobody has enough plate appearances this stretch"
            return f"Regulars this stretch: {names}."
    if re.search(r"who(\'?s| is)? on (the|this) (team|roster)", q) or "list the roster" in q or "who plays" in q or "who is on the roster" in q:
        names = ", ".join(p.name for p in all_players)
        return f"Roster ({len(all_players)}): {names}."
    return None


def _answer_pitching_summary(p: PlayerStats) -> str:
    if not _pitched(p):
        return f"{p.name} hasn't pitched this stretch."
    parts = [f"{p.name} pitched {gr.fmt_innings(p.p_ip)} innings and faced {int(p.p_bf)} batters"]
    if p.p_np:
        parts[0] += f", throwing {int(p.p_np)} pitches"
    bits = [f"{int(p.p_so)} strikeouts", f"{int(p.p_bb)} walks"]
    if p.strike_pct is not None:
        bits.append(f"{p.strike_pct:.1f}% strikes")
    if p.top_mph:
        bits.append(f"top speed {int(round(p.top_mph))} mph")
    note = "" if not p.below_pitching_floor else " That's a small sample, so read the rates loosely."
    return parts[0] + ": " + ", ".join(bits) + "." + note


def _answer_pitching_unclear(all_players: list[PlayerStats]) -> str:
    pitchers = [p for p in all_players if _pitched(p)]
    if not pitchers:
        return "This export doesn't include any pitching stats."
    names = ", ".join(p.name for p in sorted(pitchers, key=lambda p: -p.p_ip))
    return (f"Which pitching number? There's no single \"best pitcher\" stat. Try strike %, first-pitch "
            f"strikes, walks per inning, strikeouts, innings or top speed, e.g. \"who throws the most "
            f"strikes?\" or \"who has the most strikeouts pitching?\" "
            f"{len(pitchers)} players pitched this stretch: {names}.")


FALLBACK = (
    "I couldn't tell which stat you're asking about. Try naming one directly — things like batting "
    "average, OBP, slugging, OPS, strikeout rate, walk rate, home runs, RBI, hits, fielding %, range "
    "factor, or error rate all work. For example: \"who has the most home runs?\", \"what's Luke's "
    "OBP?\", \"Luke vs Owen, who has more RBI?\", or \"what's the team's total triples?\" For pitching, try "
    "\"who throws the most strikes?\" or \"who has the most strikeouts pitching?\""
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def answer_question(question: str, all_players: list[PlayerStats], regulars: list[PlayerStats],
                     defensive_pool: list[PlayerStats], config: dict) -> str:
    question = (question or "").strip()
    if not question:
        return "Ask me something about this roster — e.g. \"who has the most home runs?\" or \"what's Luke's OBP?\""
    if not all_players:
        return "There's no roster loaded yet — upload and load a CSV first."

    stat = match_stat(question)
    players_mentioned = find_players(question, all_players)

    # A pitching question must get a pitching answer: if the words only
    # matched a batting/fielding stat (or nothing), ask which pitching
    # number they mean instead of quietly answering a different question.
    if is_pitching_question(question) and stat is None and len(players_mentioned) == 1:
        return _answer_pitching_summary(players_mentioned[0])
    if is_pitching_question(question) and (stat is None or stat not in PITCHING_STATS):
        if stat is None or stat in LABELS and stat not in {"avg", "obp", "slg", "ops", "iso"}:
            if not (stat is None and _answer_roster_meta(question, all_players, regulars)):
                return _answer_pitching_unclear(all_players)

    # 3+ matches almost always means one ambiguous bare first name shared by
    # several teammates (e.g. three "Luke"s), not three players genuinely
    # named in one question — ask which one rather than silently guessing
    # and dropping someone.
    if len(players_mentioned) >= 3:
        names = ", ".join(p.name for p in players_mentioned[:6])
        return f"A few players match that — did you mean one of: {names}? Try the full name."

    if stat and len(players_mentioned) == 2:
        return _answer_comparison(stat, players_mentioned)

    if stat and len(players_mentioned) == 1:
        return _answer_lookup(stat, players_mentioned[0])

    if stat and len(players_mentioned) == 0:
        if _wants_team_aggregate(question):
            return _answer_team_aggregate(stat, question, all_players, regulars, defensive_pool, config)
        return _answer_leaderboard(stat, question, all_players, regulars, defensive_pool, config)

    meta = _answer_roster_meta(question, all_players, regulars)
    if meta:
        return meta

    if not stat and len(players_mentioned) == 1:
        return f"What do you want to know about {players_mentioned[0].name}? Name a stat, like batting average, OBP, or home runs."

    return FALLBACK
