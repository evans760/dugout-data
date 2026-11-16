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
}

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
LOWER_IS_BETTER = {"so", "k_pct", "errors", "error_rate"}
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

def match_stat(question: str) -> Optional[str]:
    q = question.lower()
    for alias, key in sorted(STAT_ALIASES.items(), key=lambda kv: -len(kv[0])):
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
    if stat == "bb_per_k":
        return f"{value:.2f}"
    if stat == "range_factor":
        return f"{value:.1f}"
    if stat == "def_innings":
        return gr.fmt_innings(value)
    if stat in COUNTING_STATS:
        return str(int(round(value)))
    return str(value)


def _pool_for_stat(stat: str, all_players: list[PlayerStats], regulars: list[PlayerStats],
                    defensive_pool: list[PlayerStats], config: dict) -> tuple[list[PlayerStats], str]:
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
    if value is None:
        return f"{player.name} doesn't have a recorded {label}."
    return f"{player.name}'s {label} is {format_stat_value(stat, value)}."


def _answer_comparison(stat: str, players: list[PlayerStats]) -> str:
    a, b = players[0], players[1]
    va, vb = getattr(a, stat, None), getattr(b, stat, None)
    label = LABELS.get(stat, stat)
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


FALLBACK = (
    "I couldn't tell which stat you're asking about. Try naming one directly — things like batting "
    "average, OBP, slugging, OPS, strikeout rate, walk rate, home runs, RBI, hits, fielding %, range "
    "factor, or error rate all work. For example: \"who has the most home runs?\", \"what's Luke's "
    "OBP?\", \"Luke vs Owen, who has more RBI?\", or \"what's the team's total triples?\""
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
