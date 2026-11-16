#!/usr/bin/env python3
"""
generate_reports.py — GameChanger CSV -> per-player batting reports.

Usage:
    python generate_reports.py roster.csv
    python generate_reports.py roster.csv --mode parent --out output/
    python generate_reports.py roster.csv --mode all --min-pa 20 --yes

Input: a GameChanger *season stats export* CSV, downloaded by a coach/staff
member from Team -> Stats -> Export Stats. This script never talks to the
network and never scrapes GameChanger. If you don't have a CSV, this tool
can't help you — that's on purpose (see README.md).

Output: one self-contained HTML file per player per requested mode, written
to the output directory. Nothing is written until every parent/player file
has passed the redaction pre-flight check (see check_redaction()).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined


class ReportBuildError(Exception):
    """Raised instead of calling sys.exit() directly from library functions
    (load_and_aggregate, render_all), so this module is safe to import into
    a long-running process (e.g. the web app) without a bad CSV or a
    redaction failure killing the whole server. The CLI entry point (main())
    catches this and converts it to sys.exit(), preserving the original
    command-line behavior exactly."""
    pass


# --------------------------------------------------------------------------
# CONFIG — edit these defaults, or override at the command line.
# --------------------------------------------------------------------------

CONFIG = {
    "team_name": "Your Team Name",
    # Season label as GameChanger displays it (e.g. "Spring 2026"). Used only
    # to sanity-check against the actual game-date span in the export, if a
    # date column is present. Leave as None to skip that check entirely.
    "season_label": None,
    # PA threshold for a player to count as a "regular" for the purposes of
    # team-average / percentile comparisons and the team-OPS audit. This is
    # a playing-time floor, not a statistical-reliability floor (see
    # RATE_STAT_FLOOR below for that). Youth rosters vary a lot — 15 PA is a
    # low bar on purpose so a short-bench kid isn't invisible in coach mode,
    # but raise it for a league with a longer season.
    "min_pa_regular": 15,
    # Which report(s) to generate when --mode isn't passed on the CLI.
    "output_mode": "all",  # one of: parent, player, coach, all
}

# Below this many plate appearances, a player's own parent/player report
# skips rate-stat interpretation (tiers, percentiles, bars/dots, the
# development focus) entirely and shows counting totals only. This is a
# fixed statistical floor, independent of min_pa_regular above.
RATE_STAT_FLOOR = 40

# Below this many total fielding chances (putouts + assists + errors), a
# player's defensive numbers are skipped entirely rather than shown as an
# unreliable card — calibrated against real GameChanger exports, where
# fielding participation varies far more than plate appearances do (many
# rostered players show 0 recorded chances at all).
DEFENSE_RATE_STAT_FLOOR = 10

# Below this many batters faced (roughly 8 innings at youth level), a
# pitcher gets one encouraging line instead of a full pitching section.
PITCHING_FLOOR_BF = 30

# Reliability thresholds. The "reliable" number below for each stat is the
# PA/AB it takes for that stat to stop bouncing around much between two
# random halves of the same sample — professional-baseball research puts
# those points at roughly 60 PA (K%), 120 PA (BB%), 320 AB (SLG), 460 PA
# (OBP), and 910 AB (AVG). Those numbers describe MLB-length seasons
# (600+ PA); no kid in a youth rec league ever reaches 910 at-bats in a
# season, so used unscaled, AVG/OBP/SLG would read "still settling" for
# every player, every season, forever — not because the tool is wrong, but
# because the bar was calibrated for a sample size nobody here has. What's
# below is calibrated instead against the actual distribution of PA/AB seen
# across real youth rosters (a genuinely full, high-attendance season tops
# out around 130-155 PA / 100-135 AB for a regular): "reliable" now marks
# the top of what a real season can produce, "partial" the middle of the
# pack, so both tiers stay reachable while still preserving the original
# research's relative ordering (K% needs the least data, AVG the most).
# This is an intentionally different, lower bar than the professional
# research's — see RELIABILITY_EXPLANATIONS below, which says so.
RELIABILITY_THRESHOLDS = {
    "k_pct": ("PA", 45),
    "bb_pct": ("PA", 70),
    "slg": ("AB", 60),
    "obp": ("PA", 90),
    "avg": ("AB", 100),
    # Defensive thresholds, calibrated the same way against real fielding
    # data: fielding %/error rate against total chances (TC), since that's
    # their natural sample-size unit; range factor against defensive
    # innings, since it's inherently a per-innings rate.
    "fpct": ("TC", 30),
    "error_rate": ("TC", 30),
    "range_factor": ("DEF_INN", 60),
    # Pitching, calibrated against real youth exports (a team's busiest
    # pitcher faces ~250-320 batters and throws ~1,000 pitches a season):
    # rates measured per pitch use pitch count (NP), the rest batters faced (BF).
    "strike_pct": ("NP", 300),
    "fps_pct": ("BF", 120),
    "swm_pct": ("NP", 400),
    "bb_per_inn": ("BF", 150),
    "k_per_bb": ("BF", 150),
    "opp_avg": ("BF", 150),
}

# Pitching is judged against simple coaching goals, not against teammates —
# a team usually has only 5-8 pitchers, too few for "above the middle" to
# mean much. These are common youth-coaching rules of thumb, shown in the
# report as "a common goal", never as an official standard.
#   goal = meeting it (green), near = close (yellow), past near = needs work (red)
PITCHING_GOALS = {
    "strike_pct": {"goal": 60.0, "near": 50.0, "higher_is_better": True,
                   "goal_text": "60% or more of pitches for strikes"},
    "fps_pct": {"goal": 60.0, "near": 50.0, "higher_is_better": True,
                "goal_text": "a first-pitch strike to 60% or more of batters"},
    "bb_per_inn": {"goal": 1.0, "near": 1.5, "higher_is_better": False,
                   "goal_text": "1 walk or fewer per inning"},
    "k_per_bb": {"goal": 2.0, "near": 1.0, "higher_is_better": True,
                 "goal_text": "2 or more strikeouts for every walk"},
}
GOAL_TIER_LABELS = {
    "goal_met": "meeting the goal",
    "near_goal": "close to the goal",
    "below_goal": "working toward the goal",
}
# One-line pitching goal, checked in this order: the first stat that's
# furthest from its goal (red before yellow) becomes the next goal.
PITCHING_GOAL_PRIORITY = ["strike_pct", "fps_pct", "bb_per_inn", "k_per_bb"]
PITCHING_NEXT_GOAL_TEXT = {
    "strike_pct": "Throw more strikes: aim for 6 of every 10 pitches.",
    "fps_pct": "Get ahead early: aim for a first-pitch strike to 6 of every 10 batters.",
    "bb_per_inn": "Cut down on walks: aim for 1 walk or fewer per inning.",
    "k_per_bb": "Strike out more batters than you walk: aim for 2 strikeouts for every walk.",
}


TEAM_OPS_STRONG_CUTOFF = 0.900  # at/above this, the bottom line explains it as a strong roster
# How parent reports compare a hitter to the team:
#   "dots" — every regular as a dot, this player highlighted (default)
#   "bars" — this player's number against a team-average marker
#   "auto" — bars, unless team OPS >= TEAM_OPS_STRONG_CUTOFF, then dots
COMPARISON_VIEW = "dots"

# Which sections/stat-cards a parent/player report can show. The CLI always
# uses every key True (unchanged v1 behavior); the web app's free-text
# "what stats do you want to see" prompt can turn individual keys off. This
# never adds anything not already in the template — it only hides modules
# that are otherwise shown by default.
VISIBLE_KEYS = [
    "trend", "focus", "k_pct", "bb_pct", "bb_per_k",
    "avg", "obp", "slg", "ops", "iso", "where_he_stands", "all_numbers",
    "defense", "pitching",
]
DEFAULT_VISIBLE = {k: True for k in VISIBLE_KEYS}

TIER_LABELS = {
    # These plug into a sentence of the shape "The player is {tier_label} among the
    # team's regular hitters" (see templates) — keep them grammatical in
    # that slot if you edit them.
    "leading": "leading",
    "above_middle": "above the middle",
    "around_middle": "right around the middle",
    "below_middle": "below the middle",
    "still_finding_it": "still finding it",
}

# Single source of truth for how a reliability tier is *displayed*. The
# internal key stays "noisy" (it's used in weaker()/build_comparisons and in
# coach-mode's raw table), but every parent/player-facing badge and the
# reliability legend must render through this dict so the word a parent
# reads in the legend always matches the word on every badge below it.
RELIABILITY_DISPLAY_LABELS = {
    "reliable": "RELIABLE",
    "partial": "PARTIAL",
    "noisy": "STILL SETTLING",
}

# Plain-language explanations of what a reliability tier actually means,
# written for a reader with zero statistics background. Shown once (in a
# compact legend) rather than repeated on every card.
RELIABILITY_EXPLANATIONS = {
    "reliable": "About as many chances as a full, well-attended youth season provides — enough to call "
                "this a real pattern, not a lucky or unlucky stretch. (This is a youth-season bar, not "
                "the much bigger sample pro-baseball research uses to call a stat fully “stabilized” "
                "— nobody at this level ever reaches that.)",
    "partial": "Getting there, but there's still some randomness mixed in. "
               "Treat it as a meaningful signal, not the final word yet.",
    "noisy": "Too few chances so far for this number to mean much on its own — "
             "one or two more games could swing it a lot. Read it as “what "
             "happened,” not “who the player is.”",
}

# --------------------------------------------------------------------------
# Column detection — GameChanger export headers vary by sport/season/export
# vintage. We never hardcode a header name as gospel: we propose a mapping,
# print it, and ask for confirmation (skippable with --yes).
# --------------------------------------------------------------------------

COLUMN_CANDIDATES = {
    "name": ["player", "name", "player name", "batter"],
    # Some GameChanger exports (the "Advanced/Full Stats" export in
    # particular) have no single name column at all — just separate Last
    # and First columns. detect_columns() still reports these individually;
    # load_and_aggregate() synthesizes a combined "name" field from them
    # when "name" itself isn't found. See missing_required_columns().
    "first": ["first", "first name", "firstname"],
    "last": ["last", "last name", "lastname"],
    "number": ["#", "number", "jersey", "no.", "no"],
    "gp": ["gp", "games played", "g"],
    "date": ["date", "game date"],
    "opponent": ["opponent", "opp", "vs"],
    "pa": ["pa", "plate appearances"],
    "ab": ["ab", "at bats", "at-bats"],
    "r": ["r", "runs"],
    "h": ["h", "hits"],
    "2b": ["2b", "doubles"],
    "3b": ["3b", "triples"],
    "hr": ["hr", "home runs"],
    "rbi": ["rbi", "rbis", "runs batted in"],
    "bb": ["bb", "walks", "base on balls"],
    "so": ["so", "strikeouts", "k"],
    "sb": ["sb", "stolen bases"],
    "cs": ["cs", "caught stealing"],
    "hbp": ["hbp", "hit by pitch"],
    "sf": ["sf", "sac flies", "sacrifice flies"],
    "sh": ["sh", "sac", "sac bunts", "sacrifice hits", "sacrifice bunts"],
    "avg": ["avg", "ba", "batting average"],
    "obp": ["obp", "on base %", "on-base percentage", "on base pct"],
    "slg": ["slg", "slugging %", "slugging percentage", "slugging pct"],
    "ops": ["ops"],
}

# Columns that indicate a pitching export, not a batting one. We ignore
# these for v1 (batting-only) but they help confirm "SO" vs "K" ambiguity —
# if a sheet has ERA/IP, "K" almost certainly means pitching strikeouts, and
# we should be more cautious about mapping it to batting SO.
PITCHING_HINT_COLUMNS = ["ip", "er", "era", "bf", "whip"]

REQUIRED_CANONICAL = ["name", "ab", "h", "bb", "so", "2b", "3b", "hr"]

# --------------------------------------------------------------------------
# Fielding columns — GameChanger's "Advanced/Full Stats" export reuses the
# SAME header text across its Batting/Pitching/Fielding sections (a batting
# "1B" = singles; a fielding "1B" = innings played at first base; "SB",
# "CS", "PIK", "CI" all repeat too). Plain name-matching against the whole
# header (what detect_columns() does for batting) would silently grab the
# wrong section's column. These are resolved instead by where they sit
# relative to the "Fielding" marker in the section-label row above the real
# header — verified identical across every real export seen so far:
#   TC, A, PO, FPCT, E, DP, TP, INN, PB, SB, SBATT, CS, CS%, PIK, CI,
#   P, C, 1B, 2B, 3B, SS, LF, CF, RF, SF, Total
# INN/PB/SB/SBATT/CS/CS%/PIK/CI in that list are catcher-specific (innings
# *caught*, not total defensive innings) and aren't used here. "Total" —
# the last column — is the real sum of innings across every position, and
# is what's used as the defensive sample-size basis below.
FIELDING_COLUMN_CANDIDATES = {
    "putouts": ["po"],
    "assists": ["a"],
    "errors": ["e"],
    "double_plays": ["dp"],
    "def_innings": ["total"],
    "pos_p": ["p"],
    "pos_c": ["c"],
    "pos_1b": ["1b"],
    "pos_2b": ["2b"],
    "pos_3b": ["3b"],
    "pos_ss": ["ss"],
    "pos_lf": ["lf"],
    "pos_cf": ["cf"],
    "pos_rf": ["rf"],
    "pos_sf": ["sf"],
}

POSITION_LABELS = {
    "pos_p": "Pitcher", "pos_c": "Catcher", "pos_1b": "First Base", "pos_2b": "Second Base",
    "pos_3b": "Third Base", "pos_ss": "Shortstop", "pos_lf": "Left Field", "pos_cf": "Center Field",
    "pos_rf": "Right Field", "pos_sf": "Short Field",
}


def _find_section_range(section_row: list[str], name: str) -> Optional[tuple[int, Optional[int]]]:
    """section_row is the raw section-label row (row 1 of a GameChanger
    Advanced/Full Stats export: mostly empty, with a few group headers like
    "Batting"/"Pitching"/"Fielding"). Returns the (start, end) column-index
    range for the named section — end is None if it's the last section."""
    markers = [(i, v.strip()) for i, v in enumerate(section_row) if v.strip()]
    for idx, (start, marker_name) in enumerate(markers):
        if marker_name.lower() == name.lower():
            end = markers[idx + 1][0] if idx + 1 < len(markers) else None
            return (start, end)
    return None


def _resolve_column_in_range(headers: list[str], start: int, end: Optional[int],
                              candidates: list[str]) -> Optional[str]:
    """Finds a column in headers[start:end] whose (normalized) name matches
    one of `candidates`, then translates it to the name pandas will
    actually assign after its own duplicate-column mangling — the Nth
    occurrence (0-indexed) of an exact header string anywhere in the full
    file becomes "NAME" (N=0) or "NAME.N" (N>0), matching pandas' default
    `mangle_dupe_cols` behavior exactly, so this stays correct even though
    the plain name (e.g. "1B") is ambiguous across sections."""
    window = headers[start:end]
    lower_candidates = {c.lower() for c in candidates}
    for local_i, h in enumerate(window):
        if h.strip().lower() in lower_candidates:
            global_i = start + local_i
            occurrence = sum(1 for h2 in headers[:global_i] if h2.strip() == h.strip())
            return h if occurrence == 0 else f"{h}.{occurrence}"
    return None


def detect_fielding_columns(headers: list[str], section_row: Optional[list[str]]) -> dict[str, Optional[str]]:
    """Returns {canonical: actual_column_name_or_None} for the fielding
    fields in FIELDING_COLUMN_CANDIDATES. Returns all-None if this export
    has no section-label row (section_row is None/empty) or no recognizable
    "Fielding" marker — fielding stats are always optional, never required,
    so a file without this section just won't get a Defense section later,
    the same "skip it, don't guess" rule used everywhere else in this tool."""
    mapping: dict[str, Optional[str]] = {canon: None for canon in FIELDING_COLUMN_CANDIDATES}
    if not section_row:
        return mapping
    rng = _find_section_range(section_row, "Fielding")
    if not rng:
        return mapping
    start, end = rng
    for canon, candidates in FIELDING_COLUMN_CANDIDATES.items():
        mapping[canon] = _resolve_column_in_range(headers, start, end, candidates)
    return mapping

# --------------------------------------------------------------------------
# Pitching columns — resolved by position inside the "Pitching" section of
# the section-label row, exactly like fielding (H, BB, SO, HR, SB, CS, LD%,
# FB%, GB%, BABIP... all repeat between Batting and Pitching). All prefixed
# "p_" so they can never collide with a batting canonical name.
PITCH_TYPES = [
    ("fb", "Fastball"), ("cb", "Curveball"), ("ct", "Cutter"),
    ("sl", "Slider"), ("ch", "Changeup"), ("os", "Off-speed"),
]
PITCHING_COLUMN_CANDIDATES = {
    "p_ip": ["ip"], "p_gp": ["gp"], "p_gs": ["gs"], "p_bf": ["bf"],
    "p_np": ["#p", "np", "pitches"], "p_w": ["w"], "p_l": ["l"], "p_sv": ["sv"],
    "p_h": ["h"], "p_r": ["r"], "p_er": ["er"], "p_bb": ["bb"], "p_so": ["so", "k"],
    "p_hbp": ["hbp"], "p_wp": ["wp"], "p_hr": ["hr"],
    # Percentages / averages as exported (not summable — converted to counts in load_and_aggregate()).
    "p_s_pct": ["s%"], "p_fps_pct": ["fps%"], "p_sm_pct": ["sm%"],
    "p_gb_pct": ["gb%"], "p_fb_pct": ["fb%"], "p_ld_pct": ["ld%"],
    "p_top_mph": ["topmph"],
    "p_era": ["era"],  # only used to detect the league's game length for ERA
}
for _t, _label in PITCH_TYPES:
    PITCHING_COLUMN_CANDIDATES[f"p_{_t}_n"] = [_t]
    PITCHING_COLUMN_CANDIDATES[f"p_{_t}_strikes"] = [f"{_t}s"]
    PITCHING_COLUMN_CANDIDATES[f"p_{_t}_mph"] = [f"mph{_t}"]
    PITCHING_COLUMN_CANDIDATES[f"p_{_t}_top"] = [f"top{_t}"]

# Plain counts: summed when rows are combined.
PITCHING_SUM_FIELDS = (
    ["p_ip", "p_gp", "p_gs", "p_bf", "p_np", "p_w", "p_l", "p_sv", "p_h", "p_r", "p_er",
     "p_bb", "p_so", "p_hbp", "p_wp", "p_hr"]
    + [f"p_{t}_{k}" for t, _ in PITCH_TYPES for k in ("n", "strikes")]
)
# Percent/average columns read from the file, used only to derive counts below.
PITCHING_RATE_SOURCE_FIELDS = (
    ["p_s_pct", "p_fps_pct", "p_sm_pct", "p_gb_pct", "p_fb_pct", "p_ld_pct"]
    + [f"p_{t}_mph" for t, _ in PITCH_TYPES]
)
# Counts derived from those rates (strikes = S% x pitches, etc.), plus the
# denominator each one was taken over, so combining rows stays exact.
PITCHING_DERIVED_FIELDS = (
    ["p_strikes", "p_np_s", "p_fps", "p_bf_fps", "p_swm", "p_np_sm", "p_gb_w", "p_fb_w", "p_ld_w"]
    + [f"p_{t}_mph_w" for t, _ in PITCH_TYPES] + [f"p_{t}_n_mph" for t, _ in PITCH_TYPES]
)
# Top velocities: combined with max(), never summed.
PITCHING_MAX_FIELDS = ["p_top_mph"] + [f"p_{t}_top" for t, _ in PITCH_TYPES]
# Speeds outside this range are treated as scoring typos (real exports have
# shown 112 and 130 mph for 10U pitchers) — hidden from family reports and
# flagged on the coach page instead.
PLAUSIBLE_MPH = (25.0, 95.0)


def _plausible_mph(v: float) -> Optional[float]:
    return v if v and PLAUSIBLE_MPH[0] <= v <= PLAUSIBLE_MPH[1] else None


def detect_pitching_columns(headers: list[str], section_row: Optional[list[str]]) -> dict[str, Optional[str]]:
    """Same contract as detect_fielding_columns(), for the "Pitching" section.
    All-None when the export has no Pitching section — pitching is optional."""
    mapping: dict[str, Optional[str]] = {canon: None for canon in PITCHING_COLUMN_CANDIDATES}
    if not section_row:
        return mapping
    rng = _find_section_range(section_row, "Pitching")
    if not rng:
        return mapping
    start, end = rng
    for canon, candidates in PITCHING_COLUMN_CANDIDATES.items():
        mapping[canon] = _resolve_column_in_range(headers, start, end, candidates)
    return mapping


def _derive_pitching_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Turns the export's per-player percentages into summable counts (and
    their denominators) so duplicate rows / per-game rows combine exactly.
    Returns (df with the derived columns added, their names)."""
    if "p_np" not in df.columns and "p_bf" not in df.columns:
        return df, []
    zero = pd.Series(0.0, index=df.index)

    def col(name, default=zero):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else default

    np_, bf = col("p_np"), col("p_bf")
    new: dict[str, pd.Series] = {}

    def from_pct(pct_col, base, out_name, denom_name=None):
        pct = col(pct_col, pd.Series(float("nan"), index=df.index))
        has = pct.notna() & base.notna()
        new[out_name] = (pct.fillna(0) / 100.0 * base.fillna(0)).where(has, 0.0)
        if denom_name:
            new[denom_name] = base.fillna(0).where(has, 0.0)

    from_pct("p_s_pct", np_, "p_strikes", "p_np_s")
    from_pct("p_fps_pct", bf, "p_fps", "p_bf_fps")
    from_pct("p_sm_pct", np_, "p_swm", "p_np_sm")
    for kind in ("gb", "fb", "ld"):
        from_pct(f"p_{kind}_pct", bf, f"p_{kind}_w")
    for t, _ in PITCH_TYPES:
        n = col(f"p_{t}_n")
        mph = col(f"p_{t}_mph", pd.Series(float("nan"), index=df.index))
        has = mph.notna() & (mph > 0) & n.notna() & (n > 0)
        new[f"p_{t}_mph_w"] = (mph.fillna(0) * n.fillna(0)).where(has, 0.0)
        new[f"p_{t}_n_mph"] = n.fillna(0).where(has, 0.0)
    df = pd.concat([df.drop(columns=[c for c in new if c in df.columns]), pd.DataFrame(new, index=df.index)], axis=1)
    return df, list(new)


# Rows at the bottom of a GameChanger export ("Totals" summary row, a blank
# spacer row, a "Glossary" key) aren't players and must never become one.
# In the exports we've seen, these rows have no Last/First at all so the
# ordinary empty-name filter already drops them — this is a second,
# explicit safety net in case some export variant puts a marker word where
# a name would otherwise be.
FOOTER_NAME_MARKERS = {"totals", "total", "team totals", "glossary", ""}


def detect_header_row(csv_path: Path, max_scan: int = 5) -> int:
    """Most GameChanger exports have the real header on line 1. The
    "Advanced/Full Stats" export instead puts a section-label row first
    (mostly empty cells, with a few group headers like "Batting" /
    "Pitching" / "Fielding") and puts the real ~190-column header on line 2.
    Scans up to max_scan leading rows and returns the index (0-based) of the
    first one that's mostly filled in — a cheap, reliable signal, since a
    section-label row is <5% filled and a real header row is 100% filled.
    Falls back to 0 (today's behavior, unchanged) if nothing qualifies or
    the file can't be read."""
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = []
            for _ in range(max_scan):
                try:
                    rows.append(next(reader))
                except StopIteration:
                    break
    except Exception:
        return 0
    for i, row in enumerate(rows):
        if not row:
            continue
        non_empty = sum(1 for c in row if c.strip())
        if non_empty / len(row) >= 0.6:
            return i
    return 0


def missing_required_columns(mapping: dict[str, Optional[str]]) -> list[str]:
    """Like checking REQUIRED_CANONICAL directly, except "name" is also
    satisfied by having both "first" and "last" mapped (some exports have
    no single name column at all)."""
    missing = [c for c in REQUIRED_CANONICAL if c != "name" and not mapping.get(c)]
    has_name = bool(mapping.get("name")) or (bool(mapping.get("first")) and bool(mapping.get("last")))
    if not has_name:
        missing.insert(0, "name (or first + last)")
    return missing


def detect_columns(headers: list[str]) -> dict[str, Optional[str]]:
    """Propose a canonical-name -> actual-header mapping. Exact (normalized)
    matches win; substring matches are the fallback. Never silently guesses
    past what's actually in the header row."""
    lower_lookup = {h.strip().lower(): h for h in headers}
    mapping: dict[str, Optional[str]] = {}

    has_pitching_hints = any(
        any(hint == h.strip().lower() for hint in PITCHING_HINT_COLUMNS)
        for h in headers
    )

    for canon, candidates in COLUMN_CANDIDATES.items():
        found = None
        # exact match on a normalized candidate string
        for cand in candidates:
            if cand in lower_lookup:
                found = lower_lookup[cand]
                break
        if not found:
            # substring fallback, longest candidate first so "strikeouts"
            # doesn't lose to a coincidental short match
            for cand in sorted(candidates, key=len, reverse=True):
                for h in headers:
                    hl = h.strip().lower()
                    if cand == "k" and has_pitching_hints:
                        # "K" is ambiguous on a sheet that also has pitching
                        # columns; don't guess it means batting strikeouts.
                        continue
                    if cand in hl:
                        found = h
                        break
                if found:
                    break
        mapping[canon] = found
    return mapping


def print_column_mapping(headers: list[str], mapping: dict[str, Optional[str]]) -> None:
    print("\nDetected header row:")
    print("  " + " | ".join(headers))
    print("\nProposed column mapping (canonical <- source column):")
    width = max(len(k) for k in mapping) + 2
    for canon, source in mapping.items():
        status = source if source else "-- NOT FOUND --"
        print(f"  {canon:<{width}} <- {status}")
    missing_required = missing_required_columns(mapping)
    if missing_required:
        print("\nMissing required columns: " + ", ".join(missing_required))


def confirm_mapping(headers: list[str], mapping: dict[str, Optional[str]], assume_yes: bool) -> dict[str, Optional[str]]:
    print_column_mapping(headers, mapping)
    missing_required = missing_required_columns(mapping)
    if missing_required:
        sys.exit(
            "\nERROR: this CSV is missing columns this tool requires for batting "
            "reports: " + ", ".join(missing_required) + ".\n"
            "This is a hard stop, not a guess — either re-export from "
            "GameChanger (Team -> Stats -> Export Stats) with those columns "
            "included, or add your export's exact header names to "
            "COLUMN_CANDIDATES at the top of this script and re-run."
        )
    if assume_yes:
        print("\n--yes passed: proceeding with the mapping above.")
        return mapping
    if not sys.stdin.isatty():
        print(
            "\nNo TTY detected (non-interactive run) and --yes was not passed — "
            "proceeding with the detected mapping. Re-run with --yes to silence "
            "this message, or fix COLUMN_CANDIDATES if the mapping above is wrong."
        )
        return mapping
    resp = input("\nProceed with this column mapping? [Y/n] ").strip().lower()
    if resp not in ("", "y", "yes"):
        sys.exit("Aborted by user. Edit COLUMN_CANDIDATES or your CSV headers and re-run.")
    return mapping


# --------------------------------------------------------------------------
# Loading + aggregation
# --------------------------------------------------------------------------

# Baseball innings are written as X.Y where Y is THIRDS of an inning (.0,
# .1, .2 = 0/3, 1/3, 2/3) — never tenths. "7.2" means 7 and two-thirds
# innings, not 7.2 decimal innings. GameChanger's fielding-innings columns
# use this notation, so they can't be summed as plain floats without
# quietly drifting wrong (e.g. 7.2 + 3.1 as raw floats = 10.3, but the real
# total is 7⅔ + 3⅓ = 11 innings exactly, thirds carrying over into a whole
# inning). These are converted to true decimal on load (see
# _innings_to_decimal()) so ordinary summation stays correct, and
# converted back to X.Y notation only for display (see fmt_innings()).
INNINGS_NOTATION_FIELDS = [
    "def_innings", "pos_p", "pos_c", "pos_1b", "pos_2b", "pos_3b",
    "pos_ss", "pos_lf", "pos_cf", "pos_rf", "pos_sf",
]

FIELDING_COUNTING_FIELDS = ["putouts", "assists", "errors", "double_plays"] + INNINGS_NOTATION_FIELDS

COUNTING_FIELDS = ["pa", "ab", "r", "h", "2b", "3b", "hr", "rbi", "bb", "so", "sb", "cs", "hbp", "sf", "sh"] + FIELDING_COUNTING_FIELDS
# Pitching counts (and top velocities, which combine with max — see _agg_for()).
PITCHING_NUMERIC_FIELDS = PITCHING_SUM_FIELDS + PITCHING_MAX_FIELDS


def _agg_for(cols: list[str]) -> dict[str, str]:
    return {c: ("max" if c in PITCHING_MAX_FIELDS else "sum") for c in cols}


def _innings_to_decimal(v: float) -> float:
    """Reinterprets a baseball-notation innings value (X.Y, Y=thirds) as
    true decimal (X + Y/3) so it can be added normally. NaN passes through
    unchanged (present_counting's later .fillna(0.0) step handles that)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    whole = math.floor(v + 1e-9)
    thirds_digit = round((v - whole) * 10)
    if thirds_digit not in (0, 1, 2):
        thirds_digit = 0  # malformed/unexpected notation — treat as whole innings rather than guess or crash
    return whole + thirds_digit / 3.0


def fmt_innings(v: Optional[float]) -> str:
    """Inverse of _innings_to_decimal — formats true-decimal innings back
    into baseball's familiar X.Y (thirds) notation for display."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        v = 0.0
    whole = math.floor(v + 1e-9)
    thirds = round((v - whole) * 3)
    if thirds >= 3:
        whole += 1
        thirds = 0
    return f"{whole}.{thirds}"


def load_and_aggregate(csv_path: Path, mapping: dict[str, Optional[str]], assume_yes: bool, header_row: int = 0):
    """Returns (season_df, pergame_by_player, flags) where:
      - season_df: one row per player, season totals, canonical lowercase columns
      - pergame_by_player: {name: [ {date, ...counting fields...}, ... ]} or {}
        if the export has no per-game granularity
      - flags: dict of data-quality notes collected along the way

    header_row is the 0-indexed row to treat as the real header (see
    detect_header_row()) — needed for GameChanger's "Advanced/Full Stats"
    export, which has a section-label row above the real header.
    """
    flags: dict[str, object] = {}

    try:
        raw = pd.read_csv(csv_path, header=header_row)
    except Exception as exc:  # pragma: no cover - defensive
        raise ReportBuildError(f"Couldn't read {csv_path} as CSV ({exc}).")

    if raw.empty:
        raise ReportBuildError(f"{csv_path} has no data rows.")

    headers = list(raw.columns)

    # rename to canonical lowercase names where mapped
    rename = {source: canon for canon, source in mapping.items() if source}
    df = raw.rename(columns=rename)

    # coerce counting columns to numeric, treat missing as absent (not zero)
    # so we can tell "column not in export" apart from "value is 0"
    present_counting = [c for c in COUNTING_FIELDS + PITCHING_NUMERIC_FIELDS if c in df.columns]
    for c in present_counting:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if c in INNINGS_NOTATION_FIELDS or c == "p_ip":
            df[c] = df[c].apply(_innings_to_decimal)
    # ERA's game length (6 innings in most youth leagues) is read back out of
    # the export itself — ERA = ER / IP x innings-per-game — rather than assumed.
    if all(c in df.columns for c in ("p_er", "p_ip", "p_era")):
        era = pd.to_numeric(df["p_era"], errors="coerce")
        ok = (df["p_er"] > 0) & (df["p_ip"] > 0) & (era > 0)
        if ok.any():
            game_len = (era[ok] * df.loc[ok, "p_ip"] / df.loc[ok, "p_er"]).median()
            if 3 <= game_len <= 9:
                flags["era_innings"] = int(round(game_len))
    df, derived = _derive_pitching_counts(df)
    present_counting += derived

    # Some exports have no single name column — just separate Last/First.
    # Build a combined "First Last" field (matters: templates split on the
    # first space to get a kid's first name, so the order isn't arbitrary).
    if "name" not in df.columns and "first" in df.columns and "last" in df.columns:
        first = df["first"].fillna("").astype(str).str.strip()
        last = df["last"].fillna("").astype(str).str.strip()
        df["name"] = (first + " " + last).str.strip()

    if "name" not in df.columns:
        raise ReportBuildError("Could not find a player-name column; aborting.")
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["name"] != ""]
    # Defensive filter: a "Totals" summary row or "Glossary" key at the
    # bottom of a GameChanger export must never be treated as a player.
    df = df[~df["name"].str.strip().str.lower().isin(FOOTER_NAME_MARKERS)]

    has_date = "date" in df.columns and df["date"].notna().any()
    n_unique_names = df["name"].nunique()
    avg_rows_per_name = (len(df) / n_unique_names) if n_unique_names else 1.0
    # A real per-game log has EVERY player appearing many times (one row
    # per game). A season-totals export with one incidental duplicate (a
    # player re-added under a different jersey number, a data-entry quirk)
    # has an average rows-per-name barely above 1.0 — "any duplicate at
    # all" was too trigger-happy a signal and wrongly classified a whole
    # season-totals roster as per-game data because of a single stray
    # repeated name, which then silently broke trend computation for every
    # player in the file (see load_and_aggregate() docstring / changelog).
    is_pergame = has_date or avg_rows_per_name >= 2.0

    pergame_by_player: dict[str, list[dict]] = {}

    if is_pergame:
        if has_date:
            df["_parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["_parsed_date"].isna().any():
                flags["unparsed_dates"] = int(df["_parsed_date"].isna().sum())
            df = df.sort_values(by=["name", "_parsed_date"], na_position="last")
        else:
            flags["pergame_no_date"] = True
            # fall back to file order as a chronology proxy, per player

        for name, group in df.groupby("name", sort=False):
            rows = []
            for _, r in group.iterrows():
                row = {c: (0 if pd.isna(r.get(c)) else float(r.get(c))) for c in present_counting}
                row["date"] = r.get("_parsed_date") if has_date else None
                rows.append(row)
            pergame_by_player[name] = rows

        agg = _agg_for(present_counting)
        season_df = df.groupby("name", as_index=False, sort=False).agg(agg)
    else:
        # Still collapse any incidental duplicate name by summing counting
        # stats, so the same player never ends up as two separate rows (and
        # two separate output files) just because they appeared twice in
        # the export. A genuine season-totals file has one row per player
        # already, so this is a no-op in the common case.
        if df["name"].duplicated().any():
            dup_names = sorted(df.loc[df["name"].duplicated(keep=False), "name"].unique().tolist())
            flags["merged_duplicate_names"] = dup_names
            other_cols = [c for c in df.columns if c not in present_counting and c != "name"]
            first_seen = df.drop_duplicates(subset="name", keep="first")[["name"] + other_cols]
            if present_counting:
                summed = df.groupby("name", as_index=False, sort=False)[present_counting].agg(_agg_for(present_counting))
                season_df = first_seen.merge(summed, on="name", how="left")
            else:
                season_df = first_seen
        else:
            season_df = df.copy()
        for c in present_counting:
            season_df[c] = season_df[c].fillna(0.0)

    flags["is_pergame"] = is_pergame
    flags["n_players"] = len(season_df)

    if has_date:
        valid_dates = df["_parsed_date"].dropna()
        if len(valid_dates):
            flags["date_min"] = valid_dates.min()
            flags["date_max"] = valid_dates.max()
            flags["n_games"] = int(df.loc[valid_dates.index, "date"].nunique()) if "date" in df.columns else None

    return season_df, pergame_by_player, flags


# --------------------------------------------------------------------------
# Derived stat computation
# --------------------------------------------------------------------------

@dataclass
class PlayerStats:
    name: str
    number: Optional[str] = None
    pa: float = 0.0
    pa_is_approx: bool = False
    ab: float = 0.0
    h: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    b3: float = 0.0
    hr: float = 0.0
    rbi: float = 0.0
    r: float = 0.0
    bb: float = 0.0
    so: float = 0.0
    sb: float = 0.0
    hbp: float = 0.0
    sf: float = 0.0
    sh: float = 0.0
    tb: float = 0.0
    avg: float = 0.0
    obp: float = 0.0
    obp_is_approx: bool = False
    slg: float = 0.0
    ops: float = 0.0
    iso: float = 0.0
    k_pct: float = 0.0
    bb_pct: float = 0.0
    bb_per_k: Optional[float] = None
    reliability: dict = field(default_factory=dict)
    is_regular: bool = False
    below_rate_stat_floor: bool = False

    # Defense (only populated when the export has a Fielding section —
    # see detect_fielding_columns()). Innings fields are true decimal
    # (see _innings_to_decimal()); format with the fmt_inn Jinja filter.
    has_fielding_data: bool = False
    putouts: float = 0.0
    assists: float = 0.0
    errors: float = 0.0
    double_plays: float = 0.0
    def_tc: float = 0.0  # total chances = PO + A + E, computed (not read from the CSV's own TC)
    def_innings: float = 0.0  # sum of innings across every position (the Fielding section's "Total" column)
    fpct: float = 0.0
    error_rate: float = 0.0
    range_factor: Optional[float] = None
    pos_innings: dict = field(default_factory=dict)  # {"pos_ss": 12.33, ...} true-decimal innings per position
    primary_position: Optional[str] = None
    primary_position_innings: float = 0.0
    below_defense_floor: bool = True

    # Pitching (only populated when the export has a Pitching section — see
    # detect_pitching_columns()). p_ip is true decimal innings.
    has_pitching_data: bool = False
    below_pitching_floor: bool = True
    p_ip: float = 0.0
    p_gp: float = 0.0
    p_gs: float = 0.0
    p_bf: float = 0.0
    p_np: float = 0.0
    p_w: float = 0.0
    p_l: float = 0.0
    p_sv: float = 0.0
    p_h: float = 0.0
    p_r: float = 0.0
    p_er: float = 0.0
    p_bb: float = 0.0
    p_so: float = 0.0
    p_hbp: float = 0.0
    p_wp: float = 0.0
    p_hr: float = 0.0
    strike_pct: Optional[float] = None
    fps_pct: Optional[float] = None
    swm_pct: Optional[float] = None
    bb_per_inn: Optional[float] = None
    k_per_bb: Optional[float] = None
    opp_avg: Optional[float] = None
    era: Optional[float] = None
    era_innings: int = 6
    whip: Optional[float] = None
    pitches_per_inn: Optional[float] = None
    gb_pct: Optional[float] = None
    ld_pct: Optional[float] = None
    fb_pct: Optional[float] = None
    top_mph: Optional[float] = None
    top_mph_suspect: Optional[float] = None  # recorded but implausible (see PLAUSIBLE_MPH)
    pitch_mix: list = field(default_factory=list)


def compute_player_stats(row: pd.Series, era_innings: int = 6) -> PlayerStats:
    def g(col):
        v = row.get(col)
        return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    ab, h, bb, so = g("ab"), g("h"), g("bb"), g("so")
    b2, b3, hr = g("2b"), g("3b"), g("hr")
    b1 = max(h - b2 - b3 - hr, 0.0)
    hbp, sf, sh = g("hbp"), g("sf"), g("sh")
    has_hbp = "hbp" in row.index and not pd.isna(row.get("hbp"))
    has_sf = "sf" in row.index and not pd.isna(row.get("sf"))
    has_sh = "sh" in row.index and not pd.isna(row.get("sh"))
    has_pa_col = "pa" in row.index and not pd.isna(row.get("pa"))

    if has_pa_col:
        pa = g("pa")
        pa_is_approx = False
    elif has_hbp or has_sf or has_sh:
        pa = ab + bb + hbp + sf + sh
        pa_is_approx = False
    else:
        pa = ab + bb
        pa_is_approx = True

    tb = b1 + 2 * b2 + 3 * b3 + 4 * hr
    avg = (h / ab) if ab > 0 else 0.0

    if has_hbp or has_sf:
        obp_denom = ab + bb + hbp + sf
        obp = ((h + bb + hbp) / obp_denom) if obp_denom > 0 else 0.0
        obp_is_approx = False
    else:
        obp_denom = ab + bb
        obp = ((h + bb) / obp_denom) if obp_denom > 0 else 0.0
        obp_is_approx = True

    slg = (tb / ab) if ab > 0 else 0.0
    ops = obp + slg
    iso = slg - avg
    k_pct = (so / pa * 100) if pa > 0 else 0.0
    bb_pct = (bb / pa * 100) if pa > 0 else 0.0
    bb_per_k = (bb / so) if so > 0 else None

    p = PlayerStats(
        name=row["name"],
        number=str(row.get("number")) if "number" in row.index and not pd.isna(row.get("number")) else None,
        pa=pa, pa_is_approx=pa_is_approx, ab=ab, h=h, b1=b1, b2=b2, b3=b3, hr=hr,
        rbi=g("rbi"), r=g("r"), bb=bb, so=so, sb=g("sb"), hbp=hbp, sf=sf, sh=sh,
        tb=tb, avg=avg, obp=obp, obp_is_approx=obp_is_approx, slg=slg, ops=ops, iso=iso,
        k_pct=k_pct, bb_pct=bb_pct, bb_per_k=bb_per_k,
    )
    p.below_rate_stat_floor = p.pa < RATE_STAT_FLOOR

    # Defense — only present when the export had a Fielding section at all.
    has_fielding_cols = any(k in row.index for k in ("putouts", "assists", "errors", "def_innings"))
    if has_fielding_cols:
        po, a, e = g("putouts"), g("assists"), g("errors")
        tc = po + a + e
        p.putouts, p.assists, p.errors = po, a, e
        p.double_plays = g("double_plays")
        p.def_tc = tc
        p.def_innings = g("def_innings")
        p.fpct = ((po + a) / tc) if tc > 0 else 0.0
        p.error_rate = (e / tc * 100) if tc > 0 else 0.0
        p.range_factor = ((po + a) / p.def_innings * 9) if p.def_innings > 0 else None
        p.pos_innings = {
            key: g(key) for key in
            ("pos_p", "pos_c", "pos_1b", "pos_2b", "pos_3b", "pos_ss", "pos_lf", "pos_cf", "pos_rf", "pos_sf")
            if key in row.index
        }
        if p.pos_innings:
            best_pos, best_innings = max(p.pos_innings.items(), key=lambda kv: kv[1])
            if best_innings > 0:
                p.primary_position = POSITION_LABELS.get(best_pos, best_pos)
                p.primary_position_innings = best_innings
        p.has_fielding_data = tc > 0 or p.def_innings > 0
        p.below_defense_floor = tc < DEFENSE_RATE_STAT_FLOOR

    # Pitching — only present when the export had a Pitching section.
    if "p_bf" in row.index or "p_ip" in row.index:
        for f in ("p_ip", "p_gp", "p_gs", "p_bf", "p_np", "p_w", "p_l", "p_sv", "p_h", "p_r",
                  "p_er", "p_bb", "p_so", "p_hbp", "p_wp", "p_hr"):
            setattr(p, f, g(f))
        p.has_pitching_data = p.p_bf > 0 or p.p_ip > 0
        p.below_pitching_floor = p.p_bf < PITCHING_FLOOR_BF
        ip = p.p_ip
        p.strike_pct = (g("p_strikes") / g("p_np_s") * 100) if g("p_np_s") > 0 else None
        p.fps_pct = (g("p_fps") / g("p_bf_fps") * 100) if g("p_bf_fps") > 0 else None
        p.swm_pct = (g("p_swm") / g("p_np_sm") * 100) if g("p_np_sm") > 0 else None
        p.bb_per_inn = (p.p_bb / ip) if ip > 0 else None
        p.k_per_bb = (p.p_so / p.p_bb) if p.p_bb > 0 else None
        ab_against = p.p_bf - p.p_bb - p.p_hbp  # approximation: ignores sacrifices / interference
        p.opp_avg = (p.p_h / ab_against) if ab_against > 0 else None
        p.era_innings = era_innings
        p.era = (p.p_er / ip * era_innings) if ip > 0 else None
        p.whip = ((p.p_bb + p.p_h) / ip) if ip > 0 else None
        p.pitches_per_inn = (p.p_np / ip) if ip > 0 and p.p_np > 0 else None
        bip = g("p_gb_w") + g("p_ld_w") + g("p_fb_w")
        if bip > 0:
            p.gb_pct = g("p_gb_w") / bip * 100
            p.ld_pct = g("p_ld_w") / bip * 100
            p.fb_pct = g("p_fb_w") / bip * 100
        raw_top = g("p_top_mph")
        p.top_mph = _plausible_mph(raw_top)
        p.top_mph_suspect = raw_top if raw_top and p.top_mph is None else None
        mix = []
        for t, label in PITCH_TYPES:
            n = g(f"p_{t}_n")
            if n > 0:
                mph_n = g(f"p_{t}_n_mph")
                mix.append({
                    "label": label, "n": n,
                    "share": (n / p.p_np * 100) if p.p_np > 0 else None,
                    "strike_pct": (g(f"p_{t}_strikes") / n * 100) if f"p_{t}_strikes" in row.index else None,
                    "avg_mph": _plausible_mph(g(f"p_{t}_mph_w") / mph_n) if mph_n > 0 else None,
                    "top_mph": _plausible_mph(g(f"p_{t}_top")),
                })
        p.pitch_mix = mix

    return p


_RELIABILITY_BASIS = {
    "PA": lambda p: p.pa,
    "AB": lambda p: p.ab,
    "TC": lambda p: p.def_tc,
    "DEF_INN": lambda p: p.def_innings,
    "NP": lambda p: p.p_np,
    "BF": lambda p: p.p_bf,
}


def reliability_tier(stat_key: str, player: PlayerStats) -> str:
    basis, threshold = RELIABILITY_THRESHOLDS[stat_key]
    n = _RELIABILITY_BASIS[basis](player)
    if n >= threshold:
        return "reliable"
    if n >= 0.4 * threshold:
        return "partial"
    return "noisy"


_RELIABILITY_ORDER = {"noisy": 0, "partial": 1, "reliable": 2}


def weaker(a: str, b: str) -> str:
    return a if _RELIABILITY_ORDER[a] <= _RELIABILITY_ORDER[b] else b


def attach_reliability(player: PlayerStats) -> None:
    r = {
        "k_pct": reliability_tier("k_pct", player),
        "bb_pct": reliability_tier("bb_pct", player),
        "slg": reliability_tier("slg", player),
        "obp": reliability_tier("obp", player),
        "avg": reliability_tier("avg", player),
    }
    r["bb_per_k"] = weaker(r["k_pct"], r["bb_pct"])
    r["ops"] = weaker(r["obp"], r["slg"])
    r["iso"] = weaker(r["slg"], r["avg"])
    if player.has_fielding_data:
        r["fpct"] = reliability_tier("fpct", player)
        r["error_rate"] = reliability_tier("error_rate", player)
        r["range_factor"] = reliability_tier("range_factor", player)
    if player.has_pitching_data:
        for key in ("strike_pct", "fps_pct", "swm_pct", "bb_per_inn", "k_per_bb", "opp_avg"):
            r[key] = reliability_tier(key, player)
    player.reliability = r


# --------------------------------------------------------------------------
# Team-level aggregation, tiers, distance annotations
# --------------------------------------------------------------------------

def team_ops_audit(regulars: list[PlayerStats]) -> dict:
    h = sum(p.h for p in regulars)
    ab = sum(p.ab for p in regulars)
    bb = sum(p.bb for p in regulars)
    hbp = sum(p.hbp for p in regulars)
    sf = sum(p.sf for p in regulars)
    tb = sum(p.tb for p in regulars)
    team_avg = (h / ab) if ab else 0.0
    obp_denom = ab + bb + hbp + sf
    team_obp = ((h + bb + hbp) / obp_denom) if obp_denom else 0.0
    team_slg = (tb / ab) if ab else 0.0
    team_ops = team_obp + team_slg
    return {
        "avg": team_avg, "obp": team_obp, "slg": team_slg, "ops": team_ops,
        "use_bars": team_ops < TEAM_OPS_STRONG_CUTOFF,
        "n_regulars": len(regulars),
    }


def percentile_from_top(value: float, pool: list[float], lower_is_better: bool) -> float:
    """1.0 = best in the pool, near 0 = worst. Ties are averaged."""
    n = len(pool)
    if n <= 1:
        return 1.0
    if lower_is_better:
        better = sum(1 for v in pool if v < value)
        equal = sum(1 for v in pool if v == value)
    else:
        better = sum(1 for v in pool if v > value)
        equal = sum(1 for v in pool if v == value)
    worse = n - better - equal
    # rank from the bottom, averaging ties, then normalize to [0,1]
    rank_from_bottom = worse + (equal - 1) / 2.0 if equal > 0 else worse
    return rank_from_bottom / (n - 1) if n > 1 else 1.0


def tier_from_percentile(pct: float) -> str:
    if pct >= 0.90:
        return "leading"
    if pct >= 0.65:
        return "above_middle"
    if pct >= 0.35:
        return "around_middle"
    if pct >= 0.10:
        return "below_middle"
    return "still_finding_it"


@dataclass
class StatCompare:
    tier: str
    tier_label: str
    distance_text: Optional[str]


def compare_points_stat(value: float, pool: list[float], tier: str, unit_name: str, unit_from_points_fn) -> Optional[str]:
    """Generic distance annotation for a stat measured on the .000 'points'
    scale (AVG/OBP/SLG/OPS). unit_from_points_fn converts a point-delta into
    a whole real-world count (extra hits, extra times on base, etc.), or
    returns None for composite stats (OPS) that don't have one natural
    real-world unit — those get a qualitative rollup instead of a bare
    three-digit point gap, which reads as alarming rather than informative."""
    sorted_desc = sorted(pool, reverse=True)
    if tier == "leading":
        if len(sorted_desc) < 2:
            return "The only regular with enough at-bats to compare this stat against."
        next_best = sorted_desc[1] if sorted_desc[0] == value else sorted_desc[0]
        pts = round((value - next_best) * 1000)
        if pts == 0:
            return "Tied for the team's best in this stat."
        if pts < 0:
            return "Right near the top of the team in this stat, with no single clear leader to chase."
        return f"{pts} points ahead of the next hitter in the group."
    else:
        median_val = statistics.median(pool)
        if tier == "around_middle":
            target = sorted(pool)[max(0, int(len(pool) * 0.65) - 1)]
        elif tier == "above_middle":
            target = max(pool)
        else:
            target = median_val
        pts = round((target - value) * 1000)
        if pts <= 0:
            return "Already at or above this benchmark for the team's regulars."
        extra = unit_from_points_fn(pts)
        if extra:
            return f"{extra} would close most of this gap."
        # No natural single unit for this stat (OPS) — describe it as a
        # composite instead of printing a bare, alarming point gap.
        return "A composite of the AVG/OBP/SLG gaps above — closing those closes this one too."


def compare_rate_stat(value_pct: float, pool_pct: list[float], tier: str, pa: float, lower_is_better: bool, unit_singular: str, unit_plural: str) -> Optional[str]:
    """Distance annotation for K%/BB%, converted to a whole real count using
    this player's own PA (exact, since these are season totals, not an
    estimate)."""
    if tier == "leading":
        sorted_pool = sorted(pool_pct) if lower_is_better else sorted(pool_pct, reverse=True)
        if len(sorted_pool) < 2:
            return "The only regular with enough at-bats to compare this stat against."
        next_best = sorted_pool[1] if sorted_pool[0] == value_pct else sorted_pool[0]
        gap = (next_best - value_pct) if lower_is_better else (value_pct - next_best)
        if gap == 0:
            return "Tied for the team's best in this stat."
        if gap < 0:
            return "Right near the top of the team in this stat, with no single clear leader to chase."
        return "Well clear of the next-closest regular on the team."
    median_val = statistics.median(pool_pct)
    if tier == "around_middle":
        s = sorted(pool_pct) if lower_is_better else sorted(pool_pct, reverse=True)
        target = s[max(0, int(len(s) * 0.65) - 1)]
        # This describes where the target actually sits (not a "tier" — that
        # word only means something where the percentile-band system is
        # visibly explained, e.g. the parent/player tier-line; this text also
        # gets reused verbatim in coach mode, which has no such system on
        # screen, so it needs to stand on its own).
        target_phrase = "would move the player into the upper third of the team's regular hitters"
    elif tier == "above_middle":
        target = min(pool_pct) if lower_is_better else max(pool_pct)
        target_phrase = "would make the player the team's best in this stat"
    else:
        target = median_val
        target_phrase = "would put the player at the middle of the team's regular hitters"
    delta_pct = (value_pct - target) if lower_is_better else (target - value_pct)
    if delta_pct <= 0:
        return "Already at or beyond this benchmark for the team's regulars."
    count = max(1, round(delta_pct / 100.0 * pa))
    noun = unit_singular if count == 1 else unit_plural
    verb = "fewer" if lower_is_better else "more"
    return f"About {count} {verb} {noun} over a season this length {target_phrase}."


def build_comparisons(player: PlayerStats, regulars: list[PlayerStats]) -> dict:
    """Returns {stat_key: StatCompare} for every stat we tier/distance,
    computed against the 'regulars' pool (which includes this player, since
    they are one, if they clear the min-PA bar)."""
    out: dict[str, StatCompare] = {}

    def do_points_stat(key, attr, unit_fn):
        pool = [getattr(p, attr) for p in regulars]
        pct = percentile_from_top(getattr(player, attr), pool, lower_is_better=False)
        tier = tier_from_percentile(pct)
        dist = compare_points_stat(getattr(player, attr), pool, tier, attr, unit_fn)
        out[key] = StatCompare(tier, TIER_LABELS[tier], dist)

    do_points_stat("avg", "avg", lambda pts: (lambda extra: f"{extra} more hits" if extra else None)(max(1, round(pts / 1000 * player.ab))))
    do_points_stat("obp", "obp", lambda pts: (lambda extra: f"{extra} more times on base" if extra else None)(max(1, round(pts / 1000 * (player.ab + player.bb + player.hbp + player.sf)))))
    do_points_stat("slg", "slg", lambda pts: (lambda extra: f"{extra} more total bases" if extra else None)(max(1, round(pts / 1000 * player.ab))))
    do_points_stat("ops", "ops", lambda pts: None)

    kpool = [p.k_pct for p in regulars]
    kpct = percentile_from_top(player.k_pct, kpool, lower_is_better=True)
    ktier = tier_from_percentile(kpct)
    kdist = compare_rate_stat(player.k_pct, kpool, ktier, player.pa, True, "strikeout", "strikeouts")
    out["k_pct"] = StatCompare(ktier, TIER_LABELS[ktier], kdist)

    bpool = [p.bb_pct for p in regulars]
    bpct = percentile_from_top(player.bb_pct, bpool, lower_is_better=False)
    btier = tier_from_percentile(bpct)
    bdist = compare_rate_stat(player.bb_pct, bpool, btier, player.pa, False, "walk", "walks")
    out["bb_pct"] = StatCompare(btier, TIER_LABELS[btier], bdist)

    bkpool = [p.bb_per_k for p in regulars if p.bb_per_k is not None]
    if player.bb_per_k is not None and bkpool:
        bkpct = percentile_from_top(player.bb_per_k, bkpool, lower_is_better=False)
        bktier = tier_from_percentile(bkpct)
        bksorted = sorted(bkpool, reverse=True)
        if bktier == "leading" and len(bksorted) > 1:
            next_best = bksorted[1] if bksorted[0] == player.bb_per_k else bksorted[0]
            mult = (player.bb_per_k / next_best) if next_best > 0 else None
            bkdist = f"Roughly {mult:.1f}x the next-closest regular's mark." if mult else None
        elif bktier != "leading":
            target = statistics.median(bkpool)
            if player.bb_per_k < target and player.so > 0:
                extra_bb = max(1, math.ceil(target * player.so - player.bb))
                bkdist = f"About {extra_bb} more walks at this strikeout rate would put the player at the middle of the group."
            else:
                bkdist = None
        else:
            bkdist = None
        out["bb_per_k"] = StatCompare(bktier, TIER_LABELS[bktier], bkdist)
    else:
        out["bb_per_k"] = StatCompare("around_middle", TIER_LABELS["around_middle"], None)

    return out


def compare_range_factor(value: float, pool: list[float], tier: str) -> Optional[str]:
    """Distance annotation for range factor (plays made per 9 innings) —
    a counting rate, not a .000-scale stat like AVG, so this stays a
    simpler, separate function rather than reusing compare_points_stat."""
    if tier == "leading":
        s = sorted(pool, reverse=True)
        if len(s) < 2:
            return None
        next_best = s[1] if s[0] == value else s[0]
        gap = value - next_best
        return "Well clear of the next-closest fielder in this group." if gap > 0 else None
    median_val = statistics.median(pool)
    if tier == "around_middle":
        target = sorted(pool)[max(0, int(len(pool) * 0.65) - 1)]
    elif tier == "above_middle":
        target = max(pool)
    else:
        target = median_val
    gap = target - value
    if gap <= 0:
        return None
    return f"About {gap:.1f} more plays made per 9 innings would close most of this gap."


def build_defensive_comparisons(player: PlayerStats, defensive_pool: list[PlayerStats]) -> dict:
    """Same tier/percentile/distance-text machinery as build_comparisons(),
    applied to fielding % and error rate (both position-agnostic enough to
    compare across a whole roster) and range factor (which build_env's
    template copy explicitly caveats as position-influenced — a catcher
    and an outfielder have very different natural opportunity rates, so
    "below the middle" there means something different than it does for
    AVG/OBP)."""
    out: dict[str, StatCompare] = {}

    fpool = [p.fpct for p in defensive_pool]
    fpct_pctl = percentile_from_top(player.fpct, fpool, lower_is_better=False)
    ftier = tier_from_percentile(fpct_pctl)
    fdist = compare_points_stat(
        player.fpct, fpool, ftier, "fpct",
        lambda pts: (lambda extra: f"{extra} fewer errors" if extra else None)(max(1, round(pts / 1000 * player.def_tc))) if player.def_tc > 0 else None,
    )
    out["fpct"] = StatCompare(ftier, TIER_LABELS[ftier], fdist)

    epool = [p.error_rate for p in defensive_pool]
    epctl = percentile_from_top(player.error_rate, epool, lower_is_better=True)
    etier = tier_from_percentile(epctl)
    edist = compare_rate_stat(player.error_rate, epool, etier, player.def_tc, True, "error", "errors")
    out["error_rate"] = StatCompare(etier, TIER_LABELS[etier], edist)

    rpool = [p.range_factor for p in defensive_pool if p.range_factor is not None]
    if player.range_factor is not None and rpool:
        rpctl = percentile_from_top(player.range_factor, rpool, lower_is_better=False)
        rtier = tier_from_percentile(rpctl)
        rdist = compare_range_factor(player.range_factor, rpool, rtier)
        out["range_factor"] = StatCompare(rtier, TIER_LABELS[rtier], rdist)
    else:
        out["range_factor"] = StatCompare("around_middle", TIER_LABELS["around_middle"], None)

    return out


def build_compare_visuals(player: PlayerStats, regulars: list[PlayerStats], team_ops: dict) -> dict:
    """Geometry for the 'Where the Player Stands on the Roster' module: a
    dot-plot distribution across the regulars (default), or bars scaled to
    the player's own number with a team-average marker. See COMPARISON_VIEW."""
    stat_keys = ["avg", "obp", "slg", "ops"]
    team_vals = {"avg": team_ops["avg"], "obp": team_ops["obp"], "slg": team_ops["slg"], "ops": team_ops["ops"]}

    use_bars = COMPARISON_VIEW == "bars" or (COMPARISON_VIEW == "auto" and team_ops["use_bars"])
    if use_bars:
        bars = {}
        for key in stat_keys:
            val = getattr(player, key)
            team_val = team_vals[key]
            domain = max(val, team_val) * 1.08 or 1.0
            bars[key] = {
                "value": val,
                "team_value": team_val,
                "bar_pct": round(val / domain * 100, 1),
                "marker_pct": round(team_val / domain * 100, 1),
            }
        return {"mode": "bars", "bars": bars}
    else:
        dots = {}
        for key in stat_keys:
            ranked = sorted(regulars, key=lambda q: getattr(q, key))
            vals = [getattr(q, key) for q in ranked]
            lo, hi = (vals[0], vals[-1]) if vals else (0, 1)
            span = (hi - lo) or 1.0
            val = getattr(player, key)
            points = []
            for q, v in zip(ranked, vals):
                pct = round((v - lo) / span * 100, 1)
                # Highlight by identity, not value, so a teammate with the
                # exact same number isn't also drawn as "the player".
                points.append({"pct": pct, "me": q is player})
            dots[key] = {"value": val, "lo": lo, "hi": hi, "points": points}
        return {"mode": "dots", "dots": dots}


# --------------------------------------------------------------------------
# Pitching: coaching goals + teammate dots + one-line pitching goal
# --------------------------------------------------------------------------

PITCHING_DOT_KEYS = ["strike_pct", "fps_pct", "swm_pct", "bb_per_inn", "k_per_bb", "opp_avg"]


def pitching_goal_tier(key: str, player: PlayerStats) -> Optional[str]:
    """goal_met / near_goal / below_goal against PITCHING_GOALS, or None
    when there's no number to judge."""
    goal = PITCHING_GOALS[key]
    v = getattr(player, key)
    if key == "k_per_bb" and v is None:
        # No walks at all: with any strikeouts that's the best possible ratio.
        return "goal_met" if player.p_so > 0 else None
    if v is None:
        return None
    if goal["higher_is_better"]:
        return "goal_met" if v >= goal["goal"] else ("near_goal" if v >= goal["near"] else "below_goal")
    return "goal_met" if v <= goal["goal"] else ("near_goal" if v <= goal["near"] else "below_goal")


def build_pitching_context(player: PlayerStats, pitch_pool: list[PlayerStats]) -> dict:
    goals = {}
    for key, meta in PITCHING_GOALS.items():
        tier = pitching_goal_tier(key, player)
        goals[key] = {
            "tier": tier,
            "tier_label": GOAL_TIER_LABELS.get(tier, ""),
            "goal_text": meta["goal_text"],
        }

    dots = {}
    for key in PITCHING_DOT_KEYS:
        val = getattr(player, key)
        ranked = sorted((q for q in pitch_pool if getattr(q, key) is not None), key=lambda q: getattr(q, key))
        if val is None or len(ranked) < 3:
            continue  # too few teammates to draw a meaningful spread
        vals = [getattr(q, key) for q in ranked]
        lo, hi = vals[0], vals[-1]
        span = (hi - lo) or 1.0
        dots[key] = {
            "value": val, "lo": lo, "hi": hi,
            "points": [{"pct": round((v - lo) / span * 100, 1), "me": q is player} for q, v in zip(ranked, vals)],
        }

    next_goal = None
    for wanted in ("below_goal", "near_goal"):
        for key in PITCHING_GOAL_PRIORITY:
            if goals[key]["tier"] == wanted:
                next_goal = {"key": key, "text": PITCHING_NEXT_GOAL_TEXT[key]}
                break
        if next_goal:
            break

    return {
        "goals": goals,
        "dots": dots,
        "next_goal": next_goal,
        "n_pitchers": len(pitch_pool),
        "has_batted_ball": player.gb_pct is not None,
        "has_velocity": bool(player.top_mph) or any(m["avg_mph"] or m["top_mph"] for m in player.pitch_mix),
        "has_pitch_mix": bool(player.pitch_mix),
    }


# --------------------------------------------------------------------------
# Development focus (exactly one, chosen programmatically)
# --------------------------------------------------------------------------

FOCUS_PRIORITY = ["k_pct", "bb_pct", "obp", "slg", "avg"]
FOCUS_STAT_META = {
    "k_pct": {"label": "Strikeout Rate (K%)", "fmt": lambda v: f"{v:.1f}%"},
    "bb_pct": {"label": "Walk Rate (BB%)", "fmt": lambda v: f"{v:.1f}%"},
    "obp": {"label": "On-Base % (OBP)", "fmt": lambda v: f".{round(v*1000):03d}" if v < 1 else f"{v:.3f}"},
    "slg": {"label": "Slugging % (SLG)", "fmt": lambda v: f".{round(v*1000):03d}" if v < 1 else f"{v:.3f}"},
    "avg": {"label": "Batting Average (AVG)", "fmt": lambda v: f".{round(v*1000):03d}" if v < 1 else f"{v:.3f}"},
}


def choose_focus(player: PlayerStats, comparisons: dict, regulars: list[PlayerStats]) -> Optional[dict]:
    """Pick exactly one development focus: prefer a 'reliable'-labeled stat
    with the clearest room for improvement; fall back to 'partial'; if the
    player is already leading (or tied for leading) in everything reliable
    and partial, pick the least-dominant one as a positively-framed stretch
    goal, so every eligible report still gets a real, honest focus block."""
    if player.below_rate_stat_floor:
        return None

    def pct_for(key):
        pool_map = {
            "k_pct": ([p.k_pct for p in regulars], True),
            "bb_pct": ([p.bb_pct for p in regulars], False),
            "obp": ([p.obp for p in regulars], False),
            "slg": ([p.slg for p in regulars], False),
            "avg": ([p.avg for p in regulars], False),
        }
        pool, lower_better = pool_map[key]
        val = getattr(player, key)
        return percentile_from_top(val, pool, lower_better)

    for tier_filter in ("reliable", "partial"):
        candidates = [k for k in FOCUS_PRIORITY if player.reliability.get(k) == tier_filter]
        if not candidates:
            continue
        candidates.sort(key=pct_for)  # ascending percentile = most room first
        chosen = candidates[0]
        if pct_for(chosen) < 0.90:  # real room to grow
            return _focus_payload(chosen, player, comparisons, stretch=False)

    # Fallback: everything reliable/partial is already near the top of the
    # roster. Give an honest stretch-goal focus rather than no focus at all.
    reliable_or_partial = [k for k in FOCUS_PRIORITY if player.reliability.get(k) in ("reliable", "partial")]
    if reliable_or_partial:
        reliable_or_partial.sort(key=pct_for)
        return _focus_payload(reliable_or_partial[0], player, comparisons, stretch=True)
    return None


def _focus_payload(stat_key: str, player: PlayerStats, comparisons: dict, stretch: bool) -> dict:
    meta = FOCUS_STAT_META[stat_key]
    cmp_ = comparisons.get(stat_key)
    return {
        "stat_key": stat_key,
        "label": meta["label"],
        "current_value": meta["fmt"](getattr(player, stat_key)),
        "tier_label": cmp_.tier_label if cmp_ else None,
        "distance_text": cmp_.distance_text if cmp_ else None,
        "stretch": stretch,
    }


# --------------------------------------------------------------------------
# Trend (season-thirds), only when per-game rows are available
# --------------------------------------------------------------------------

def _trend_change(curr: float, prev: float, higher_is_better: bool, flat_epsilon: float) -> dict:
    """A small, labeled delta indicator for one stat between two trend
    periods. Direction is based on the row's own metric (never a shared,
    ambiguous arrow between whole cards), and 'good'/'bad'/'flat' drives the
    color so a parent doesn't have to remember whether up or down is good
    for K% vs. OBP. Deltas smaller than flat_epsilon are shown as flat
    rather than flip-flopping color on noise."""
    delta = curr - prev
    if abs(delta) < flat_epsilon:
        return {"arrow": "→", "css_class": "flat"}
    up = delta > 0
    good = up if higher_is_better else not up
    return {"arrow": "▲" if up else "▼", "css_class": "good" if good else "bad"}


def compute_trend(pergame_rows: list[dict]) -> Optional[list[dict]]:
    if not pergame_rows or len(pergame_rows) < 6:
        return None
    n = len(pergame_rows)
    third = n // 3
    chunks = [pergame_rows[:third], pergame_rows[third:2 * third], pergame_rows[2 * third:]]
    if any(len(c) == 0 for c in chunks):
        return None

    out = []
    labels = ["Early", "Mid", "Late"]
    prev = None
    for label, chunk in zip(labels, chunks):
        h = sum(r.get("h", 0) for r in chunk)
        bb = sum(r.get("bb", 0) for r in chunk)
        so = sum(r.get("so", 0) for r in chunk)
        hbp = sum(r.get("hbp", 0) for r in chunk)
        sf = sum(r.get("sf", 0) for r in chunk)
        sh = sum(r.get("sh", 0) for r in chunk)
        ab = sum(r.get("ab", 0) for r in chunk)
        pa = sum(r.get("pa", 0) for r in chunk) or (ab + bb + hbp + sf + sh) or (ab + bb)
        obp_denom = ab + bb + hbp + sf
        obp = ((h + bb + hbp) / obp_denom) if obp_denom else 0.0
        entry = {
            "label": label,
            "games": len(chunk),
            "obp": obp,
            "k_pct": (so / pa * 100) if pa else 0.0,
            "bb_pct": (bb / pa * 100) if pa else 0.0,
        }
        if prev is not None:
            # dead zones: <10 OBP points, or <1 percentage point for K%/BB%,
            # read as "flat" rather than a coin-flip arrow on noise.
            entry["obp_change"] = _trend_change(entry["obp"], prev["obp"], True, 0.010)
            entry["k_pct_change"] = _trend_change(entry["k_pct"], prev["k_pct"], False, 1.0)
            entry["bb_pct_change"] = _trend_change(entry["bb_pct"], prev["bb_pct"], True, 1.0)
        else:
            entry["obp_change"] = None
            entry["k_pct_change"] = None
            entry["bb_pct_change"] = None
        out.append(entry)
        prev = entry
    return out


# --------------------------------------------------------------------------
# Data-quality flags
# --------------------------------------------------------------------------

def build_dataquality_notes(flags: dict, config: dict) -> dict:
    notes = {
        "pa_approx_any": False,
        "obp_approx_any": False,
        "season_span_flag": None,
        "no_trend_reason": None,
        "merged_names_flag": None,
    }
    dup_names = flags.get("merged_duplicate_names")
    if dup_names:
        names_str = ", ".join(dup_names)
        notes["merged_names_flag"] = (
            f"{names_str} appeared more than once in this export (e.g. under a different jersey "
            "number) — those rows were combined into one player rather than counted as two people. "
            "Worth a quick check that this was one kid, not two teammates who happen to share a name."
        )
    date_min = flags.get("date_min")
    date_max = flags.get("date_max")
    if date_min is not None and date_max is not None:
        span_days = (date_max - date_min).days
        span_months = span_days / 30.44
        if config.get("season_label") and span_months > 5:
            notes["season_span_flag"] = (
                f"GameChanger's stats export was labeled \"{config['season_label']}\", "
                f"but the games in this file run from {date_min.strftime('%b %d, %Y')} to "
                f"{date_max.strftime('%b %d, %Y')} — about {span_months:.0f} months. "
                "Worth double-checking that's really one season and not several stitched together."
            )
    if not flags.get("is_pergame"):
        notes["no_trend_reason"] = (
            "This export has one row per player (season totals only), not one row per game, "
            "so no within-season trend could be computed."
        )
    return notes


# --------------------------------------------------------------------------
# Coach-mode actionable rollups — practice grouping, team trend, playing
# time. Everything here is a re-aggregation of numbers already computed per
# player (each player's own choose_focus()/compute_trend() output) rather
# than new coaching judgment — the goal is to save a coach from having to
# mentally cross-reference 15 individual reports to find the patterns.
# --------------------------------------------------------------------------

def build_practice_groups(all_players: list["PlayerStats"], per_player_ctx: dict) -> dict:
    """Clusters players by their single development focus stat (the same
    focus already shown on each player's own report), so a coach can run
    focused practice stations instead of guessing who needs what. Returns
    groups sorted largest-first, plus separate buckets for players whose
    focus is a 'stretch goal' (already strong — don't lump them in with a
    real weakness) and players with too few plate appearances to have a
    focus computed at all."""
    groups: dict[str, dict] = {}
    stretch = []
    no_focus_yet = []

    for p in all_players:
        ctx = per_player_ctx[p.name]
        if p.below_rate_stat_floor:
            no_focus_yet.append({"name": p.name, "pa": p.pa})
            continue
        focus = ctx.get("focus")
        if not focus:
            continue
        entry = {
            "name": p.name,
            "current_value": focus["current_value"],
            "distance_text": focus.get("distance_text"),
        }
        if focus.get("stretch"):
            stretch.append({**entry, "label": focus["label"]})
        else:
            bucket = groups.setdefault(focus["stat_key"], {"label": focus["label"], "players": []})
            bucket["players"].append(entry)

    sorted_groups = sorted(groups.values(), key=lambda g: len(g["players"]), reverse=True)
    return {"groups": sorted_groups, "stretch": stretch, "no_focus_yet": no_focus_yet}


def build_coach_player_summaries(all_players: list["PlayerStats"], per_player_ctx: dict) -> list[dict]:
    """One line per player of strength + what-to-work-on, reusing the exact
    same _best_established_strength()/choose_focus() output already computed
    for that player's own parent/player report — no new judgment invented
    here, just both halves surfaced together in one place so a coach doesn't
    have to open 15 separate files to get a read on the whole roster before
    practice. Sorted alphabetically (this section isn't a ranking)."""
    out = []
    for p in sorted(all_players, key=lambda p: p.name):
        ctx = per_player_ctx[p.name]
        if p.below_rate_stat_floor:
            out.append({
                "name": p.name,
                "strength_text": None,
                "focus_text": None,
                "no_data": True,
                "pa": p.pa,
            })
            continue

        strength = ctx.get("strength")
        if strength:
            strength_text = strength["phrase"][0].upper() + strength["phrase"][1:]
            if strength["tier"] != "leading":
                strength_text += f" ({strength['tier_label'].lower()})"
        else:
            strength_text = "No single reliable stat stands out above the rest yet."

        focus = ctx.get("focus")
        if focus:
            lead = "Stretch goal" if focus.get("stretch") else "Focus"
            focus_text = f"{lead}: {focus['label']} ({focus['current_value']})"
            if focus.get("distance_text"):
                focus_text += f" — {focus['distance_text'][0].lower() + focus['distance_text'][1:]}"
            else:
                focus_text += "."
        else:
            focus_text = None

        out.append({
            "name": p.name,
            "strength_text": strength_text,
            "focus_text": focus_text,
            "no_data": False,
            "pa": p.pa,
        })
    return out


def compute_team_trend(pergame_by_player: dict, dq_flags: dict) -> Optional[list[dict]]:
    """Team-wide version of compute_trend(): pools every player's per-game
    rows, sorted by real calendar date, into one season-thirds trend. Only
    possible when the export has real per-game dates (without them there's
    no honest way to align different players' games into the same 'early/
    mid/late' buckets) — returns None otherwise, same 'say nothing rather
    than fake it' rule as the per-player trend."""
    if not dq_flags.get("is_pergame") or "date_min" not in dq_flags:
        return None
    all_rows = []
    for rows in pergame_by_player.values():
        for r in rows:
            if r.get("date") is not None:
                all_rows.append(r)
    all_rows.sort(key=lambda r: r["date"])
    return compute_trend(all_rows)


def build_notable_trend_movers(all_players: list["PlayerStats"], per_player_ctx: dict) -> dict:
    """Surfaces the players whose on-base percentage moved the most from
    the first to the last third of the season, using each player's own
    already-computed trend — a re-sort of numbers already on their
    individual reports, not a new stat. A .030 (30-point) OBP move is the
    bar for 'notable' here, to keep small-sample noise out of a list a
    coach might actually act on."""
    movers = []
    for p in all_players:
        trend = per_player_ctx[p.name].get("trend")
        if not trend or len(trend) < 2:
            continue
        first, last = trend[0], trend[-1]
        movers.append({
            "name": p.name,
            "obp_delta": last["obp"] - first["obp"],
            "first_obp": first["obp"],
            "last_obp": last["obp"],
        })
    most_improved = sorted([m for m in movers if m["obp_delta"] > 0.030], key=lambda m: m["obp_delta"], reverse=True)[:2]
    worth_checkin = sorted([m for m in movers if m["obp_delta"] < -0.030], key=lambda m: m["obp_delta"])[:2]
    return {"most_improved": most_improved, "worth_checkin": worth_checkin}


def build_equity_flags(all_players: list["PlayerStats"], min_pa: float) -> list[dict]:
    """Players who've fallen under the 'regular' playing-time floor this
    stretch, surfaced explicitly rather than left for a coach to notice by
    scanning the PA column of a 15-row table."""
    under = [{"name": p.name, "pa": p.pa} for p in all_players if p.pa < min_pa]
    under.sort(key=lambda x: x["pa"])
    return under


# --------------------------------------------------------------------------
# Redaction pre-flight check
# --------------------------------------------------------------------------

def check_redaction(html: str, subject_name: str, all_names: list[str]) -> list[tuple[str, str]]:
    """Returns a list of (full_name, matched_variant) for every OTHER
    player whose name (full, or first name standing alone) appears anywhere
    in the rendered HTML. Empty list = clean.

    A shared first name is a real thing on youth rosters (three "Luke"s
    isn't rare) — every report legitimately uses the SUBJECT's own first
    name throughout ("Keep swinging, Luke."). So a teammate's first-name
    variant is only a leak if it doesn't also happen to be the subject's
    own first name; the teammate's full name is still always checked and
    always a leak if it appears."""
    leaks: list[tuple[str, str]] = []
    subject_parts = subject_name.strip().split()
    subject_first = subject_parts[0].lower() if subject_parts else ""
    for other in all_names:
        other_stripped = other.strip()
        if other_stripped.lower() == subject_name.strip().lower():
            continue
        other_parts = other_stripped.split()
        other_first = other_parts[0] if other_parts else other_stripped
        variants = {other_stripped, other_first}
        for variant in variants:
            if len(variant) < 3:
                continue
            if variant == other_first and variant.lower() == subject_first:
                continue
            pattern = r"\b" + re.escape(variant) + r"\b"
            if re.search(pattern, html, flags=re.IGNORECASE):
                leaks.append((other, variant))
                break
    return leaks


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "player"


def fmt3(v: float) -> str:
    if v >= 1:
        return f"{v:.3f}"
    return f".{round(v * 1000):03d}"


def build_env(templates_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt3"] = fmt3
    env.filters["fmt_inn"] = fmt_innings
    env.filters["pct1"] = lambda v: "—" if v is None else f"{v:.1f}%"
    env.filters["dec2"] = lambda v: "—" if v is None else f"{v:.2f}"
    env.filters["dec1"] = lambda v: "—" if v is None else f"{v:.1f}"
    env.filters["fmt3o"] = lambda v: "—" if v is None else fmt3(v)
    env.filters["reliability_label"] = lambda key: RELIABILITY_DISPLAY_LABELS.get(key, str(key).upper())
    env.filters["reliability_explain"] = lambda key: RELIABILITY_EXPLANATIONS.get(key, "")
    return env


def build_player_context(player: PlayerStats, comparisons: Optional[dict], focus: Optional[dict],
                          trend: Optional[list[dict]], team_ops: dict, dq_notes: dict,
                          config: dict, n_regulars: int, n_excluded: int, home_run_count_team: int,
                          n_players_team: int, compare_visuals: Optional[dict] = None,
                          visible: Optional[dict] = None, def_comparisons: Optional[dict] = None,
                          generated_date: str = "", pitching: Optional[dict] = None) -> dict:
    return {
        "config": config,
        "player": player,
        "cmp": comparisons,
        "def_cmp": def_comparisons,
        "pitch": pitching,
        "pitching_floor_bf": PITCHING_FLOOR_BF,
        "focus": focus,
        "trend": trend,
        "team_ops": team_ops,
        "dq": dq_notes,
        "n_regulars": n_regulars,
        "n_excluded": n_excluded,
        "home_run_count_team": home_run_count_team,
        "n_players_team": n_players_team,
        "visuals": compare_visuals,
        "rate_stat_floor": RATE_STAT_FLOOR,
        "defense_rate_stat_floor": DEFENSE_RATE_STAT_FLOOR,
        "bottom_line_mode": _bottom_line_mode(comparisons, team_ops),
        "strength": _best_established_strength(player, comparisons),
        "visible": visible or DEFAULT_VISIBLE,
        "generated_date": generated_date,
    }


_TIER_RANK = {"leading": 0, "above_middle": 1, "around_middle": 2, "below_middle": 3, "still_finding_it": 4}

STRENGTH_PHRASES = {
    "k_pct": "avoiding strikeouts",
    "bb_pct": "drawing walks",
    "bb_per_k": "plate discipline — walks compared to strikeouts",
    "avg": "getting hits",
    "obp": "getting on base",
    "slg": "extra-base power",
    "ops": "all-around hitting",
}


def _bottom_line_mode(comparisons: Optional[dict], team_ops: dict) -> str:
    """Which tone the closing "Bottom Line" paragraph should take:
      - "tough": genuinely below-average on a normal-strength roster —
        supportive, weakness-first framing (previous, only behavior).
      - "deep_roster": below-average only because the roster itself is
        unusually strong (team OPS at/above the dot-plot cutoff) — a solid
        season on paper shouldn't read as discouraging just because of who
        else is on the roster. A .455 OBP is a good season; it shouldn't
        get "tough stretch" copy just because three teammates are having
        historic ones.
      - "strong": at or above the middle of the roster.
    Splitting "below the middle" into two cases this way is exactly the
    correction a real reference report made by hand (see design notes) —
    this generalizes it so it happens automatically for any roster."""
    if not comparisons:
        return "strong"
    low_tiers = {"below_middle", "still_finding_it"}
    vals = [c.tier for c in comparisons.values()]
    is_low_majority = sum(1 for t in vals if t in low_tiers) >= (len(vals) + 1) // 2
    if not is_low_majority:
        return "strong"
    return "tough" if team_ops.get("use_bars", True) else "deep_roster"


def _best_established_strength(player: PlayerStats, comparisons: Optional[dict]) -> Optional[dict]:
    """The single stat with the most favorable roster-relative tier, among
    stats that are actually reliable enough to trust — used to name a real
    strength in the Bottom Line before getting into what to build on next,
    the way a coach summing up a season out loud would. Returns None if
    nothing reliable is even around-the-middle or better (nothing false or
    inflated to report)."""
    if not comparisons:
        return None
    reliable = [(k, c) for k, c in comparisons.items() if player.reliability.get(k) == "reliable"]
    pool = reliable or list(comparisons.items())
    if not pool:
        return None
    best_key, best_cmp = min(pool, key=lambda kv: _TIER_RANK[kv[1].tier])
    if best_cmp.tier not in ("leading", "above_middle", "around_middle"):
        return None
    return {
        "key": best_key,
        "phrase": STRENGTH_PHRASES.get(best_key, best_key),
        "tier": best_cmp.tier,
        "tier_label": best_cmp.tier_label,
        "reliable": player.reliability.get(best_key) == "reliable",
    }


def build_player_list(season_df: pd.DataFrame, config: dict) -> tuple[list[PlayerStats], list[PlayerStats], dict]:
    """Computes every player's stats + reliability from the aggregated
    season DataFrame, and splits them into (all_players, regulars,
    team_ops). Shared by render_all() and the web app's Q&A endpoint so both
    always agree on the exact same numbers."""
    all_players: list[PlayerStats] = []
    for _, row in season_df.iterrows():
        p = compute_player_stats(row, config.get("era_innings", 6))
        attach_reliability(p)
        all_players.append(p)

    min_pa = config["min_pa_regular"]
    regulars = [p for p in all_players if p.pa >= min_pa]
    for p in all_players:
        p.is_regular = p in regulars

    team_ops = team_ops_audit(regulars) if regulars else {"avg": 0, "obp": 0, "slg": 0, "ops": 0, "use_bars": True, "n_regulars": 0}
    return all_players, regulars, team_ops


def render_all(season_df: pd.DataFrame, pergame_by_player: dict, dq_flags: dict, config: dict,
                templates_dir: Path, out_dir: Path, modes: list[str],
                visible: Optional[dict] = None) -> None:
    env = build_env(templates_dir)
    config = {**config, "era_innings": dq_flags.get("era_innings", config.get("era_innings", 6))}

    all_players, regulars, team_ops = build_player_list(season_df, config)
    min_pa = config["min_pa_regular"]
    n_excluded = len(all_players) - len(regulars)

    dq_notes = build_dataquality_notes(dq_flags, config)
    home_run_count_team = sum(1 for p in all_players if p.hr > 0)

    # Defensive participation doesn't track plate appearances — a low-PA
    # bench kid can still catch every game, and a high-PA leadoff hitter can
    # have barely any fielding chances — so this is its own pool, not the
    # batting "regulars" list.
    defensive_pool = [p for p in all_players if p.has_fielding_data and not p.below_defense_floor]
    pitch_pool = [p for p in all_players if p.has_pitching_data and not p.below_pitching_floor]

    now = datetime.now()
    generated_date = f"{now:%B} {now.day}, {now:%Y}"

    per_player_ctx: dict[str, dict] = {}
    for p in all_players:
        if p.below_rate_stat_floor:
            comparisons = None
            focus = None
            visuals = None
        else:
            pool = regulars if p.is_regular else regulars + [p]
            comparisons = build_comparisons(p, pool)
            focus = choose_focus(p, comparisons, pool)
            visuals = build_compare_visuals(p, pool, team_ops)
        trend = compute_trend(pergame_by_player.get(p.name, [])) if not p.below_rate_stat_floor else None
        if p.has_fielding_data and not p.below_defense_floor:
            def_comparisons = build_defensive_comparisons(p, defensive_pool)
        else:
            def_comparisons = None
        per_player_ctx[p.name] = build_player_context(
            p, comparisons, focus, trend, team_ops, dq_notes, config,
            len(regulars), n_excluded, home_run_count_team, len(all_players),
            compare_visuals=visuals, visible=visible, def_comparisons=def_comparisons,
            generated_date=generated_date,
            pitching=build_pitching_context(p, pitch_pool) if p in pitch_pool else None,
        )

    all_names = [p.name for p in all_players]

    # ---- Pass 1: render every parent/player file into memory, redaction-check ALL of them,
    # and only write anything to disk if the whole batch is clean. ----
    rendered: list[tuple[Path, str]] = []
    leak_report: list[str] = []

    for mode in modes:
        if mode not in ("parent", "player"):
            continue
        template = env.get_template(f"{mode}.html")
        for p in all_players:
            ctx = per_player_ctx[p.name]
            html = template.render(**ctx)
            leaks = check_redaction(html, p.name, all_names)
            if leaks:
                for other, variant in leaks:
                    leak_report.append(
                        f"  [{mode}] {p.name}'s report contains \"{variant}\" (matches teammate {other})"
                    )
            else:
                out_path = out_dir / f"{slugify(p.name)}-{mode}.html"
                rendered.append((out_path, html))

    if leak_report:
        raise ReportBuildError(
            "REDACTION CHECK FAILED — build aborted, nothing was written.\n"
            "The following individual reports mention a teammate by name:\n"
            + "\n".join(leak_report)
            + "\n\nFix the template (or the data) and re-run."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for out_path, html in rendered:
        out_path.write_text(html, encoding="utf-8")
        print(f"  wrote {out_path}")

    # ---- coach mode: names + ordinal ranks allowed, one internal file ----
    if "coach" in modes:
        coach_rows = []
        # ordinal ranks computed on regulars, by OPS descending, for the roster table
        ranked = sorted(regulars, key=lambda p: p.ops, reverse=True)
        rank_of = {p.name: i + 1 for i, p in enumerate(ranked)}
        for p in all_players:
            coach_rows.append({
                "player": p,
                "rank": rank_of.get(p.name),
                "reliability": p.reliability,
            })
        coach_rows_sorted = sorted(
            coach_rows,
            key=lambda r: (r["rank"] is None, r["rank"] if r["rank"] is not None else 0, r["player"].name),
        )
        coach_ctx = {
            "config": config,
            "rows": coach_rows_sorted,
            "team_ops": team_ops,
            "dq": dq_notes,
            "n_regulars": len(regulars),
            "n_excluded": n_excluded,
            "n_players_team": len(all_players),
            "min_pa": min_pa,
            "rate_stat_floor": RATE_STAT_FLOOR,
            "practice_groups": build_practice_groups(all_players, per_player_ctx),
            "player_summaries": build_coach_player_summaries(all_players, per_player_ctx),
            "team_trend": compute_team_trend(pergame_by_player, dq_flags),
            "movers": build_notable_trend_movers(all_players, per_player_ctx),
            "equity_flags": build_equity_flags(all_players, min_pa),
            "generated_date": generated_date,
            "defense_rows": sorted(
                (p for p in all_players if p.has_fielding_data),
                key=lambda p: p.def_tc, reverse=True,
            ),
            "defense_rate_stat_floor": DEFENSE_RATE_STAT_FLOOR,
            "pitching_rows": sorted(
                (p for p in all_players if p.has_pitching_data),
                key=lambda p: p.p_ip, reverse=True,
            ),
            "pitching_floor_bf": PITCHING_FLOOR_BF,
            "team_pitch_ip": sum(p.p_ip for p in all_players if p.has_pitching_data),
            "team_pitch_np": sum(p.p_np for p in all_players if p.has_pitching_data),
            "pitch_goals": {p.name: {k: pitching_goal_tier(k, p) for k in PITCHING_GOALS}
                            for p in all_players if p.has_pitching_data},
        }
        html = env.get_template("coach.html").render(**coach_ctx)
        out_path = out_dir / "coach-summary.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"  wrote {out_path}")

        csv_path = out_dir / "coach-summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Rank", "Player", "PA", "AB", "AVG", "OBP", "SLG", "OPS", "K%", "BB%", "BB/K",
                        "Reliability(K%)", "Reliability(BB%)", "Reliability(AVG)", "Is Regular"])
            for row in sorted(coach_rows, key=lambda r: (r["rank"] is None, r["rank"] or 0)):
                p = row["player"]
                w.writerow([
                    row["rank"] or "", p.name, int(p.pa), int(p.ab), fmt3(p.avg), fmt3(p.obp), fmt3(p.slg),
                    fmt3(p.ops), f"{p.k_pct:.1f}%", f"{p.bb_pct:.1f}%",
                    f"{p.bb_per_k:.2f}" if p.bb_per_k is not None else "-",
                    p.reliability.get("k_pct", ""), p.reliability.get("bb_pct", ""), p.reliability.get("avg", ""),
                    "yes" if p.is_regular else "no",
                ])
        print(f"  wrote {csv_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate GameChanger batting reports from a Staff CSV export.")
    parser.add_argument("csv_path", type=Path, help="Path to the GameChanger season stats export CSV.")
    parser.add_argument("--mode", choices=["parent", "player", "coach", "all"], default=None,
                         help="Which report(s) to generate. Default: value of CONFIG['output_mode'].")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory (default: ./output).")
    parser.add_argument("--team-name", default=None, help="Override CONFIG['team_name'].")
    parser.add_argument("--season-label", default=None, help="Override CONFIG['season_label'].")
    parser.add_argument("--min-pa", type=int, default=None, help="Override CONFIG['min_pa_regular'].")
    parser.add_argument("--templates-dir", type=Path, default=None,
                         help="Templates directory (default: ./templates next to this script).")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the interactive column-mapping confirmation.")
    args = parser.parse_args(argv)

    if not args.csv_path.exists():
        sys.exit(f"ERROR: {args.csv_path} does not exist.")

    config = dict(CONFIG)
    if args.team_name:
        config["team_name"] = args.team_name
    if args.season_label:
        config["season_label"] = args.season_label
    if args.min_pa is not None:
        config["min_pa_regular"] = args.min_pa

    mode = args.mode or config["output_mode"]
    modes = ["parent", "player", "coach"] if mode == "all" else [mode]

    templates_dir = args.templates_dir or (Path(__file__).parent / "templates")
    if not templates_dir.exists():
        sys.exit(f"ERROR: templates directory not found at {templates_dir}")

    header_row = detect_header_row(args.csv_path)
    with args.csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        skipped_rows = [next(reader) for _ in range(header_row)]
        headers = next(reader)
    section_row = skipped_rows[0] if skipped_rows else None
    if header_row > 0:
        print(f"Detected a section-label row above the real header — using row {header_row + 1} as the header.")
    mapping = detect_columns(headers)
    mapping.update(detect_fielding_columns(headers, section_row))
    if any(mapping[k] for k in FIELDING_COLUMN_CANDIDATES):
        print("Detected a Fielding section — defensive stats will be included where available.")
    mapping.update(detect_pitching_columns(headers, section_row))
    if any(mapping[k] for k in PITCHING_COLUMN_CANDIDATES):
        print("Detected a Pitching section — pitching stats will be included where available.")
    mapping = confirm_mapping(headers, mapping, args.yes)

    try:
        season_df, pergame_by_player, dq_flags = load_and_aggregate(args.csv_path, mapping, args.yes, header_row=header_row)

        print(f"\nLoaded {len(season_df)} players. Per-game data available: {dq_flags.get('is_pergame')}.")
        print(f"Generating modes: {', '.join(modes)}\n")

        render_all(season_df, pergame_by_player, dq_flags, config, templates_dir, args.out, modes)
    except ReportBuildError as e:
        sys.exit(f"ERROR: {e}")

    print(f"\nDone. Output in {args.out.resolve()}")


if __name__ == "__main__":
    main()
