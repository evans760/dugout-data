"""
browser_api.py — the in-browser stand-in for webapp.py.

This runs inside Pyodide (Python compiled to WebAssembly) in the visitor's
own browser. It exposes the same four operations the FastAPI app did —
upload, confirm, ask, generate — but there is no server: the CSV, the parsed
roster and the generated reports all live in Pyodide's in-memory filesystem
and vanish when the tab is closed. Nothing is sent over the network.

Every function takes/returns JSON strings so the JavaScript side can treat
them exactly like the old fetch() responses ({"_status": 4xx, "detail": ...}
on error).
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

import generate_reports as gr
import query_engine as qe

WORK = Path("/work")
CSV_PATH = WORK / "upload.csv"
OUT_DIR = WORK / "output"
ZIP_PATH = WORK / "dugout-data-reports.zip"
TEMPLATES_DIR = Path("/app/templates")
ALLOWED_MODES = {"parent", "player", "coach"}
TEAM_FILE_LABELS = {
    "coach-summary.html": "Coach Summary (internal)",
}

STATE: dict = {}


def _err(detail: str, status: int = 400) -> str:
    return json.dumps({"_status": status, "detail": detail})


def upload(filename: str) -> str:
    """The CSV bytes have already been written to CSV_PATH by JavaScript."""
    STATE.clear()
    if not filename.lower().endswith(".csv"):
        return _err("Please upload a .csv file exported from GameChanger.")
    try:
        header_row = gr.detect_header_row(CSV_PATH)
        with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            skipped_rows = [next(reader) for _ in range(header_row)]
            headers = next(reader)
    except Exception as exc:  # noqa: BLE001 — surface any parse failure to the user
        return _err(f"Couldn't read that as a CSV file ({exc}).")
    if not headers:
        return _err("That CSV has no header row.")

    section_row = skipped_rows[0] if skipped_rows else None
    mapping = gr.detect_columns(headers)
    mapping.update(gr.detect_fielding_columns(headers, section_row))
    STATE.update({
        "header_row": header_row,
        "fielding_mapping": {k: v for k, v in mapping.items() if k in gr.FIELDING_COLUMN_CANDIDATES},
    })
    return json.dumps({
        "session_id": "local",
        "headers": headers,
        "mapping": mapping,
        "missing_required": gr.missing_required_columns(mapping),
        "candidate_fields": list(gr.COLUMN_CANDIDATES.keys()),
        "header_row_skipped": header_row,
        "has_fielding": any(mapping[k] for k in gr.FIELDING_COLUMN_CANDIDATES),
    })


def confirm(body: str) -> str:
    req = json.loads(body)
    if "header_row" not in STATE:
        return _err("Please upload your CSV again.", 404)
    mapping = req.get("mapping") or {}
    missing = gr.missing_required_columns(mapping)
    if missing:
        return _err(
            "This CSV is missing columns required for batting reports: " + ", ".join(missing)
            + ". Re-export from GameChanger (Team -> Stats -> Export Stats) with those columns included."
        )
    full_mapping = dict(mapping)
    full_mapping.update(STATE["fielding_mapping"])
    try:
        season_df, pergame_by_player, flags = gr.load_and_aggregate(
            CSV_PATH, full_mapping, True, STATE["header_row"]
        )
    except gr.ReportBuildError as exc:
        return _err(str(exc))

    config = dict(gr.CONFIG)
    config["team_name"] = req.get("team_name") or config["team_name"]
    config["season_label"] = req.get("season_label")
    config["min_pa_regular"] = int(req.get("min_pa") or 15)
    STATE.update({
        "season_df": season_df,
        "pergame_by_player": pergame_by_player,
        "flags": flags,
        "config": config,
    })
    return json.dumps({
        "n_players": int(len(season_df)),
        "is_pergame": bool(flags.get("is_pergame")),
        "player_names": sorted(season_df["name"].astype(str).tolist()),
    })


def ask(body: str) -> str:
    req = json.loads(body)
    if "season_df" not in STATE:
        return _err("Load a roster before asking questions.")
    all_players, regulars, _team_ops = gr.build_player_list(STATE["season_df"], STATE["config"])
    defensive_pool = [p for p in all_players if p.has_fielding_data and not p.below_defense_floor]
    answer = qe.answer_question(req.get("question", ""), all_players, regulars, defensive_pool, STATE["config"])
    return json.dumps({"answer": answer})


def generate(body: str) -> str:
    """Renders reports into OUT_DIR. Returned "url" values are FS paths;
    the JavaScript side swaps each for a blob: URL of the file's contents."""
    req = json.loads(body)
    if "season_df" not in STATE:
        return _err("Load a roster before generating reports.")
    modes = [m for m in req.get("modes", []) if m in ALLOWED_MODES] or ["parent", "player"]
    visible = dict(gr.DEFAULT_VISIBLE)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    try:
        gr.render_all(
            STATE["season_df"], STATE["pergame_by_player"], STATE["flags"], STATE["config"],
            TEMPLATES_DIR, OUT_DIR, modes, visible,
        )
    except gr.ReportBuildError as exc:
        return _err(str(exc))

    slug_to_name = {gr.slugify(n): n for n in STATE["season_df"]["name"].astype(str)}
    player_reports: dict[str, dict] = {}
    team_reports = []
    files = sorted(p for p in OUT_DIR.iterdir() if p.suffix == ".html") if OUT_DIR.exists() else []
    for p in files:
        matched = False
        for mode in ("parent", "player"):
            suffix = f"-{mode}.html"
            if p.name.endswith(suffix):
                slug = p.name[: -len(suffix)]
                name = slug_to_name.get(slug, slug.replace("-", " ").title())
                player_reports.setdefault(slug, {"name": name, "reports": {}})["reports"][mode] = str(p)
                matched = True
                break
        if not matched and p.name != "index.html":
            team_reports.append({"label": TEAM_FILE_LABELS.get(p.name, p.name), "url": str(p)})

    # One zip of everything, so reports survive after the tab is closed.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT_DIR.iterdir()):
            if p.is_file():
                zf.write(p, p.name)
    ZIP_PATH.write_bytes(buf.getvalue())

    return json.dumps({
        "visible": visible,
        "players": sorted(player_reports.values(), key=lambda e: e["name"]),
        "team_reports": team_reports,
        "zip": str(ZIP_PATH),
    })
