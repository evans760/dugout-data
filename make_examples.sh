#!/bin/sh
# Rebuilds examples/*.html (the "See example reports" page) from the
# made-up sample roster (batting, fielding and pitching). Run after changing anything in templates/ or py/.
# Uses the Python environment from the private gc_report_tool project.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-$HOME/gc_report_tool/venv/bin/python}"
SAMPLE_PLAYER="ava-thompson"
TMP="$(mktemp -d)"
"$PY" py/generate_reports.py sample/synthetic_full_roster.csv --mode all \
  --templates-dir templates --team-name "Sample Team" --season-label "Spring 2026" \
  --out "$TMP" -y >/dev/null
mkdir -p examples
cp "$TMP/$SAMPLE_PLAYER-parent.html" examples/parent.html
cp "$TMP/$SAMPLE_PLAYER-player.html" examples/player.html
cp "$TMP/coach-summary.html" examples/coach.html
rm -rf "$TMP"
echo "Rebuilt examples/ (parent, player, coach)"
