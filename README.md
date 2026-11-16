# Batting Report Generator (browser version)

Turns a GameChanger season stats CSV into per-player batting reports
(parent and player versions), a coach summary, and an org leave-behind.

**Everything runs in your browser.** The page uses
[Pyodide](https://pyodide.org) to run the report code (Python) locally.
Your CSV is never uploaded, and no stats leave your device.

## Use it

Open the site, drop in a GameChanger Staff export
(Team → Stats → Export Stats), confirm the columns, and click
**Generate reports**. Use **Download all reports (.zip)** to keep them.

## Embed it (e.g. Squarespace)

```html
<iframe src="https://evans760.github.io/batting-reports/"
  title="Batting Report Generator"
  style="width:100%;height:1600px;border:0;"></iframe>
```

## What's here

- `index.html` is the page and the in-browser engine loader.
- `py/` holds the report code: `generate_reports.py` and `query_engine.py`
  (shared with the CLI/server version), plus `browser_api.py` (the browser glue).
- `templates/` holds the report HTML templates.
- `sample/synthetic_roster.csv` is a made-up roster for testing. It has no real players.
