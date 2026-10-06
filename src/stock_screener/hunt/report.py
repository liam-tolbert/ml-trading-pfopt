"""Weekend-hunt HTML report, built from a hunt directory's persisted state
(diagnostics.csv + verdicts.csv + meta.json). Self-contained single file:
Google-Fonts faces with real fallbacks, light/dark via tokens, no JS deps.

Two optional inputs extend it: ``narrative.md`` (the reviewer's written read, rendered
under "Reviewer's read") and ``charts/`` (each review sheet appended under its verdicts).
``mirror_report`` copies the finished page and its sheets to the deliverable folder,
``docs/hunt/<date>/`` by default. The print stylesheet lays the page out for a browser's
own print.
"""
from __future__ import annotations

import csv
import html as _html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import pipeline as pl

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs" / "hunt"
NARRATIVE_MD = "narrative.md"
CATALYST_JSON = "catalyst.json"  # written by the reviewer from the hunt's news/ files
SHEETS_JSON = "sheets.json"     # written by charts.render_sheets: sheet file -> tickers
PER_FIG = 4                     # tickers per review sheet; charts.render_sheets reads it here


def _esc(s) -> str:
    return _html.escape(str(s), quote=True)


# ---- the reviewer's Markdown ---------------------------------------------- #
_INLINE = ((re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
           (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
           (re.compile(r"(?<![\w*])\*([^*\n]+?)\*(?!\w)"), r"<i>\1</i>"))
_TABLE_RULE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _inline(s: str) -> str:
    s = _esc(s)
    for rx, rep in _INLINE:
        s = rx.sub(rep, s)
    return s


def md_to_html(text: str) -> str:
    """Render the Markdown subset the reviewer writes: ``#``/``##``/``###`` headings
    (as h3/h4/h5), paragraphs, ``-``/``*`` bullets, ``1.`` lists, pipe tables, ``---``,
    and inline ``**bold**``, ``*em*`` and ``code``. Text is escaped first, so the
    narrative cannot carry markup of its own."""
    out: List[str] = []
    para: List[str] = []
    table: List[str] = []
    lst: Optional[str] = None

    def flush_para():
        if para:
            out.append(f"<p>{' '.join(_inline(x) for x in para)}</p>")
            para.clear()

    def flush_list():
        nonlocal lst
        if lst:
            out.append(f"</{lst}>")
            lst = None

    def flush_table():
        if not table:
            return
        rows = [[c.strip() for c in r.strip().strip("|").split("|")]
                for r in table if not _TABLE_RULE.match(r)]
        head, body = rows[0], rows[1:]
        cells = "".join(f"<th>{_inline(c)}</th>" for c in head)
        trs = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                      for r in body)
        out.append(f'<div class="scroll"><table><tr>{cells}</tr>{trs}</table></div>')
        table.clear()

    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush_para(); flush_list(); flush_table()
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            flush_para(); flush_list(); flush_table()
            level = len(m.group(1)) + 2
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            continue
        if re.match(r"^-{3,}$", s.strip()):
            flush_para(); flush_list(); flush_table()
            out.append("<hr>")
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", s)
        kind = "ul" if m else None
        if not m:
            m = re.match(r"^\s*\d+[.)]\s+(.*)$", s)
            kind = "ol" if m else None
        if m:
            flush_para(); flush_table()
            if lst != kind:
                flush_list()
                out.append(f"<{kind}>")
                lst = kind
            out.append(f"<li>{_inline(m.group(1))}</li>")
            continue
        if s.lstrip().startswith("|"):
            flush_para(); flush_list()
            table.append(s)
            continue
        flush_list(); flush_table()
        para.append(s.strip())
    flush_para(); flush_list(); flush_table()
    return "".join(out)


def review_sheets(hunt_path: Path, diag_rows: List[dict]) -> List[Tuple[Path, Optional[List[str]]]]:
    """``(png, tickers)`` per review sheet in ``charts/``, in sheet order. ``tickers`` is
    None when the sheet's contents cannot be established: no ``sheets.json`` and a sheet
    count the default sheet size does not account for."""
    charts = hunt_path / "charts"
    index = charts / SHEETS_JSON
    if index.exists():
        named: Dict[str, List[str]] = json.loads(index.read_text(encoding="utf-8"))
        return [(charts / name, tk) for name, tk in named.items() if (charts / name).exists()]
    pngs = sorted(charts.glob("sheet_*.png"))
    tickers = [r["ticker"] for r in diag_rows]
    if len(pngs) != -(-len(tickers) // PER_FIG):
        return [(p, None) for p in pngs]
    return [(p, tickers[i * PER_FIG:(i + 1) * PER_FIG]) for i, p in enumerate(pngs)]


def _vcls(v: str) -> str:
    return {"PASS": "p", "PASS-": "c", "FAIL": "f"}.get(v, "o")


def _vlabel(v: str) -> str:
    return {"PASS": "PASS", "PASS-": "PASS&middot;", "FAIL": "FAIL"}.get(v, _esc(v).upper())


def _f(x, fmt="{:.2f}", dash="-"):
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return dash


def _opt_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_catalysts(hunt_path: Path) -> Dict[str, dict]:
    """The reviewer's catalyst reads from ``catalyst.json``: ``{ticker: {category,
    sentiment, summary, sources, read_at}}``. A missing, unreadable or malformed file
    reads as ``{}``; an entry that is not a dict is dropped. Never raises."""
    try:
        data = json.loads((hunt_path / CATALYST_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(t).upper(): v for t, v in data.items() if isinstance(v, dict)}


def _catalyst_html(c: Optional[dict]) -> str:
    """One line under a row's notes: category, sentiment and the summary."""
    if not c:
        return ""
    head = " &middot; ".join(_esc(x) for x in (c.get("category"), c.get("sentiment")) if x)
    body = _esc(c.get("summary") or "")
    sep = " &mdash; " if head and body else ""
    return f'<div class="cat"><b>{head}</b>{sep}{body}</div>' if head or body else ""


def _mini_table(rows: List[dict], verdicts: Dict[str, dict],
                catalysts: Optional[Dict[str, dict]] = None) -> str:
    tr = []
    for r in rows:
        star = ' <span class="wl" title="on watchlist">&#9733;</span>' if int(r.get("wl") or 0) else ""
        note = (verdicts.get(r["ticker"]) or {}).get("notes", "")
        vs = float(r["vs_pivot_pct"])
        cat = _catalyst_html((catalysts or {}).get(r["ticker"]))
        tr.append(
            f'<tr><td class="tk">{_esc(r["ticker"])}{star}</td>'
            f'<td class="n">{_f(r["q"], "{:.0f}")}</td><td class="n">{r["rs"]}</td>'
            f'<td class="n">{_f(r["close"])}</td><td class="n">{_f(r["pivot"])}</td>'
            f'<td class="n {"pos" if vs >= 0 else "neg"}">{vs:+.1f}%</td>'
            f'<td class="n">{_f(r["adv_musd"], "{:.1f}")}</td>'
            f'<td class="n">{_f(r["volume_ratio"], "{:.2f}")}&times;</td>'
            f'<td class="note">{_esc(note)}{cat}</td></tr>')
    head = ('<tr><th>Ticker</th><th class="n">Q</th><th class="n">RS</th><th class="n">Close</th>'
            '<th class="n">Pivot</th><th class="n">vs piv</th><th class="n">ADV$M</th>'
            '<th class="n">Fri vol</th><th>Chart notes</th></tr>')
    body = "".join(tr) or '<tr><td colspan="9" class="dim">none</td></tr>'
    return f'<div class="scroll"><table>{head}{body}</table></div>'


@dataclass
class HuntState:
    diag_rows: List[dict]           # diagnostics.csv rows, numeric columns coerced
    verdicts: Dict[str, dict]
    meta: dict
    n: Dict[str, int]               # verdict -> count
    passing: List[dict]             # rows with verdict PASS
    buckets: Dict[str, List[dict]]  # buy_zone / approaching / below / past_entry
    blocked: List[dict]             # PASS, earnings inside the block window
    confirmed: List[dict]           # PASS, volume-confirmed breakouts
    audit: List[dict]               # pipeline.watchlist_audit cards
    gated: List[str]                # buy-zone tickers clearing ``min_fund``


def load_state(hunt_path: Path, min_fund: int = 0) -> HuntState:
    """Read a hunt directory into the state every report format renders.

    ``hunt_path`` MUST hold diagnostics.csv and meta.json; a missing verdicts.csv reads
    as no verdicts. ``min_fund`` filters ``gated`` only."""
    with open(hunt_path / "diagnostics.csv", encoding="utf-8") as f:
        diag_rows = list(csv.DictReader(f))
    verdicts = pl.read_verdicts(hunt_path / "verdicts.csv")
    meta = json.loads((hunt_path / "meta.json").read_text(encoding="utf-8"))

    # Numeric round-trip from CSV. An empty cell is a None the pipeline wrote (no ADV on a
    # dead tape) and MUST read back as None. A hunt dir from before the eight-check columns
    # existed MUST still load: the missing checks read as not passed.
    for r in diag_rows:
        for k in ("q", "close", "pivot", "vs_pivot_pct", "volume_ratio"):
            r[k] = float(r[k])
        for k in ("adv_musd", "stop", "max_order_usd"):
            r[k] = _opt_float(r.get(k))
        for k in ("rs", "fund", "wl", "dist_days", "breakout_today",
                  "f_rev", "f_eps", "f_accel", "f_margin"):
            r[k] = int(float(r[k]))
        for k, default in (("f_max", 8), ("f_code33", 0), ("f_fy", 0), ("f_est", 0),
                           ("f_react", 0)):
            r[k] = int(float(r.get(k) or default))

    n = {"PASS": 0, "PASS-": 0, "FAIL": 0}
    for v in verdicts.values():
        if v["verdict"] in n:
            n[v["verdict"]] += 1

    import pandas as pd
    diag_df = pd.DataFrame(diag_rows)

    # The buckets MUST come from pipeline.gates, so every report and the `gates` CLI agree.
    # min_fund=0 because a report shows every PASS name; the fundamental gate applies
    # only to the summary line.
    passing = [r for r in diag_rows if (verdicts.get(r["ticker"]) or {}).get("verdict") == "PASS"]
    g = pl.gates(diag_df, verdicts, min_fund=0)
    by_ticker = {r["ticker"]: r for r in diag_rows}
    return HuntState(
        diag_rows=diag_rows, verdicts=verdicts, meta=meta, n=n, passing=passing,
        buckets={k: [by_ticker[x["ticker"]] for x in g[k]]
                 for k in ("buy_zone", "approaching", "below", "past_entry")},
        blocked=[by_ticker[x["ticker"]] for x in g["earnings_blocked"]],
        confirmed=[by_ticker[x["ticker"]] for x in g["volume_confirmed"]],
        audit=pl.watchlist_audit(diag_df, verdicts),
        gated=[x["ticker"] for x in pl.gates(diag_df, verdicts, min_fund=min_fund)["buy_zone"]],
    )


def build_report(hunt_path: Path, min_fund: int = 0) -> Path:
    st = load_state(hunt_path, min_fund)
    diag_rows, verdicts, meta, n = st.diag_rows, st.verdicts, st.meta, st.n
    passing, buckets, blocked = st.passing, st.buckets, st.blocked
    confirmed, audit, gated = st.confirmed, st.audit, st.gated

    # ---- fragments -------------------------------------------------------- #
    def chk(b): return ('<td class="n chk-y">&#10003;</td>' if b
                        else '<td class="n chk-n">&ndash;</td>')
    fund_tr = "".join(
        f'<tr><td class="tk">{_esc(r["ticker"])}</td>'
        f'<td>{ {"buy_zone": "buy zone", "approaching": "approaching", "below": "below pivot", "past_entry": "past entry"}[pl.bucket(r["vs_pivot_pct"])] }</td>'
        f'<td class="n"><b>{r["fund"]}</b>/{r["f_max"]}</td>'
        + chk(r["f_rev"]) + chk(r["f_eps"]) + chk(r["f_accel"]) + chk(r["f_margin"])
        + chk(r["f_code33"]) + chk(r["f_fy"]) + chk(r["f_est"]) + chk(r["f_react"]) +
        f'<td class="n">{_f(r["rev_yoy"], "{:+.1f}%")}</td>'
        f'<td class="n">{_f(r["eps_yoy"], "{:+.1f}%")}</td>'
        f'<td class="n">{_f(r.get("code33"), "{:.0f}/3")}</td>'
        f'<td class="n">{"&#9888;" if r.get("inv_flag") == "True" else ""}</td>'
        f'<td class="n">{_f(r.get("earn_react"), "{:+.1f}%")}'
        f'{" &#9888;" if r.get("earn_flag") == "hard_drop" else ""}</td>'
        f'<td class="n">{_f(r.get("est_rev_90d"), "{:+.1f}%")}</td>'
        f'<td class="n">{_f(r.get("inst_count"), "{:.0f}")}</td></tr>'
        for r in sorted(passing, key=lambda r: -r["fund"]))

    ern_tr = "".join(
        f'<tr><td class="tk">{_esc(r["ticker"])}</td>'
        f'<td class="n">{int(float(r["earnings_in"]))}</td>'
        f'<td class="note">{_esc((verdicts.get(r["ticker"]) or {}).get("notes", ""))}</td></tr>'
        for r in sorted(blocked, key=lambda r: int(float(r["earnings_in"]))))

    wl_cards = "".join(
        f'<div class="wlc"><span class="tk">{_esc(c["ticker"])}</span>'
        f'<span class="pill {_vcls(c["state"])}">{_vlabel(c["state"]) if c["state"] in ("PASS", "PASS-", "FAIL") else _esc(c["state"]).replace("_", " ").upper()}</span>'
        f'<span class="wln">{_esc(c.get("note", ""))}</span></div>'
        for c in audit)

    full_tr = "".join(
        (lambda v:
         f'<tr data-v="{_vcls(v)}" data-t="{_esc(r["ticker"].lower())}">'
         f'<td class="n dim">{r["rank"]}</td>'
         f'<td class="tk">{_esc(r["ticker"])}{" &#9733;" if r["wl"] else ""}</td>'
         f'<td class="dim">{_esc(r.get("industry") or "-")}</td>'
         f'<td><span class="pill {_vcls(v)}">{_vlabel(v)}</span></td>'
         f'<td class="n">{_f(r["q"], "{:.0f}")}</td><td class="n">{r["rs"]}</td>'
         f'<td>{_esc(r.get("rs_trend") or "-")}</td>'
         f'<td class="n">{_f(r.get("sma200_m"), "{:.1f}")}</td>'
         f'<td class="n">{r["fund"]}</td>'
         f'<td class="n">{_f(r["close"])}</td><td class="n">{_f(r["pivot"])}</td>'
         f'<td class="n {"pos" if r["vs_pivot_pct"] >= 0 else "neg"}">{r["vs_pivot_pct"]:+.1f}%</td>'
         f'<td class="n">{_f(r["adv_musd"], "{:.1f}")}</td>'
         f'<td class="n">{_f(r.get("max_order_usd"), "{:,.0f}")}</td>'
         f'<td class="n">{r["dist_days"]}</td>'
         f'<td class="mono dim">{_esc(r["depths"])}</td>'
         f'<td class="n">{_f(r.get("depth_vs_spy"), "{:.1f}")}</td>'
         f'<td class="mono dim">{_esc(r.get("step3") or "-")}</td>'
         f'<td class="note">{_esc((verdicts.get(r["ticker"]) or {}).get("notes", ""))}</td></tr>'
         )((verdicts.get(r["ticker"]) or {}).get("verdict", "unreviewed"))
        for r in diag_rows)

    from collections import Counter
    _groups = Counter(r.get("industry") for r in passing if r.get("industry"))
    groups_line = (" &middot; ".join(f"{_esc(k)} {v}" for k, v in _groups.most_common(6))
                   if _groups else "no industry labels in this scan")

    catalysts = load_catalysts(hunt_path)
    narrative_path = hunt_path / NARRATIVE_MD
    narrative = ""
    if narrative_path.exists():
        narrative = ('<h2>Reviewer&rsquo;s read</h2><div class="narr">'
                     + md_to_html(narrative_path.read_text(encoding="utf-8")) + "</div>")

    sheets = review_sheets(hunt_path, diag_rows)
    sheet_blocks = []
    for i, (png, tickers) in enumerate(sheets, start=1):
        strip = "".join(
            (lambda v:
             f'<div class="sl"><span class="tk">{_esc(t)}</span>'
             f'<span class="pill {_vcls(v)}">{_vlabel(v)}</span>'
             f'<span class="wln">{_esc((verdicts.get(t) or {}).get("notes", ""))}</span></div>'
             )((verdicts.get(t) or {}).get("verdict", "unreviewed"))
            for t in (tickers or []))
        sheet_blocks.append(
            f'<div class="sheet"><div class="eyebrow">Sheet {i} of {len(sheets)}</div>'
            f'<div class="strip">{strip}</div>'
            f'<img src="charts/{_esc(png.name)}" alt="{_esc(png.stem)}" loading="lazy"></div>')
    sheets_html = ""
    if sheet_blocks:
        sheets_html = (f'<h2 class="sheets-h">Review sheets <span class="cnt">&middot; '
                       f'{len(sheets)}, scan order</span></h2>' + "".join(sheet_blocks))

    regime = meta.get("regime") or {}
    date_label = meta.get("scan_time", "")[:10]

    page = _TEMPLATE.format(
        date=_esc(date_label),
        regime=_esc(str(regime.get("regime", "?"))),
        breadth=_f(regime.get("phase2_pct"), "{:.1f}"),
        nhnl=(f'{regime["new_highs"]}/{regime.get("new_lows")}'
              if regime.get("new_highs") is not None else "-"),
        n_scanned=meta.get("n_scanned", "?"), n_tmpl=meta.get("n_passed_template", "?"),
        n_tier_a=meta.get("n_tier_a", "?"), n_elig=meta.get("n_eligible", len(diag_rows)),
        min_rs=meta.get("min_rs", pl.MIN_RS),
        n_pass=n["PASS"], n_cav=n["PASS-"], n_fail=n["FAIL"],
        zone_max=f"{pl.BUY_ZONE_MAX_PCT:.0f}", appr_min=f"{abs(pl.APPROACH_MIN_PCT):.0f}",
        vol_ratio=f"{pl.VOL_CONFIRM_RATIO:.1f}", ern_days=pl.EARNINGS_BLOCK_DAYS,
        n_zone=len(buckets["buy_zone"]), n_appr=len(buckets["approaching"]),
        n_below=len(buckets["below"]), n_past=len(buckets["past_entry"]),
        n_conf=len(confirmed),
        conf_line=(", ".join(r["ticker"] for r in confirmed) if confirmed
                   else "none &mdash; every cross so far is on below-average volume; "
                        "the intraday trigger job is the confirmation watch"),
        min_fund=min_fund, gated=", ".join(gated) or "none", groups_line=groups_line,
        zone_tbl=_mini_table(buckets["buy_zone"], verdicts, catalysts),
        appr_tbl=_mini_table(buckets["approaching"], verdicts, catalysts),
        below_tbl=_mini_table(buckets["below"], verdicts, catalysts),
        past_tbl=_mini_table(buckets["past_entry"], verdicts, catalysts),
        ern_tr=ern_tr or '<tr><td colspan="3" class="dim">none inside the window</td></tr>',
        fund_tr=fund_tr, wl_cards=wl_cards, full_tr=full_tr, n_all=len(diag_rows),
        narrative=narrative, sheets=sheets_html,
    )
    out = hunt_path / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def mirror_report(hunt_path: Path, dest: Path) -> Path:
    """Copy ``report.html`` and the sheets it shows from ``hunt_path`` into ``dest``.

    ``dest/charts`` is replaced, so a sheet the report no longer shows does not linger.
    Returns the copied page's path. Raises FileNotFoundError when the report is not built."""
    page = hunt_path / "report.html"
    if not page.exists():
        raise FileNotFoundError(f"{page} not found — run `report` first.")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(page, dest / "report.html")
    charts = dest / "charts"
    shutil.rmtree(charts, ignore_errors=True)
    sheets = review_sheets(hunt_path, [])
    if sheets:
        charts.mkdir()
        for png, _ in sheets:
            shutil.copy2(png, charts / png.name)
    return dest / "report.html"


_TEMPLATE = """<title>Weekend Hunt &middot; {date}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --bg:#F7F8F6; --panel:#FFFFFF; --ink:#182119; --mut:#5D6B61; --line:#D9DFD9;
  --acc:#2E6E4E; --acc-ink:#FFFFFF;
  --pass:#2E6E4E; --pass-bg:#E2EFE6; --cav:#A87A1C; --cav-bg:#F5ECD7;
  --fail:#A63A30; --fail-bg:#F6E2DF; --oth:#5D6B61; --oth-bg:#E7EAE6;
  --pos:#2E6E4E; --neg:#A63A30; --hover:#EEF2EE;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#111614; --panel:#181F1B; --ink:#E3EAE4; --mut:#93A198; --line:#2A332D;
    --acc:#63A983; --acc-ink:#0C120E;
    --pass:#74B892; --pass-bg:#1C2B22; --cav:#D9A544; --cav-bg:#2C2515;
    --fail:#D9695E; --fail-bg:#301B18; --oth:#93A198; --oth-bg:#202722;
    --pos:#74B892; --neg:#D9695E; --hover:#1E2621;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#111614; --panel:#181F1B; --ink:#E3EAE4; --mut:#93A198; --line:#2A332D;
  --acc:#63A983; --acc-ink:#0C120E;
  --pass:#74B892; --pass-bg:#1C2B22; --cav:#D9A544; --cav-bg:#2C2515;
  --fail:#D9695E; --fail-bg:#301B18; --oth:#93A198; --oth-bg:#202722;
  --pos:#74B892; --neg:#D9695E; --hover:#1E2621;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:16px/1.55 "Source Sans 3", "Segoe UI", system-ui, sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:32px 24px 80px; }}
h1,h2 {{ font-family:"Archivo Narrow", "Arial Narrow", sans-serif; text-wrap:balance; margin:0; }}
h1 {{ font-size:2.3rem; font-weight:700; }}
h2 {{ font-size:1.35rem; font-weight:600; margin:40px 0 8px; }}
h2 .cnt {{ color:var(--mut); font-weight:500; }}
.sub {{ color:var(--mut); margin:2px 0 0; }}
.eyebrow {{ font-family:"IBM Plex Mono", monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--acc); font-weight:600; }}
.statrow {{ display:flex; flex-wrap:wrap; gap:10px; margin:20px 0 0; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:10px 16px; min-width:130px; }}
.stat b {{ display:block; font-family:"IBM Plex Mono",monospace; font-size:1.25rem;
  font-variant-numeric:tabular-nums; font-weight:600; }}
.stat span {{ font-size:.78rem; color:var(--mut); }}
.funnel {{ font-family:"IBM Plex Mono",monospace; font-size:.85rem; color:var(--mut); margin-top:14px; }}
.funnel b {{ color:var(--ink); }}
p.method {{ max-width:68ch; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; background:var(--panel); }}
table {{ border-collapse:collapse; width:100%; font-size:.86rem; }}
th {{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--mut); text-align:left; font-weight:500;
  padding:9px 10px; border-bottom:1px solid var(--line); white-space:nowrap;
  position:sticky; top:0; background:var(--panel); z-index:1; }}
td {{ padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
.scroll tr:hover td {{ background:var(--hover); }}
.tk {{ font-family:"IBM Plex Mono",monospace; font-weight:600; white-space:nowrap; }}
.n {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  text-align:right; white-space:nowrap; }}
th.n {{ text-align:right; }}
.mono {{ font-family:"IBM Plex Mono",monospace; white-space:nowrap; }}
.dim {{ color:var(--mut); }}
.pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
.note {{ min-width:290px; }}
.cat {{ color:var(--mut); font-size:.8rem; margin-top:4px; }}
.cat b {{ color:var(--acc); font-weight:600; }}
.pill {{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; font-weight:600;
  border-radius:4px; padding:2px 7px; white-space:nowrap; }}
.pill.p {{ color:var(--pass); background:var(--pass-bg); }}
.pill.c {{ color:var(--cav); background:var(--cav-bg); }}
.pill.f {{ color:var(--fail); background:var(--fail-bg); }}
.pill.o {{ color:var(--oth); background:var(--oth-bg); }}
.wl {{ color:var(--cav); }}
.chk-y {{ color:var(--pass); font-weight:600; }} .chk-n {{ color:var(--mut); }}
.wlgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:10px; }}
.wlc {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:10px 14px; display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.wlc .wln {{ font-size:.82rem; color:var(--mut); flex-basis:100%; }}
.controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:14px 0 10px; align-items:center; }}
.fbtn {{ font-family:"IBM Plex Mono",monospace; font-size:.75rem;
  border:1px solid var(--line); background:var(--panel); color:var(--ink);
  border-radius:5px; padding:6px 12px; cursor:pointer; }}
.fbtn[aria-pressed="true"] {{ background:var(--acc); color:var(--acc-ink); border-color:var(--acc); }}
.fbtn:focus-visible, #q:focus-visible {{ outline:2px solid var(--acc); outline-offset:2px; }}
#q {{ font:inherit; font-size:.85rem; background:var(--panel); color:var(--ink);
  border:1px solid var(--line); border-radius:5px; padding:6px 10px; width:200px; }}
.foot {{ margin-top:48px; font-size:.78rem; color:var(--mut); max-width:75ch; }}
.narr {{ max-width:80ch; }}
.narr h3 {{ font-family:"Archivo Narrow", "Arial Narrow", sans-serif; font-size:1.15rem; margin:22px 0 6px; }}
.narr h4, .narr h5 {{ font-size:1rem; margin:18px 0 4px; }}
.narr p, .narr li {{ margin:6px 0; }}
.narr ul, .narr ol {{ padding-left:1.4em; }}
.narr code {{ font-family:"IBM Plex Mono",monospace; font-size:.85em; }}
.narr .scroll {{ margin:10px 0; }}
.narr hr {{ border:0; border-top:1px solid var(--line); margin:18px 0; }}
.sheet {{ margin-top:28px; }}
.sheet .strip {{ margin:6px 0 10px; display:grid; gap:4px; }}
.sheet .sl {{ display:flex; gap:10px; align-items:baseline; font-size:.86rem; }}
.sheet .sl .wln {{ color:var(--mut); }}
.sheet img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }}
@page {{ size:letter landscape; margin:.45in; }}
@page sheet {{ size:letter portrait; }}
@media print {{
  /* Paper is light whatever the OS theme: the selectors MUST outrank the dark-mode ones. */
  :root, :root:not([data-theme="light"]), :root[data-theme="dark"] {{
    --bg:#FFFFFF; --panel:#FFFFFF; --ink:#182119; --mut:#5D6B61; --line:#D9DFD9;
    --acc:#2E6E4E; --acc-ink:#FFFFFF;
    --pass:#2E6E4E; --pass-bg:#E2EFE6; --cav:#A87A1C; --cav-bg:#F5ECD7;
    --fail:#A63A30; --fail-bg:#F6E2DF; --oth:#5D6B61; --oth-bg:#E7EAE6;
    --pos:#2E6E4E; --neg:#A63A30; --hover:#FFFFFF;
  }}
  body {{ font-size:12px; }}
  .wrap {{ max-width:none; padding:0; }}
  .scroll {{ overflow:visible; border:none; }}
  .scroll[style] {{ max-height:none !important; overflow:visible !important; }}
  table {{ font-size:.72rem; }}
  th {{ position:static; }}
  td, th {{ padding:4px 6px; }}
  .controls {{ display:none; }}
  h2 {{ break-after:avoid; margin-top:26px; }}
  tr, .wlc, .sl {{ break-inside:avoid; }}
  .sheets-h {{ break-before:page; }}
  .sheet {{ page:sheet; break-before:page; margin-top:0; }}
  .sheet img {{ border:none; }}
}}
</style>
<div class="wrap">
  <div class="eyebrow">SEPA Cockpit &middot; Weekend Hunt</div>
  <h1>Weekend Hunt &middot; {date}</h1>
  <p class="sub">Manual Step-3 chart review of every Tier&nbsp;A candidate with RS&nbsp;&ge;&nbsp;{min_rs}</p>

  <div class="statrow">
    <div class="stat"><b style="color:var(--pass)">{regime}</b>
      <span>regime &middot; breadth {breadth}% &middot; NH/NL {nhnl}</span></div>
    <div class="stat"><b>{n_elig}</b><span>reviewed (Tier A &middot; RS &ge; {min_rs})</span></div>
    <div class="stat"><b style="color:var(--pass)">{n_pass}</b><span>PASS</span></div>
    <div class="stat"><b style="color:var(--cav)">{n_cav}</b><span>PASS with caveats</span></div>
    <div class="stat"><b style="color:var(--fail)">{n_fail}</b><span>FAIL on review</span></div>
  </div>
  <div class="funnel">{n_scanned} scanned &rarr; {n_tmpl} passed 8/8 template &rarr; {n_tier_a} Tier&nbsp;A
  &rarr; <b>{n_elig} with RS&nbsp;&ge;&nbsp;{min_rs}</b> &rarr; <b>{n_pass} clean</b> after chart review</div>

  {narrative}

  <h2>How to read this</h2>
  <p class="method">Verdicts are Step-3 chart judgments against the SEPA checklist. The mechanical rules applied
  below: buy zone = pivot to +{zone_max}% (no chasing); approaching = within {appr_min}% below pivot;
  volume confirmation = a close above the pivot on &ge;{vol_ratio}&times; average volume; entries are
  barred with earnings inside {ern_days} days. Fundamentals (F, 0&ndash;8) are reported, with this run&rsquo;s
  gate at F&nbsp;&ge;&nbsp;{min_fund}. Step-4 &mdash; entries, stops, sizing &mdash; stays with you.</p>

  <h2>Volume-confirmed breakouts <span class="cnt">&middot; the only &ldquo;buy now&rdquo; state ({n_conf})</span></h2>
  <p class="method">{conf_line}</p>
  <p class="method">Buy-zone names clearing this run&rsquo;s F&nbsp;&ge;&nbsp;{min_fund} gate: <b>{gated}</b></p>
  <p class="method">Groups among PASS names: {groups_line}</p>

  <h2>In the buy zone <span class="cnt">&middot; PASS, pivot to +{zone_max}% ({n_zone})</span></h2>
  {zone_tbl}
  <h2>Approaching pivot <span class="cnt">&middot; PASS, within {appr_min}% below &mdash; not yet triggered ({n_appr})</span></h2>
  {appr_tbl}
  <h2>Constructive, below pivot <span class="cnt">&middot; PASS ({n_below})</span></h2>
  {below_tbl}
  <h2>Past the entry range <span class="cnt">&middot; PASS, above +{zone_max}% &mdash; chasing ({n_past})</span></h2>
  {past_tbl}

  <h2>Earnings inside {ern_days} days <span class="cnt">&middot; entry barred regardless of chart</span></h2>
  <div class="scroll"><table>
    <tr><th>Ticker</th><th class="n">Days to earnings</th><th>Chart notes</th></tr>{ern_tr}
  </table></div>

  <h2>Step-2 fundamentals <span class="cnt">&middot; PASS names, sorted by score</span></h2>
  <div class="scroll"><table>
    <tr><th>Ticker</th><th>Position</th><th class="n">F</th><th class="n">Rev&nbsp;grw</th>
    <th class="n">EPS&nbsp;grw</th><th class="n">EPS&nbsp;accel</th><th class="n">Margin</th>
    <th class="n">C33</th><th class="n">FY&nbsp;EPS</th><th class="n">Est&nbsp;&uarr;</th>
    <th class="n">Report held</th>
    <th class="n">Rev YoY</th><th class="n">EPS YoY</th><th class="n">Code&nbsp;33</th>
    <th class="n">Inventory</th><th class="n">Last report</th><th class="n">Est&nbsp;90d</th>
    <th class="n">Funds</th></tr>{fund_tr}
  </table></div>

  <h2>Watchlist audit</h2>
  <div class="wlgrid">{wl_cards}</div>

  <h2>Full review <span class="cnt">&middot; all {n_all}, scan order</span></h2>
  <div class="controls" role="group" aria-label="Filter verdicts">
    <button class="fbtn" data-f="all" aria-pressed="true">ALL {n_all}</button>
    <button class="fbtn" data-f="p" aria-pressed="false">PASS {n_pass}</button>
    <button class="fbtn" data-f="c" aria-pressed="false">PASS&middot; {n_cav}</button>
    <button class="fbtn" data-f="f" aria-pressed="false">FAIL {n_fail}</button>
    <input id="q" type="search" placeholder="find ticker&hellip;" aria-label="Find ticker">
  </div>
  <div class="scroll" style="max-height:72vh; overflow-y:auto;">
  <table id="big"><thead>
    <tr><th class="n">#</th><th>Ticker</th><th>Industry</th><th>Verdict</th><th class="n">Q</th>
    <th class="n">RS</th>
    <th>RS line</th><th class="n">200d mo</th>
    <th class="n">F</th><th class="n">Close</th><th class="n">Pivot</th><th class="n">vs piv</th>
    <th class="n">ADV$M</th><th class="n">Max order $</th><th class="n">DD</th><th>Legs %</th>
    <th class="n">Depth&times;mkt</th><th>Base reads</th><th>Chart notes</th></tr>
  </thead><tbody>{full_tr}</tbody></table>
  </div>

  {sheets}

  <p class="foot">Q = mechanical VCP quality &middot; RS = relative strength &middot; F = fundamental
  checks 0&ndash;8 &middot; vs piv = close relative to detected pivot &middot; ADV$M = 20-day average
  dollar volume &middot; DD = distribution days, last 25 sessions &middot; Base reads: DU = the final
  tight area's volume over its 50-day average / near-silent days &middot; Legs = detected contraction
  sequence, oldest first &middot; &#9733; = watchlist name. Chart verdicts are review notes against the
  SEPA checklist, not trade instructions.</p>
</div>
<script>
(function() {{
  var f = "all";
  var rows = Array.prototype.slice.call(document.querySelectorAll("#big tbody tr"));
  var btns = Array.prototype.slice.call(document.querySelectorAll(".fbtn"));
  var q = document.getElementById("q");
  function apply() {{
    var t = q.value.trim().toLowerCase();
    rows.forEach(function(r) {{
      var okF = (f === "all") || (r.getAttribute("data-v") === f);
      var okT = !t || r.getAttribute("data-t").indexOf(t) === 0;
      r.style.display = (okF && okT) ? "" : "none";
    }});
  }}
  btns.forEach(function(b) {{
    b.addEventListener("click", function() {{
      f = b.getAttribute("data-f");
      btns.forEach(function(x) {{ x.setAttribute("aria-pressed", x === b ? "true" : "false"); }});
      apply();
    }});
  }});
  q.addEventListener("input", apply);
}})();
</script>
"""
