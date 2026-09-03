"""Weekend-hunt HTML report, built from a hunt directory's persisted state
(diagnostics.csv + verdicts.csv + meta.json). Self-contained single file:
Google-Fonts faces with real fallbacks, light/dark via tokens, no JS deps.
"""
from __future__ import annotations

import csv
import html as _html
import json
from pathlib import Path
from typing import Dict, List

from . import pipeline as pl


def _esc(s) -> str:
    return _html.escape(str(s), quote=True)


def _vcls(v: str) -> str:
    return {"PASS": "p", "PASS-": "c", "FAIL": "f"}.get(v, "o")


def _vlabel(v: str) -> str:
    return {"PASS": "PASS", "PASS-": "PASS&middot;", "FAIL": "FAIL"}.get(v, _esc(v).upper())


def _f(x, fmt="{:.2f}", dash="-"):
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return dash


def _mini_table(rows: List[dict], verdicts: Dict[str, dict]) -> str:
    tr = []
    for r in rows:
        star = ' <span class="wl" title="on watchlist">&#9733;</span>' if int(r.get("wl") or 0) else ""
        note = (verdicts.get(r["ticker"]) or {}).get("notes", "")
        vs = float(r["vs_pivot_pct"])
        tr.append(
            f'<tr><td class="tk">{_esc(r["ticker"])}{star}</td>'
            f'<td class="n">{_f(r["q"], "{:.0f}")}</td><td class="n">{r["rs"]}</td>'
            f'<td class="n">{_f(r["close"])}</td><td class="n">{_f(r["pivot"])}</td>'
            f'<td class="n {"pos" if vs >= 0 else "neg"}">{vs:+.1f}%</td>'
            f'<td class="n">{_f(r["adv_musd"], "{:.1f}")}</td>'
            f'<td class="n">{_f(r["volume_ratio"], "{:.2f}")}&times;</td>'
            f'<td class="note">{_esc(note)}</td></tr>')
    head = ('<tr><th>Ticker</th><th class="n">Q</th><th class="n">RS</th><th class="n">Close</th>'
            '<th class="n">Pivot</th><th class="n">vs piv</th><th class="n">ADV$M</th>'
            '<th class="n">Fri vol</th><th>Chart notes</th></tr>')
    body = "".join(tr) or '<tr><td colspan="9" class="dim">none</td></tr>'
    return f'<div class="scroll"><table>{head}{body}</table></div>'


def build_report(hunt_path: Path, min_fund: int = 0) -> Path:
    diag_rows = list(csv.DictReader(open(hunt_path / "diagnostics.csv", encoding="utf-8")))
    verdicts = pl.read_verdicts(hunt_path / "verdicts.csv")
    meta = json.loads((hunt_path / "meta.json").read_text(encoding="utf-8"))

    for r in diag_rows:                         # numeric round-trip from CSV
        for k in ("q", "close", "pivot", "vs_pivot_pct", "adv_musd", "volume_ratio"):
            r[k] = float(r[k])
        for k in ("rs", "fund", "wl", "dist_days", "breakout_today",
                  "f_rev", "f_eps", "f_accel", "f_margin"):
            r[k] = int(float(r[k]))

    n = {"PASS": 0, "PASS-": 0, "FAIL": 0}
    for v in verdicts.values():
        if v["verdict"] in n:
            n[v["verdict"]] += 1

    import pandas as pd
    diag_df = pd.DataFrame(diag_rows)

    # The buckets come from pipeline.gates so the HTML and the `gates` CLI can never
    # disagree about the same run; min_fund=0 here because the report SHOWS every PASS
    # name and applies the fundamental gate only to the summary line below.
    passing = [r for r in diag_rows if (verdicts.get(r["ticker"]) or {}).get("verdict") == "PASS"]
    g = pl.gates(diag_df, verdicts, min_fund=0)
    by_ticker = {r["ticker"]: r for r in diag_rows}
    buckets = {k: [by_ticker[x["ticker"]] for x in g[k]]
               for k in ("buy_zone", "approaching", "below", "past_entry")}
    blocked = [by_ticker[x["ticker"]] for x in g["earnings_blocked"]]
    confirmed = [by_ticker[x["ticker"]] for x in g["volume_confirmed"]]

    audit = pl.watchlist_audit(diag_df, verdicts)

    # ---- fragments -------------------------------------------------------- #
    def chk(b): return ('<td class="n chk-y">&#10003;</td>' if b
                        else '<td class="n chk-n">&ndash;</td>')
    fund_tr = "".join(
        f'<tr><td class="tk">{_esc(r["ticker"])}</td>'
        f'<td>{ {"buy_zone": "buy zone", "approaching": "approaching", "below": "below pivot", "past_entry": "past entry"}[pl.bucket(r["vs_pivot_pct"])] }</td>'
        f'<td class="n"><b>{r["fund"]}</b>/4</td>'
        + chk(r["f_rev"]) + chk(r["f_eps"]) + chk(r["f_accel"]) + chk(r["f_margin"]) +
        f'<td class="n">{_f(r["rev_yoy"], "{:+.1f}%")}</td>'
        f'<td class="n">{_f(r["eps_yoy"], "{:+.1f}%")}</td></tr>'
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
         f'<td><span class="pill {_vcls(v)}">{_vlabel(v)}</span></td>'
         f'<td class="n">{_f(r["q"], "{:.0f}")}</td><td class="n">{r["rs"]}</td>'
         f'<td class="n">{r["fund"]}</td>'
         f'<td class="n">{_f(r["close"])}</td><td class="n">{_f(r["pivot"])}</td>'
         f'<td class="n {"pos" if r["vs_pivot_pct"] >= 0 else "neg"}">{r["vs_pivot_pct"]:+.1f}%</td>'
         f'<td class="n">{_f(r["adv_musd"], "{:.1f}")}</td><td class="n">{r["dist_days"]}</td>'
         f'<td class="mono dim">{_esc(r["depths"])}</td>'
         f'<td class="note">{_esc((verdicts.get(r["ticker"]) or {}).get("notes", ""))}</td></tr>'
         )((verdicts.get(r["ticker"]) or {}).get("verdict", "unreviewed"))
        for r in diag_rows)

    regime = meta.get("regime") or {}
    date_label = meta.get("scan_time", "")[:10]
    gated = [x["ticker"] for x in pl.gates(diag_df, verdicts, min_fund=min_fund)["buy_zone"]]

    page = _TEMPLATE.format(
        date=_esc(date_label),
        regime=_esc(str(regime.get("regime", "?"))),
        breadth=_f(regime.get("phase2_pct"), "{:.1f}"),
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
        min_fund=min_fund, gated=", ".join(gated) or "none",
        zone_tbl=_mini_table(buckets["buy_zone"], verdicts),
        appr_tbl=_mini_table(buckets["approaching"], verdicts),
        below_tbl=_mini_table(buckets["below"], verdicts),
        past_tbl=_mini_table(buckets["past_entry"], verdicts),
        ern_tr=ern_tr or '<tr><td colspan="3" class="dim">none inside the window</td></tr>',
        fund_tr=fund_tr, wl_cards=wl_cards, full_tr=full_tr, n_all=len(diag_rows),
    )
    out = hunt_path / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


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
</style>
<div class="wrap">
  <div class="eyebrow">SEPA Cockpit &middot; Weekend Hunt</div>
  <h1>Weekend Hunt &middot; {date}</h1>
  <p class="sub">Manual Step-3 chart review of every Tier&nbsp;A candidate with RS&nbsp;&ge;&nbsp;{min_rs}</p>

  <div class="statrow">
    <div class="stat"><b style="color:var(--pass)">{regime}</b><span>regime &middot; breadth {breadth}%</span></div>
    <div class="stat"><b>{n_elig}</b><span>reviewed (Tier A &middot; RS &ge; {min_rs})</span></div>
    <div class="stat"><b style="color:var(--pass)">{n_pass}</b><span>PASS</span></div>
    <div class="stat"><b style="color:var(--cav)">{n_cav}</b><span>PASS with caveats</span></div>
    <div class="stat"><b style="color:var(--fail)">{n_fail}</b><span>FAIL on review</span></div>
  </div>
  <div class="funnel">{n_scanned} scanned &rarr; {n_tmpl} passed 8/8 template &rarr; {n_tier_a} Tier&nbsp;A
  &rarr; <b>{n_elig} with RS&nbsp;&ge;&nbsp;{min_rs}</b> &rarr; <b>{n_pass} clean</b> after chart review</div>

  <h2>How to read this</h2>
  <p class="method">Verdicts are Step-3 chart judgments against the SEPA checklist. The mechanical rules applied
  below: buy zone = pivot to +{zone_max}% (no chasing); approaching = within {appr_min}% below pivot;
  volume confirmation = a close above the pivot on &ge;{vol_ratio}&times; average volume; entries are
  barred with earnings inside {ern_days} days. Fundamentals (F, 0&ndash;4) are reported, with this run&rsquo;s
  gate at F&nbsp;&ge;&nbsp;{min_fund}. Step-4 &mdash; entries, stops, sizing &mdash; stays with you.</p>

  <h2>Volume-confirmed breakouts <span class="cnt">&middot; the only &ldquo;buy now&rdquo; state ({n_conf})</span></h2>
  <p class="method">{conf_line}</p>
  <p class="method">Buy-zone names clearing this run&rsquo;s F&nbsp;&ge;&nbsp;{min_fund} gate: <b>{gated}</b></p>

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
    <th class="n">Rev YoY</th><th class="n">EPS YoY</th></tr>{fund_tr}
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
    <tr><th class="n">#</th><th>Ticker</th><th>Verdict</th><th class="n">Q</th><th class="n">RS</th>
    <th class="n">F</th><th class="n">Close</th><th class="n">Pivot</th><th class="n">vs piv</th>
    <th class="n">ADV$M</th><th class="n">DD</th><th>Legs %</th><th>Chart notes</th></tr>
  </thead><tbody>{full_tr}</tbody></table>
  </div>

  <p class="foot">Q = mechanical VCP quality &middot; RS = relative strength &middot; F = fundamental
  checks 0&ndash;4 &middot; vs piv = close relative to detected pivot &middot; ADV$M = 20-day average
  dollar volume &middot; DD = distribution days, last 25 sessions &middot; Legs = detected contraction
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
