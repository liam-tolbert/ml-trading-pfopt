"""Ingest CRSP + Compustat from WRDS into a local cache for the daily backtest.

Run this in an environment with the `wrds` package installed and a WRDS account:

    python src/stock_screener/backtest_daily/ingest_wrds.py --start 2003-01-01 --top-n 3000 --out data/wrds

Credentials: put WRDS_USERNAME and WRDS_PASSWORD in `.env` (gitignored) for a
non-interactive connection; if they are blank/absent the wrds package prompts and/or
uses ~/.pgpass. Writes five tables (universe / prices / delist / fundamentals / spy)
consumed by wrds_provider.py.

Validate small first:  --max-names 300 --start 2018-01-01

Uses classic SIZ CRSP tables (crsp.dsf/msf/msenames/dsedelist) + CCM
(crsp.ccmxpf_lnkhist) + Compustat (comp.fundq). If your subscription is CIZ-only,
swap the price/delist queries to crsp.wrds_dsfv2 / crsp.stkdlysecuritydata
(DlyRet already incorporates the delisting return).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.stock_screener.backtest_daily.cache_io import write_table  # noqa: E402

SPY_PERMNO = 84398  # SPY ETF in CRSP


# --------------------------------------------------------------------------- #
def connect():
    """Connect to WRDS. Credentials are read from `.env` (WRDS_USERNAME / WRDS_PASSWORD) when
    present, giving a non-interactive connection (libpq reads the password from PGPASSWORD);
    otherwise the wrds package prompts and/or falls back to ~/.pgpass."""
    import wrds  # lazy: only needed for a live pull
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    user = os.environ.get("WRDS_USERNAME") or None
    pw = os.environ.get("WRDS_PASSWORD") or None
    if pw:
        os.environ["PGPASSWORD"] = pw          # libpq reads this -> no prompt, no ~/.pgpass needed
    return wrds.Connection(wrds_username=user) if user else wrds.Connection()


# --------------------------------------------------------------------------- #
def build_universe(db, start, end, top_n):
    """Quarterly top-N by market cap from monthly CRSP (cheap). Returns
    (universe_df[rebalance_date, permno], exch_by_permno)."""
    sql = f"""
        select a.permno, a.date, abs(a.prc) * a.shrout as mcap, b.exchcd
        from crsp.msf a
        join crsp.msenames b
          on a.permno = b.permno and b.namedt <= a.date and a.date <= b.nameendt
        where a.date between '{start}' and '{end}'
          and b.shrcd in (10, 11) and b.exchcd in (1, 2, 3)
          and a.prc is not null and a.shrout is not null and a.shrout > 0
    """
    msf = db.raw_sql(sql, date_cols=["date"])
    msf = msf.dropna(subset=["mcap"])
    msf = msf[msf["date"].dt.month.isin([3, 6, 9, 12])]          # quarter-ends
    rows = []
    for d, g in msf.groupby("date"):
        for p in g.nlargest(top_n, "mcap")["permno"].astype(int):
            rows.append({"rebalance_date": d, "permno": int(p)})
    uni = pd.DataFrame(rows)
    exch_by_permno = (msf.sort_values("date").groupby("permno")["exchcd"]
                      .last().astype(int).to_dict())
    return uni, exch_by_permno


def build_prices(db, permnos, start, end, batch=400):
    """Daily CRSP prices for the name set, chunked, split-adjusted & sign-fixed."""
    permnos = sorted(set(int(p) for p in permnos))
    parts = []
    for i in range(0, len(permnos), batch):
        chunk = permnos[i:i + batch]
        inlist = ",".join(str(p) for p in chunk)
        sql = f"""
            select permno, date, openprc, askhi, bidlo, prc, vol, cfacpr
            from crsp.dsf
            where permno in ({inlist}) and date between '{start}' and '{end}'
        """
        df = db.raw_sql(sql, date_cols=["date"])
        if len(df):
            parts.append(df)
        print(f"  prices batch {i // batch + 1}/{(len(permnos) - 1) // batch + 1}: "
              f"{len(df)} rows ({len(chunk)} names)")
    px = pd.concat(parts, ignore_index=True)
    cf = px["cfacpr"].replace(0, np.nan).fillna(1.0)
    out = pd.DataFrame({
        "permno": px["permno"].astype(int),
        "date": px["date"],
        "open": px["openprc"].abs() / cf,
        "high": px["askhi"].abs() / cf,
        "low": px["bidlo"].abs() / cf,
        "close": px["prc"].abs() / cf,
        "volume": px["vol"].fillna(0.0),
        "raw_close": px["prc"].abs(),               # UNADJUSTED price, for the Minervini floor
    })
    out = out.dropna(subset=["close"])
    for c in ("open", "high", "low"):                            # fall back to close
        out[c] = out[c].fillna(out["close"])
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    # despike cfacpr / penny artifacts (data cleaning at ingestion, not a trading signal):
    g = out.groupby("permno")["close"]
    med = g.transform("median")
    prev, nxt = g.shift(1), g.shift(-1)
    # (a) gross level outliers vs the name's robust full-series median
    gross = (med > 0) & ((out["close"] > 100.0 * med) | (out["close"] < med / 100.0))
    # (b) isolated single-bar spike-and-revert (corrupt print, not a real level shift):
    #     >5x vs prev then reverts next bar (or the symmetric down-spike). Real moves that
    #     HOLD their new level are not flagged.
    up = (prev > 0) & (out["close"] > 5.0 * prev) & (nxt < 2.0 * prev)
    dn = (prev > 0) & (out["close"] < prev / 5.0) & (nxt > prev / 2.0)
    spike = gross | up | dn
    if int(spike.sum()):
        print(f"  despiked {int(spike.sum())} outlier price row(s)")
    return out[~spike].reset_index(drop=True)


def build_delist(db, permnos, prices, exch_by_permno, end):
    """Delisting returns; impute performance-delist codes; snap date to the last
    in-cache trading day (so the engine's exact-date exit fires)."""
    inlist = ",".join(str(int(p)) for p in sorted(set(permnos)))
    dl = db.raw_sql(
        f"select permno, dlstdt, dlret, dlstcd from crsp.dsedelist where permno in ({inlist})",
        date_cols=["dlstdt"])
    last_date = prices.groupby("permno")["date"].max()
    end_ts = pd.Timestamp(end)
    rows = []
    for r in dl.itertuples(index=False):
        p = int(r.permno)
        if p not in last_date.index or pd.isna(r.dlstdt) or r.dlstdt > end_ts:
            continue
        ret = r.dlret
        if pd.isna(ret) and pd.notna(r.dlstcd) and (r.dlstcd == 500 or 520 <= r.dlstcd <= 584):
            ret = -0.30 if exch_by_permno.get(p, 3) in (1, 2) else -0.55   # Shumway
        if pd.isna(ret):
            continue
        rows.append({"permno": p, "delist_date": last_date.loc[p], "dlret": float(ret)})
    return pd.DataFrame(rows, columns=["permno", "delist_date", "dlret"])


def build_fundamentals(db, permnos, start, end, batch=800):
    """Compustat fundq linked to PERMNO via CCM, lagged to RDQ. Filtered to the name
    set in SQL (chunked) so a validation slice doesn't pull all of Compustat."""
    permnos = sorted(set(int(p) for p in permnos))
    parts = []
    for i in range(0, len(permnos), batch):
        inlist = ",".join(str(p) for p in permnos[i:i + batch])
        sql = f"""
            select f.gvkey, f.datadate, f.rdq, f.revtq, f.saleq, f.epsfxq, f.epspxq, f.invtq,
                   l.lpermno as permno
            from comp.fundq f
            join crsp.ccmxpf_lnkhist l
              on f.gvkey = l.gvkey
             and l.linktype in ('LU', 'LC') and l.linkprim in ('P', 'C')
             and l.linkdt <= f.datadate and (l.linkenddt is null or f.datadate <= l.linkenddt)
            where f.indfmt = 'INDL' and f.datafmt = 'STD' and f.popsrc = 'D' and f.consol = 'C'
              and f.datadate between '{start}' and '{end}'
              and l.lpermno in ({inlist})
        """
        df = db.raw_sql(sql, date_cols=["datadate", "rdq"])
        if len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["permno", "datadate", "rdq", "revtq", "eps", "invtq"])
    f = pd.concat(parts, ignore_index=True)
    f["permno"] = f["permno"].astype(int)
    f["revtq"] = f["revtq"].fillna(f["saleq"])
    f["eps"] = f["epsfxq"].fillna(f["epspxq"])
    f["rdq"] = f["rdq"].fillna(f["datadate"] + pd.Timedelta(days=90))   # mild look-ahead, documented
    out = f[["permno", "datadate", "rdq", "revtq", "eps", "invtq"]].dropna(subset=["datadate"])
    return out.sort_values(["permno", "datadate"]).reset_index(drop=True)


def build_spy(db, start, end, permno=SPY_PERMNO):
    sql = f"""
        select date, openprc, askhi, bidlo, prc, vol, cfacpr
        from crsp.dsf where permno = {permno} and date between '{start}' and '{end}'
    """
    px = db.raw_sql(sql, date_cols=["date"])
    cf = px["cfacpr"].replace(0, np.nan).fillna(1.0)
    out = pd.DataFrame({
        "date": px["date"],
        "open": px["openprc"].abs() / cf, "high": px["askhi"].abs() / cf,
        "low": px["bidlo"].abs() / cf, "close": px["prc"].abs() / cf,
        "volume": px["vol"].fillna(0.0),
    })
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    return out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Ingest CRSP+Compustat from WRDS to a local cache.")
    ap.add_argument("--start", default="2003-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--top-n", type=int, default=3000)
    ap.add_argument("--max-names", type=int, default=None, help="cap the name set (validation slice)")
    ap.add_argument("--batch", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "data" / "wrds"))
    args = ap.parse_args()

    print(f"Connecting to WRDS ...")
    db = connect()
    try:
        print(f"[1/5] universe (top {args.top_n} by cap, {args.start}->{args.end}) ...")
        uni, exch = build_universe(db, args.start, args.end, args.top_n)
        permnos = sorted(uni["permno"].unique())
        if args.max_names:
            permnos = permnos[:args.max_names]
            uni = uni[uni["permno"].isin(set(permnos))]
        print(f"      {len(uni)} membership rows, {len(permnos)} distinct names")

        print(f"[2/5] daily prices ({len(permnos)} names) ...")
        prices = build_prices(db, permnos, args.start, args.end, batch=args.batch)
        print(f"      {len(prices):,} price rows")

        print("[3/5] delisting ...")
        delist = build_delist(db, permnos, prices, exch, args.end)
        print(f"      {len(delist)} delistings in-universe")

        print("[4/5] fundamentals (Compustat via CCM) ...")
        fundamentals = build_fundamentals(db, permnos, args.start, args.end)
        print(f"      {len(fundamentals):,} quarterly rows")

        print("[5/5] SPY benchmark ...")
        spy = build_spy(db, args.start, args.end)
        print(f"      {len(spy):,} SPY rows")
    finally:
        try:
            db.close()
        except Exception:
            pass

    out = Path(args.out)
    for name, df in [("universe", uni), ("prices", prices), ("delist", delist),
                     ("fundamentals", fundamentals), ("spy", spy)]:
        path = write_table(df, out, name)
        print(f"wrote {path}")

    # sanity: survivorship-free build MUST contain delisted-while-in-universe names
    print(f"\nSANITY: {len(delist)} delisted-in-universe names "
          f"({'OK' if len(delist) > 0 else 'WARNING: 0 -> universe likely survivorship-biased'})")
    print(f"calendar: {spy['date'].min().date()} -> {spy['date'].max().date()} ({len(spy)} days)")


if __name__ == "__main__":
    main()
