"""Patch Main.ipynb: re-index fundamentals by SEC filing date instead of fp_end.

Changes:
 1. Cell 7b19d317-...: compute_historical_fundamentals gains a
    filing_dates_dict parameter. When provided, each ticker's qdf is
    re-indexed from fp_end -> filed after the TTM calculations and
    before resample("W-FRI"). Removes the ~30-60-day look-ahead leak.
 2. Cell 4de9441a-...: call site updated to pass earnings_filings.

Idempotent — sentinel 'filing_dates_dict' in cell source skips the patch.
Writes Main.ipynb.bak{N} before mutating.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "Main.ipynb"
FUNC_CELL_ID = "7b19d317-a87b-4c9c-86bb-f9a5c5524c53"
CALL_CELL_ID = "4de9441a-d275-41aa-a2a2-becc8734f10b"
SENTINEL = "filing_dates_dict"

NEW_FUNC = '''def compute_historical_fundamentals(historical_fund_dict, stock_data_filename,
                                    filing_dates_dict=None):
    """
    Derive point-in-time fundamental ratios from quarterly financials.
    Computes TTM (trailing 4 quarter) metrics, merges with weekly close
    prices, and forward-fills to weekly frequency. No look-ahead bias.

    If filing_dates_dict is provided (dict of {ticker: DataFrame[filed, fp_end]}
    produced by fetch_earnings_filing_dates), each ticker's quarterly frame is
    re-indexed from fp_end to the actual SEC filing date AFTER TTM calculations
    and BEFORE weekly resampling. This removes the ~30-60-day look-ahead leak
    that exists when fundamentals are joined by fiscal-period-end. Quarters with
    no matching filing date are dropped (no approximate shift).

    Returns a dict of {ticker: DataFrame} indexed on weekly dates with columns:
      HistPE, PB, ProfitMargin, ROE, ROA, DebtToEquity, FCFYield,
      RevenueGrowthYoY, EarningsGrowthYoY, PE_Change_4wk,
      Accruals, BuybackYield, OperatingMargin, AssetTurnover, CFO_to_NI, PSR, PEG
    """
    fund_dict = {}

    n_missing_ticker = 0
    matched_total = 0
    dropped_total = 0

    for ticker, qdf in historical_fund_dict.items():
        qdf = qdf.copy()

        # --- TTM (trailing 4 quarter) aggregations ---
        if "NetIncome" in qdf.columns:
            qdf["NetIncome_TTM"] = qdf["NetIncome"].rolling(4, min_periods=4).sum()
        if "Revenue" in qdf.columns:
            qdf["Revenue_TTM"] = qdf["Revenue"].rolling(4, min_periods=4).sum()
        if "FreeCashFlow" in qdf.columns:
            qdf["FCF_TTM"] = qdf["FreeCashFlow"].rolling(4, min_periods=4).sum()
        if "OperatingIncome" in qdf.columns:
            qdf["OperatingIncome_TTM"] = qdf["OperatingIncome"].rolling(4, min_periods=4).sum()
        if "OperatingCashFlow" in qdf.columns:
            qdf["OperatingCashFlow_TTM"] = qdf["OperatingCashFlow"].rolling(4, min_periods=4).sum()

        # --- YoY growth (compare to 4 quarters ago) ---
        if "Revenue_TTM" in qdf.columns:
            qdf["RevenueGrowthYoY"] = qdf["Revenue_TTM"].pct_change(periods=4)
        if "NetIncome_TTM" in qdf.columns:
            qdf["EarningsGrowthYoY"] = qdf["NetIncome_TTM"].pct_change(periods=4)

        # --- Point-in-time balance sheet ratios ---
        if "Equity" in qdf.columns and "TotalDebt" in qdf.columns:
            qdf["DebtToEquity"] = qdf["TotalDebt"] / qdf["Equity"]
        if "NetIncome_TTM" in qdf.columns and "Equity" in qdf.columns:
            qdf["ROE"] = qdf["NetIncome_TTM"] / qdf["Equity"]
        if "NetIncome_TTM" in qdf.columns and "TotalAssets" in qdf.columns:
            qdf["ROA"] = qdf["NetIncome_TTM"] / qdf["TotalAssets"]
        if "NetIncome_TTM" in qdf.columns and "Revenue_TTM" in qdf.columns:
            qdf["ProfitMargin"] = qdf["NetIncome_TTM"] / qdf["Revenue_TTM"]

        # --- Tier 1 derived features (orthogonal quality / valuation signals) ---
        if {"NetIncome_TTM", "OperatingCashFlow_TTM", "TotalAssets"}.issubset(qdf.columns):
            qdf["Accruals"] = (qdf["NetIncome_TTM"] - qdf["OperatingCashFlow_TTM"]) / qdf["TotalAssets"]
        if "SharesOutstanding" in qdf.columns:
            qdf["BuybackYield"] = -qdf["SharesOutstanding"].pct_change(periods=4)
        if {"OperatingIncome_TTM", "Revenue_TTM"}.issubset(qdf.columns):
            qdf["OperatingMargin"] = qdf["OperatingIncome_TTM"] / qdf["Revenue_TTM"]
        if {"Revenue_TTM", "TotalAssets"}.issubset(qdf.columns):
            qdf["AssetTurnover"] = qdf["Revenue_TTM"] / qdf["TotalAssets"]
        if {"OperatingCashFlow_TTM", "NetIncome_TTM"}.issubset(qdf.columns):
            qdf["CFO_to_NI"] = qdf["OperatingCashFlow_TTM"] / qdf["NetIncome_TTM"]

        # --- EPS for P/E ---
        if "NetIncome_TTM" in qdf.columns and "SharesOutstanding" in qdf.columns:
            qdf["EPS_TTM"] = qdf["NetIncome_TTM"] / qdf["SharesOutstanding"]

        # --- Re-index from fp_end to SEC filing date (point-in-time fix) ---
        # Done AFTER TTM calcs (so rolling sums see correct quarterly order)
        # and BEFORE weekly resample (so forward-fill uses filing-date timing).
        if filing_dates_dict is not None:
            if ticker in filing_dates_dict:
                filings = filing_dates_dict[ticker]
                fp_to_filed = (filings.dropna(subset=["fp_end", "filed"])
                                      .sort_values("filed")
                                      .drop_duplicates(subset=["fp_end"], keep="first")
                                      .set_index("fp_end")["filed"])
                qdf["__filed__"] = qdf.index.map(fp_to_filed)
                n_before = len(qdf)
                qdf = qdf.dropna(subset=["__filed__"])
                matched_total += len(qdf)
                dropped_total += n_before - len(qdf)
                qdf = qdf.set_index("__filed__").sort_index()
                qdf.index.name = "Date"
            else:
                n_missing_ticker += 1

        # --- Merge with weekly prices for price-based ratios ---
        try:
            prices = extract_ticker_dataframe(stock_data_filename, ticker)
            weekly_close = prices["Close"]
        except (ValueError, KeyError):
            continue

        # Forward-fill quarterly data to weekly frequency
        weekly_q = qdf.resample("W-FRI").ffill()
        weekly_q = weekly_q.reindex(weekly_close.index, method="ffill")

        result = pd.DataFrame(index=weekly_close.index)

        # MarketCap reused across price-based ratios
        market_cap = None
        if "SharesOutstanding" in weekly_q.columns:
            market_cap = weekly_close * weekly_q["SharesOutstanding"]

        # Price-based ratios
        if "EPS_TTM" in weekly_q.columns:
            result["HistPE"] = weekly_close / weekly_q["EPS_TTM"]
            result["PE_Change_4wk"] = result["HistPE"].pct_change(periods=4)

        if "Equity" in weekly_q.columns and "SharesOutstanding" in weekly_q.columns:
            book_per_share = weekly_q["Equity"] / weekly_q["SharesOutstanding"]
            result["PB"] = weekly_close / book_per_share

        if "FCF_TTM" in weekly_q.columns and market_cap is not None:
            result["FCFYield"] = weekly_q["FCF_TTM"] / market_cap

        if "Revenue_TTM" in weekly_q.columns and market_cap is not None:
            result["PSR"] = market_cap / weekly_q["Revenue_TTM"]

        if "HistPE" in result.columns and "EarningsGrowthYoY" in weekly_q.columns:
            # PEG only meaningful when growth is positive
            growth = weekly_q["EarningsGrowthYoY"].where(weekly_q["EarningsGrowthYoY"] > 0)
            result["PEG"] = result["HistPE"] / growth

        # Non-price ratios (already computed, just align)
        for col in ["ProfitMargin", "ROE", "ROA", "DebtToEquity",
                     "RevenueGrowthYoY", "EarningsGrowthYoY",
                     "Accruals", "BuybackYield", "OperatingMargin",
                     "AssetTurnover", "CFO_to_NI"]:
            if col in weekly_q.columns:
                result[col] = weekly_q[col]

        # Clean up infinities
        result = result.replace([np.inf, -np.inf], np.nan)

        fund_dict[ticker] = result

    if filing_dates_dict is not None:
        total = matched_total + dropped_total
        pct = (100.0 * matched_total / total) if total else 0.0
        print(f"[fundamentals] filing-date match: {matched_total}/{total} quarters ({pct:.1f}%)"
              f"  dropped={dropped_total}  tickers_missing_from_filings_dict={n_missing_ticker}")

    return fund_dict
'''


def next_backup_path(nb_path: Path) -> Path:
    n = 1
    while True:
        cand = nb_path.with_suffix(nb_path.suffix + f".bak{n}")
        if not cand.exists():
            return cand
        n += 1


def find_block(source_lines, start_token, end_predicate):
    """Locate a contiguous block of cell source by the start line content
    and an end predicate. Returns (start_idx, end_idx_exclusive) or None."""
    start = None
    for i, ln in enumerate(source_lines):
        if start_token in ln:
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(source_lines)):
        if end_predicate(source_lines[j]):
            return start, j + 1
    return None


def patch_function_cell(nb):
    cell = next((c for c in nb["cells"] if c.get("id") == FUNC_CELL_ID), None)
    if cell is None:
        print(f"FAIL: function cell {FUNC_CELL_ID} not found", file=sys.stderr)
        sys.exit(1)

    source = cell["source"]
    joined = "".join(source)
    if SENTINEL in joined:
        print(f"Function cell already patched (sentinel '{SENTINEL}' present); skipping.")
        return False

    # Block: from "def compute_historical_fundamentals" through "return fund_dict"
    block = find_block(
        source,
        start_token="def compute_historical_fundamentals",
        end_predicate=lambda ln: ln.strip() == "return fund_dict",
    )
    if block is None:
        print("FAIL: could not locate function block in 7b19d317 cell", file=sys.stderr)
        sys.exit(2)
    start, end = block
    new_lines = [ln + "\n" for ln in NEW_FUNC.splitlines()]
    cell["source"] = source[:start] + new_lines + source[end:]
    print(f"Patched function cell {FUNC_CELL_ID} (replaced lines {start}:{end})")
    return True


def patch_call_cell(nb):
    cell = next((c for c in nb["cells"] if c.get("id") == CALL_CELL_ID), None)
    if cell is None:
        print(f"FAIL: call cell {CALL_CELL_ID} not found", file=sys.stderr)
        sys.exit(3)

    source = cell["source"]
    joined = "".join(source)
    if "filing_dates_dict=earnings_filings" in joined:
        print(f"Call cell already patched; skipping.")
        return False

    new_source = []
    matched = False
    for ln in source:
        if "compute_historical_fundamentals(historical_fundamentals" in ln and "filing_dates_dict" not in ln:
            new_ln = ln.replace(
                'compute_historical_fundamentals(historical_fundamentals, "data/training_set.csv")',
                'compute_historical_fundamentals(historical_fundamentals, "data/training_set.csv", filing_dates_dict=earnings_filings)',
            )
            if new_ln == ln:
                # Fallback: handle different quoting / spacing variants by appending arg before final ')'
                idx = ln.rfind(")")
                if idx == -1:
                    new_source.append(ln)
                    continue
                new_ln = ln[:idx] + ", filing_dates_dict=earnings_filings" + ln[idx:]
            new_source.append(new_ln)
            matched = True
        else:
            new_source.append(ln)

    if not matched:
        print("FAIL: did not match compute_historical_fundamentals call line", file=sys.stderr)
        sys.exit(4)

    cell["source"] = new_source
    print(f"Patched call cell {CALL_CELL_ID}")
    return True


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    # Snapshot before any mutation
    backup = next_backup_path(NB_PATH)
    backup.write_text(NB_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    changed_a = patch_function_cell(nb)
    changed_b = patch_call_cell(nb)

    if not (changed_a or changed_b):
        backup.unlink()  # nothing to do
        print("No changes (already patched). Backup removed.")
        return

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote notebook. Backup: {backup.name}")


if __name__ == "__main__":
    main()
