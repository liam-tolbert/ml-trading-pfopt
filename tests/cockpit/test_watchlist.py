"""Cockpit tests — watchlist persistence, normalization, migration and CSV/txt export.

Runs standalone (`python tests/cockpit/test_watchlist.py`) or as part of the full
gate (`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


def test_watchlist_export_helpers():
    """The watchlist CSV builders: the decision list keeps user order and never drops a
    picked ticker (stale ones survive as ticker-only rows); the OHLCV dump stacks every
    present name long-format with a Ticker column."""
    import pandas as pd
    from src.stock_screener.cockpit.export import (watchlist_list_csv,
                                                   watchlist_ohlcv_csv)

    cand = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"],
                         "tier": ["A", "B", "A"], "pivot": [10.0, 20.0, 30.0]})

    # list CSV takes ENTRIES (dicts and/or bare strings): order follows the watchlist
    # (BBB before AAA); a stale pick (ZZZ, not in the scan) still appears so nothing the
    # user chose is silently lost; the frozen-pivot metadata columns always ride along.
    ents = [{"ticker": "BBB", "judged_pivot": 19.5, "date_added": "2026-07-12",
             "pivot_source": "judged", "note": "n1"},
            "AAA",                                             # bare string still accepted
            {"ticker": "ZZZ"}]
    csv = watchlist_list_csv(cand, ents, columns=["ticker", "tier", "pivot"]).decode()
    lines = csv.strip().splitlines()
    assert lines[0] == "ticker,tier,pivot,judged_pivot,date_added,pivot_source,note"
    order = [ln.split(",")[0] for ln in lines[1:]]
    assert order == ["BBB", "AAA", "ZZZ"], order
    assert lines[1].startswith("BBB,B,20.0,19.5,2026-07-12,judged,n1")
    assert lines[2].split(",")[3] == ""                        # unfrozen -> empty judged_pivot

    # empty candidates -> ticker-only rows + (empty) metadata columns (never raises)
    empty = watchlist_list_csv(pd.DataFrame(), ["AAA", "BBB"]).decode()
    elines = empty.strip().splitlines()
    assert elines[0] == "ticker,judged_pivot,date_added,pivot_source,note"
    assert [ln.split(",")[0] for ln in elines[1:]] == ["AAA", "BBB"]

    # OHLCV CSV: two names stacked, Date + Ticker lead, present names only
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=5)
    mk = lambda base: pd.DataFrame(  # noqa: E731
        {"Open": base, "High": base + 1, "Low": base - 1, "Close": base,
         "Volume": 100}, index=idx)
    payloads = {"AAA": {"df": mk(10.0)}, "BBB": {"df": mk(20.0)}}
    ocsv = watchlist_ohlcv_csv(["AAA", "BBB", "ZZZ"], payloads).decode()
    olines = ocsv.strip().splitlines()
    assert olines[0].split(",")[:2] == ["Date", "Ticker"], olines[0]
    assert len(olines) == 1 + 2 * len(idx)                      # header + 5 bars × 2 names
    assert "ZZZ" not in ocsv                                    # absent name omitted
    assert ",AAA," in ocsv and ",BBB," in ocsv

    assert watchlist_ohlcv_csv([], payloads) == b""             # empty list -> empty bytes


def test_watchlist_list_csv_keeps_stale_in_order():
    """Issue 7: a stale (not-in-scan) ticker keeps its watchlist POSITION instead of being moved
    to the end. For [ZZZ(stale), BBB, AAA] the rows come out ZZZ, BBB, AAA — matching the download
    help's 'in the order you added them' (pre-fix a concat-append produced BBB, AAA, ZZZ). The
    existing helper test can't catch this because its stale pick already sat last."""
    import pandas as pd
    from src.stock_screener.cockpit.export import watchlist_list_csv

    cand = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"],
                         "tier": ["A", "B", "A"], "pivot": [10.0, 20.0, 30.0]})
    ents = [{"ticker": "ZZZ", "judged_pivot": 5.0, "date_added": "2026-07-18",
             "pivot_source": "judged", "note": ""},            # stale AND first in the watchlist
            "BBB", "AAA"]
    csv = watchlist_list_csv(cand, ents, columns=["ticker", "tier", "pivot"]).decode()
    lines = csv.strip().splitlines()
    assert [ln.split(",")[0] for ln in lines[1:]] == ["ZZZ", "BBB", "AAA"], csv
    zzz = lines[1].split(",")
    assert zzz[1] == "" and zzz[2] == ""                        # stale -> empty tier/pivot cells
    assert zzz[3].startswith("5")                               # frozen judged_pivot preserved
    # the in-scan rows still carry their decision columns, in place
    bbb = lines[2].split(",")
    assert bbb[0] == "BBB" and bbb[1] == "B" and bbb[2] == "20.0"


def test_watchlist_persistence():
    """save_watchlist/load_watchlist round-trip ENTRY DICTS (ticker upper/strip, de-duped
    first-seen); a legacy string-array file migrates to unfrozen entries at load; load
    returns [] for missing/corrupt/non-list files."""
    import json as _json
    import tempfile
    from pathlib import Path as _P
    from src.stock_screener.cockpit.export import (save_watchlist, load_watchlist,
                                                   make_entry)

    with tempfile.TemporaryDirectory() as tmp:
        p = _P(tmp) / "sub" / "watchlist.json"                 # parent dir created on save
        assert load_watchlist(p) == []                         # missing file -> []

        # bare strings coerce to unfrozen entries: dedupe + upper + strip preserved
        save_watchlist(p, ["nvda", "MSFT", "nvda", " aapl "])
        assert p.exists()
        got = load_watchlist(p)
        assert [e["ticker"] for e in got] == ["NVDA", "MSFT", "AAPL"]
        assert all(e["judged_pivot"] is None and e["pivot_source"] is None for e in got)

        # entry dicts round-trip the frozen-pivot metadata (pivot rounded to cents)
        ents = [make_entry("ebay", 34.119, date_added="2026-07-12",
                           pivot_source="judged", note="clean base"),
                make_entry("BAP")]
        save_watchlist(p, ents)
        got2 = load_watchlist(p)
        assert got2[0] == {"ticker": "EBAY", "judged_pivot": 34.12,
                           "date_added": "2026-07-12", "pivot_source": "judged",
                           "note": "clean base"}
        assert got2[1]["ticker"] == "BAP" and got2[1]["judged_pivot"] is None

        save_watchlist(p, [])                                  # clearing persists an empty list
        assert load_watchlist(p) == []

        # LEGACY file on disk (raw string array) -> migrated in memory, file untouched
        p.write_text(_json.dumps(["EBAY", "BAP"]), encoding="utf-8")
        legacy = load_watchlist(p)
        assert [e["ticker"] for e in legacy] == ["EBAY", "BAP"]
        assert all(e["judged_pivot"] is None for e in legacy)

        p.write_text("{ not json", encoding="utf-8")           # corrupt -> [] (never raises)
        assert load_watchlist(p) == []
        p.write_text('{"a": 1}', encoding="utf-8")             # valid JSON, not a list -> []
        assert load_watchlist(p) == []


def test_watchlist_entry_normalization():
    """make_entry/_coerce_entry: ticker case/strip, float()-coerced pivot (np.float64 in,
    plain json-safe float out), bad pivots -> None, pivot_source only sticks alongside a
    valid pivot, junk elements dropped."""
    import numpy as np
    from src.stock_screener.cockpit.export import make_entry, _coerce_entry

    e = make_entry(" ebay ", np.float64(34.119), date_added="2026-07-12",
                   pivot_source="judged", note=None)
    assert e == {"ticker": "EBAY", "judged_pivot": 34.12, "date_added": "2026-07-12",
                 "pivot_source": "judged", "note": ""}
    assert type(e["judged_pivot"]) is float                    # not np.float64 (json-safe)

    assert make_entry("") is None and make_entry(None) is None
    for bad in (0, -3, "garbage", float("nan"), None):
        assert make_entry("AAA", bad)["judged_pivot"] is None, bad
    # pivot_source never sticks without a valid pivot, or with an unknown source name
    assert make_entry("AAA", None, pivot_source="judged")["pivot_source"] is None
    assert make_entry("AAA", 10.0, pivot_source="bogus")["pivot_source"] is None
    assert make_entry("AAA", 10.0, pivot_source="auto")["pivot_source"] == "auto"

    assert _coerce_entry("bap")["ticker"] == "BAP"             # legacy bare string
    assert _coerce_entry({"ticker": "x", "judged_pivot": 5})["judged_pivot"] == 5.0
    assert _coerce_entry(42) is None and _coerce_entry(None) is None

    # Item 15: dotted class shares adopt the yfinance dash convention at the make_entry
    # choke point (so entries match scan-payload keys, and legacy dotted files heal at load).
    from src.stock_screener.cockpit.export import watchlist_tickers
    assert make_entry("brk.b")["ticker"] == "BRK-B"
    assert _coerce_entry("BRK.B")["ticker"] == "BRK-B"         # legacy file entry heals
    assert watchlist_tickers([{"ticker": "BRK.B"}, "BRK-B"]) == ["BRK-B"]   # collapse to one


def test_watchlist_tickers_projection():
    """watchlist_tickers projects mixed dict/str input to ordered unique upper tickers."""
    from src.stock_screener.cockpit.export import watchlist_tickers
    mixed = [{"ticker": "EBAY", "judged_pivot": 34.12}, "bap", {"ticker": "ebay"},
             42, "", {"ticker": "PECO"}]
    assert watchlist_tickers(mixed) == ["EBAY", "BAP", "PECO"]
    assert watchlist_tickers([]) == [] and watchlist_tickers(None) == []


def test_watchlist_legacy_migration_roundtrip():
    """A legacy string-array file loads as unfrozen entries (the load itself never writes);
    saving what was loaded rewrites the file in the dict schema; re-loading is idempotent."""
    import json as _json
    import tempfile
    from pathlib import Path as _P
    from src.stock_screener.cockpit.export import load_watchlist, save_watchlist

    with tempfile.TemporaryDirectory() as tmp:
        p = _P(tmp) / "watchlist.json"
        p.write_text(_json.dumps(["EBAY", "BAP", "PECO", "EIX"]), encoding="utf-8")
        ents = load_watchlist(p)
        assert [e["ticker"] for e in ents] == ["EBAY", "BAP", "PECO", "EIX"]
        assert isinstance(_json.loads(p.read_text(encoding="utf-8"))[0], str), \
            "load must not rewrite the file"
        save_watchlist(p, ents)                                # first mutation-persist
        raw = _json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(raw[0], dict) and raw[0]["ticker"] == "EBAY"
        assert load_watchlist(p) == ents                       # idempotent


def test_save_watchlist_atomic():
    """save_watchlist writes via a sibling temp file + os.replace: a successful save
    leaves no temp litter, and a failure at the replace step leaves the EXISTING file
    byte-intact (the old truncate-in-place write could be caught mid-write, and
    load_watchlist silently reads a truncated file as [])."""
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import export

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "watchlist.json"
        export.save_watchlist(p, ["NVDA", "MSFT"])
        assert [e["ticker"] for e in export.load_watchlist(p)] == ["NVDA", "MSFT"]
        assert list(Path(tmp).glob("*.tmp")) == []          # no temp litter on success

        before = p.read_text(encoding="utf-8")
        with patch.object(export.os, "replace", side_effect=OSError("boom")):
            export.save_watchlist(p, ["AAPL"])              # swallowed, never raises
        assert p.read_text(encoding="utf-8") == before, "failed save must not touch the file"
        assert list(Path(tmp).glob("*.tmp")) == []          # temp cleaned up on failure


def test_watchlist_merge_frozen_pivots():
    """merge_frozen_pivots (the item-11 lost-update fix): primary keeps membership,
    order, notes, and its own frozen pivots; an UNFROZEN primary entry adopts the
    donor's frozen pivot; donor-only tickers are never resurrected."""
    from src.stock_screener.cockpit.export import make_entry, merge_frozen_pivots

    # App direction: primary = the (stale) session copy, donor = disk, where the
    # trigger job auto-froze KEEP and the user removed GONE in another tab.
    session = [make_entry("KEEP", note="my note"),                     # unfrozen in session
               make_entry("MINE", 50.0, date_added="2026-07-15",
                          pivot_source="judged"),                      # session's own freeze
               make_entry("NEWB")]                                     # added this session
    disk = [make_entry("KEEP", 34.12, date_added="2026-07-17", pivot_source="auto"),
            make_entry("MINE", 99.0, date_added="2026-07-17", pivot_source="auto"),
            make_entry("GONE", 12.0, date_added="2026-07-17", pivot_source="auto")]
    merged = merge_frozen_pivots(session, disk)

    assert [e["ticker"] for e in merged] == ["KEEP", "MINE", "NEWB"]   # membership+order
    by = {e["ticker"]: e for e in merged}
    assert by["KEEP"]["judged_pivot"] == 34.12                         # adopted from disk
    assert by["KEEP"]["pivot_source"] == "auto" and by["KEEP"]["date_added"] == "2026-07-17"
    assert by["KEEP"]["note"] == "my note"                             # note stays primary's
    assert by["MINE"]["judged_pivot"] == 50.0                          # own freeze wins
    assert by["MINE"]["pivot_source"] == "judged"
    assert by["NEWB"]["judged_pivot"] is None                          # nothing to adopt
    assert "GONE" not in by                                            # never resurrected

    # An unfrozen donor entry contributes nothing; mixed legacy strings coerce fine.
    out = merge_frozen_pivots(["aaa"], [make_entry("AAA")])
    assert out[0]["ticker"] == "AAA" and out[0]["judged_pivot"] is None
    assert merge_frozen_pivots([], disk) == []                         # clear stays cleared


def test_parse_ticker_list():
    """The .txt-upload parser tokenizes on commas AND whitespace/newlines, upper-cases,
    drops blanks, and de-duplicates while preserving first-seen order."""
    from src.stock_screener.cockpit.export import parse_ticker_list

    assert parse_ticker_list("aapl, msft\nnvda,,  tsla ") == ["AAPL", "MSFT", "NVDA", "TSLA"]
    assert parse_ticker_list("MSFT,msft , AAPL") == ["MSFT", "AAPL"]   # case-insensitive dedupe
    assert parse_ticker_list("") == [] and parse_ticker_list("  , \n ") == []
    assert parse_ticker_list("BRK.B goog") == ["BRK.B", "GOOG"]        # dots kept, ws-split



if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "watchlist"))
