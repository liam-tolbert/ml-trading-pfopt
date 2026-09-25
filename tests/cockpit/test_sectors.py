"""Cockpit tests — sector and industry labels (sectors.py) and the group reads built on them.

Runs standalone (`python -m tests.cockpit.test_sectors`) or as part of the full gate
(`python tests/test_cockpit.py`).
"""
from tests.cockpit._common import *  # noqa: F401,F403


class _FakeTicker:
    def __init__(self, info=None, raises=False):
        self.calls = 0
        self._info, self._raises = info or {}, raises

    @property
    def info(self):
        self.calls += 1
        if self._raises:
            raise RuntimeError("yahoo down")
        return self._info


def test_sectors_cache_round_trip():
    """§6.84 (audit Step-1 #5): a label is fetched once and kept 180 days; a failed or
    empty fetch is remembered for 7, so the nightly screen doesn't hammer Yahoo; without
    network a stale entry is returned rather than nothing; a corrupt file reads empty."""
    import tempfile
    from src.stock_screener.cockpit import sectors

    DAY = 86400.0
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "sectors.json"
        tk = _FakeTicker({"sector": "Technology", "industry": "Semiconductors"})
        got = sectors.get_sector("nvda", tk, path=p, now=1000.0)
        assert got["industry"] == "Semiconductors" and got["sector"] == "Technology"
        assert tk.calls == 1
        assert sectors.read_all(p)["NVDA"]["industry"] == "Semiconductors"
        # fresh: no second fetch
        again = sectors.get_sector("NVDA", tk, path=p, now=1000.0 + 179 * DAY)
        assert again["industry"] == "Semiconductors" and tk.calls == 1
        # stale, no network: the old label, still no fetch
        stale = sectors.get_sector("NVDA", tk, path=p, now=1000.0 + 181 * DAY,
                                   allow_network=False)
        assert stale["industry"] == "Semiconductors" and tk.calls == 1
        # stale with network: refetched
        sectors.get_sector("NVDA", tk, path=p, now=1000.0 + 181 * DAY)
        assert tk.calls == 2

        # a failed fetch is stored as a miss and not retried inside 7 days
        bad = _FakeTicker(raises=True)
        miss = sectors.get_sector("ZZZ", bad, path=p, now=1000.0)
        assert miss["industry"] is None and bad.calls == 1
        sectors.get_sector("ZZZ", bad, path=p, now=1000.0 + 6 * DAY)
        assert bad.calls == 1
        sectors.get_sector("ZZZ", bad, path=p, now=1000.0 + 8 * DAY)
        assert bad.calls == 2
        # an empty info dict counts as a miss too; blank strings are not labels
        blank = _FakeTicker({"sector": "  ", "industry": ""})
        assert sectors.get_sector("BLNK", blank, path=p, now=1000.0)["industry"] is None

        # no network and nothing cached: empty, no fetch attempted
        never = _FakeTicker({"industry": "x"})
        assert sectors.get_sector("NEW", never, path=p, allow_network=False) == \
            {"sector": None, "industry": None, "fetched": None}
        assert never.calls == 0
        assert not list(Path(tmp).glob("*.tmp"))

        p.write_text("{not json", encoding="utf-8")
        assert sectors.read_all(p) == {}
    assert sectors.get_sector("", path=Path(tmp) / "x.json") == \
        {"sector": None, "industry": None, "fetched": None}


def test_screen_universe_rows_carry_industry():
    """§6.84: the scan asks the injected labeller for each Step-1 passer only, and the
    label reaches the candidate rows; without a labeller the column is present and None."""
    prices, spy, _ = _synthetic_slice()
    asked = []

    def labeller(t):
        asked.append(t)
        return {"sector": "Technology", "industry": "Semis" if t[-1] in "02468" else "Banks"}

    res = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0),
                          get_sector=labeller)
    assert set(asked) == set(res.candidates["ticker"]), "labelled a non-passer"
    by = res.candidates.set_index("ticker")
    for t in by.index:
        assert by.loc[t, "industry"] == ("Semis" if t[-1] in "02468" else "Banks")
    bare = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0))
    assert "industry" in bare.candidates.columns
    assert bare.candidates["industry"].isna().all()

    def broken(t):
        raise RuntimeError("yahoo down")
    ok = screen_universe(list(prices), prices, spy, cfg=ScanConfig(min_rs=0.0),
                         get_sector=broken)
    assert len(ok.candidates) == len(bare.candidates) and not ok.errors


def test_leading_groups():
    """§6.84: the banner's leading groups rank industries by tier-A candidates, then new
    highs, then candidates; names without a label are left out; an old table without the
    column gives nothing."""
    import pandas as pd
    from src.stock_screener.cockpit.scan import leading_groups

    cand = pd.DataFrame([
        {"ticker": "A1", "tier": "A", "industry": "Semis", "new_high": True},
        {"ticker": "A2", "tier": "A", "industry": "Semis", "new_high": False},
        {"ticker": "B1", "tier": "A", "industry": "Banks", "new_high": True},
        {"ticker": "B2", "tier": "A", "industry": "Banks", "new_high": True},
        {"ticker": "C1", "tier": "B", "industry": "Oil", "new_high": True},
        {"ticker": "C2", "tier": "B", "industry": "Oil", "new_high": True},
        {"ticker": "C3", "tier": "B", "industry": "Oil", "new_high": True},
        {"ticker": "X", "tier": "A", "industry": None, "new_high": True},
    ])
    g = leading_groups(cand)
    assert [x["industry"] for x in g] == ["Banks", "Semis", "Oil"], g
    assert g[0] == {"industry": "Banks", "n": 2, "tier_a": 2, "new_highs": 2}
    assert len(leading_groups(cand, n=1)) == 1
    assert leading_groups(cand.drop(columns=["industry"])) == []
    assert leading_groups(cand.drop(columns=["new_high"]))[0]["new_highs"] == 0
    assert leading_groups(pd.DataFrame()) == [] and leading_groups(None) == []


def test_industry_concentration_and_leader_break():
    """§6.84: the Positions page warns when three held names, or half of two or more,
    share an industry; and when a top-3-RS scan name in a held name's industry closed
    under its 50-day on >= 1.5x volume (the books' "when the leader sneezes")."""
    import pandas as pd
    from src.stock_screener.cockpit import advisories

    P = lambda s, i: {"symbol": s, "industry": i}                 # noqa: E731
    assert advisories.industry_concentration([P("A", "Semis"), P("B", "Semis"),
                                              P("C", "Semis"), P("D", "Oil")])
    # two of three is two-thirds of the book: the share rule fires
    assert "2 of your 3" in (advisories.industry_concentration(
        [P("A", "Semis"), P("B", "Semis"), P("C", "Oil")]) or "")
    # two of five is 40%, under both rules
    assert advisories.industry_concentration(
        [P("A", "Semis"), P("B", "Semis"), P("C", "Oil"), P("D", "Banks"),
         P("E", "Retail")]) is None
    half = advisories.industry_concentration([P("A", "Semis"), P("B", "Semis")])
    assert half and "2 of your 2 positions are in **Semis**" in half
    assert advisories.industry_concentration([P("A", "Semis"), P("B", "Oil")]) is None
    assert advisories.industry_concentration([P("A", "Semis")]) is None
    assert advisories.industry_concentration([P("A", None), P("B", None)]) is None
    assert advisories.industry_concentration([]) is None

    # leader break: LEAD (top RS) closes under its 50-day on 2x volume -> warning
    closes = [100.0] * 59 + [90.0]
    vols = [1000] * 59 + [2000]
    broke = _trigger_frame("2026-06-30", closes, vols)
    fine = _trigger_frame("2026-06-30", [100.0 + 0.1 * i for i in range(60)])
    cand = pd.DataFrame([{"ticker": "LEAD", "rs": 97, "industry": "Semis"},
                         {"ticker": "PEER", "rs": 90, "industry": "Semis"},
                         {"ticker": "HELD", "rs": 99, "industry": "Semis"},
                         {"ticker": "OTHR", "rs": 95, "industry": "Oil"}])
    payloads = {"LEAD": {"df": broke}, "PEER": {"df": fine}, "HELD": {"df": broke},
                "OTHR": {"df": broke}}
    msg = advisories.group_leader_break("Semis", cand, payloads, exclude="HELD")
    assert msg and msg.startswith("group leader LEAD") and "2.0× volume" in msg
    # a quiet break (normal volume) is not the warning
    quiet = {**payloads, "LEAD": {"df": _trigger_frame("2026-06-30", closes)}}
    assert advisories.group_leader_break("Semis", cand, quiet, exclude="HELD") is None
    assert advisories.group_leader_break("Oil", cand, {"OTHR": {"df": fine}}) is None
    assert advisories.group_leader_break(None, cand, payloads) is None
    assert advisories.group_leader_break("Semis", pd.DataFrame(), payloads) is None
    assert advisories.group_leader_break("Semis", cand.drop(columns=["industry"]),
                                         payloads) is None


def test_positions_page_industry_concentration():
    """§6.84 on the page: three holdings in one industry raise the concentration warning;
    the offline positions without an industry label raise nothing."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        print(f"  SKIP test_positions_page_industry_concentration (AppTest unavailable: {e})")
        return
    import tempfile
    from unittest.mock import patch
    from src.stock_screener.cockpit import cache, trade

    one = _positions_offline()
    base = one["positions"][0]
    three = {"account": {**one["account"], "positions_count": 3},
             "positions": [{**base, "symbol": s, "industry": "Semiconductors"}
                           for s in ("AAA", "BBB", "CCC")]}
    page = str(ROOT / "src" / "stock_screener" / "cockpit" / "pages" / "2_Positions.py")

    def _render(offline):
        with tempfile.TemporaryDirectory() as _tmp, \
                patch.object(trade, "fetch_positions", return_value=offline), \
                patch.object(trade, "fetch_order_fills",
                             side_effect=trade.TradeUnavailable("offline")), \
                patch.object(cache, "WATCHLIST_JSON", Path(_tmp) / "watchlist.json"), \
                patch.object(cache, "TRIGGERS_DIR", Path(_tmp) / "triggers"):
            at = AppTest.from_file(page, default_timeout=60)
            at.run()
        assert not at.exception, f"positions page raised: {at.exception}"
        return " ".join(str(w.value) for w in at.warning)

    assert "3 of your 3 positions are in **Semiconductors**" in _render(three)
    assert "positions are in" not in _render(one)


if __name__ == "__main__":
    raise SystemExit(run_suite(globals(), "sectors"))
