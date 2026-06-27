"""STUB — live CRSP/Compustat providers (the survivorship-free data pull; NEXT task).

These implement the provider interfaces against WRDS. Not yet wired: each raises
NotImplementedError carrying the concrete query plan, so filling them in is a
fill-in-the-blanks job once a WRDS connection is available in this environment.

All ids are PERMNO. Feed the per-name quarterly frames to
``fundamentals_adapter.compustat_to_scorer_dict`` (which enforces the rdq<=t leak gate).
"""
from __future__ import annotations

from .providers import FundamentalsProvider, PriceProvider, UniverseProvider


class WrdsPriceProvider(PriceProvider):
    """Daily CRSP prices + delisting returns.

    TODO:
      - crsp.dsf: permno, date, openprc, askhi, bidlo, prc, vol, ret, cfacpr, shrout
        (abs(prc) for the bid/ask-average sign; split-consistent series via prc/cfacpr;
        returns from `ret`, which already includes dividends).
      - Restrict to common US stock via crsp.dsenames: shrcd in (10,11), exchcd in (1,2,3).
      - Delisting from crsp.dsedelist (dlstdt, dlret); impute -0.30 (NYSE/AMEX) / -0.55
        (NASDAQ) on missing performance-delist codes (Shumway). OR use the CIZ table
        crsp.stkdlysecuritydata whose DlyRet already incorporates the delisting return.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self.__doc__)

    def calendar(self): raise NotImplementedError
    def permnos(self): raise NotImplementedError
    def prices(self, permno): raise NotImplementedError
    def spy(self): raise NotImplementedError
    def delisting(self, permno): raise NotImplementedError


class WrdsUniverseProvider(UniverseProvider):
    """Point-in-time top-N by market cap (broad — include small/mid-cap for Minervini).

    TODO: each rebalance (quarter-end), rank common stocks by abs(prc)*shrout from
    crsp.msf, take the top-N permnos, carry membership forward to daily. Survivorship-free
    by construction: dead names are present for exactly the quarters they qualified.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self.__doc__)

    def members_asof(self, date): raise NotImplementedError


class WrdsFundamentalsProvider(FundamentalsProvider):
    """Compustat fundq linked to PERMNO via CCM, lagged to the report date (rdq).

    TODO: comp.fundq (gvkey, datadate, rdq, revtq, epspxq|epsfxq, invtq) joined to permno
    via crsp.ccmxpf_lnkhist (linktype in (LU,LC), linkprim in (P,C), linkdt<=date<=linkenddt);
    hand each name's quarterly frame to fundamentals_adapter.compustat_to_scorer_dict.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self.__doc__)

    def fundamentals_asof(self, permno, date): raise NotImplementedError
