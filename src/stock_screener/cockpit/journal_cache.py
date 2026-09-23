"""The paginated order-history pull, memoized per browser session — shared by the scan page's
trade panel, the Positions page, and the Journal page. Every caller passes the same
``st.session_state["jr_nonce"]``, so the Journal page's Refresh busts every consumer at once.

Per SESSION, never ``st.cache_data``: that cache is one per server process, and every session
starts at jr_nonce 1, so every new visitor was handed the first visitor's order history with no
expiry — a new buy stayed out of the Positions page's P1 entry dates and the Journal until
someone pressed Refresh here. Exceptions are not memoized: a credentials fix + Refresh recovers."""
from __future__ import annotations

import time

import streamlit as st

from . import trade

FILLS_MAX_AGE_S = 60    # a memoized read older than this is re-fetched on the next rerun


def cached_fills(nonce):
    """``trade.fetch_order_fills()`` for this session, reused while ``nonce`` matches and the
    read is under FILLS_MAX_AGE_S old (session_state outlives page switches, so age is what
    catches "came back after trading elsewhere")."""
    memo = st.session_state.get("fills_memo")
    if (memo is not None and memo["nonce"] == nonce
            and time.monotonic() - memo["mono"] < FILLS_MAX_AGE_S):
        return memo["data"]
    with st.spinner("Reading the order history…"):
        data = trade.fetch_order_fills()        # MODULE attribute, so a test patch is honored
    st.session_state["fills_memo"] = {"nonce": nonce, "mono": time.monotonic(), "data": data}
    return data
