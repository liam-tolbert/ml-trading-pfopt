"""The paginated order-history pull, memoized per browser session.

The scan page's trade panel, the Positions page and the Journal page share it. Every
caller passes the same ``st.session_state["jr_nonce"]``, so the Journal page's Refresh
busts every consumer at once.

The memo MUST live in the session, never in ``st.cache_data``. That cache is one per
server process, and every session starts at jr_nonce 1, so every visitor would get the
first visitor's order history with no expiry. Exceptions are not memoized, so a
credentials fix plus Refresh recovers."""
from __future__ import annotations

import time

import streamlit as st

from . import trade

FILLS_MAX_AGE_S = 60    # a memoized read older than this is re-fetched on the next rerun


def cached_fills(nonce):
    """``trade.fetch_order_fills()`` for this session. Reused while ``nonce`` matches and the
    read is under ``FILLS_MAX_AGE_S`` old. session_state outlives page switches, so the age
    limit is what catches trades placed elsewhere in the meantime."""
    memo = st.session_state.get("fills_memo")
    if (memo is not None and memo["nonce"] == nonce
            and time.monotonic() - memo["mono"] < FILLS_MAX_AGE_S):
        return memo["data"]
    with st.spinner("Reading the order history…"):
        data = trade.fetch_order_fills()        # MODULE attribute, so a test patch is honored
    st.session_state["fills_memo"] = {"nonce": nonce, "mono": time.monotonic(), "data": data}
    return data
