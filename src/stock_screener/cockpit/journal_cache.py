"""The order history, read page by page from the broker and stored in session_state for
this browser session.

The scan page's trade panel, the Positions page and the Journal page share it. Every
caller passes the same ``st.session_state["jr_nonce"]``, a counter that is bumped to
force a re-read, so the Journal page's Refresh makes all three re-read at once.

``fills_memo`` (the stored read) MUST live in session_state, not in ``st.cache_data``.
That cache is one per server process, and every session starts at ``jr_nonce`` 1, so
every visitor would get the first visitor's order history, never re-read. An exception
is not stored, so fixing the credentials and pressing Refresh recovers."""
from __future__ import annotations

import time

import streamlit as st

from . import trade

FILLS_MAX_AGE_S = 60    # a stored read older than this is re-fetched on the next rerun


def cached_fills(nonce):
    """``trade.fetch_order_fills()`` for this session. Reused while ``nonce`` matches the
    stored one and the read is under ``FILLS_MAX_AGE_S`` old. session_state outlives page switches, so the age
    limit is what catches trades placed elsewhere in the meantime. Exceptions from the
    fetch propagate."""
    memo = st.session_state.get("fills_memo")
    if (memo is not None and memo["nonce"] == nonce
            and time.monotonic() - memo["mono"] < FILLS_MAX_AGE_S):
        return memo["data"]
    with st.spinner("Reading the order history…"):
        data = trade.fetch_order_fills()        # via the module, so a test's patch works
    st.session_state["fills_memo"] = {"nonce": nonce, "mono": time.monotonic(), "data": data}
    return data
