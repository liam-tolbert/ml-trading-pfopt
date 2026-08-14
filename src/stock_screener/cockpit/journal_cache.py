"""One ``st.cache_data`` wrapper around the expensive paginated order-history pull,
shared by the scan page's trade panel, the Positions page, and the Journal page — one
network fetch per session serves all three. Every caller passes the same
``st.session_state["jr_nonce"]``, so the Journal page's Refresh busts every consumer at
once. Exceptions are not cached: a credentials fix + Refresh recovers."""
from __future__ import annotations

import streamlit as st

from . import trade


@st.cache_data(show_spinner="Reading the order history…")
def cached_fills(nonce):
    # Reference the MODULE attribute (trade.fetch_order_fills) so a test patch is honored;
    # the nonce is the cache key.
    return trade.fetch_order_fills()
