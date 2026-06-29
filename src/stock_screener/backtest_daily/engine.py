"""The daily event-driven backtest loop.

Per day t: mark-to-market -> regime gate -> EXITS (delist / stop / regime / sell) on
held names -> ENTRIES (scan-cadence + regime + capacity gated, vendored buy signals,
risk-based sizing, cost) -> snapshot. Decide-at-close / fill-at-close. Leak-safe: all
rule calls go through cache.ohlcv_upto(<=t); delisting is realized only on its date.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import regime, signals
from .config import BacktestConfig
from .indicators_cache import IndicatorsCache
from .metrics import build_report
from .portfolio import Portfolio
from .sizing import make_sizer


class BacktestEngine:
    def __init__(self, price, universe, fundamentals, sizer=None, config=None):
        self.price = price
        self.universe = universe
        self.fundamentals = fundamentals
        self.cfg = config or BacktestConfig()
        self.sizer = sizer or make_sizer(self.cfg)

    def run(self, cache=None):
        cfg = self.cfg
        spread = cfg.spread_per_share
        if cache is None:                       # allow a prebuilt cache to be reused across configs
            cache = IndicatorsCache.build(self.price, cfg)
        live = [t for t in cache.calendar
                if (cfg.start is None or t >= cfg.start)
                and (cfg.end is None or t <= cfg.end)]

        pf = Portfolio(cfg.initial_equity)
        records = []
        prev_equity = cfg.initial_equity
        prev_spy = None
        days_since_scan = cfg.scan_every_days        # force a scan on the first eligible day
        regime_on_streak = 0                          # consecutive days the regime has allowed buys
        start_time = time.time()

        for i, t in enumerate(live):
            if cfg.progress_every and i and i % cfg.progress_every == 0:
                el = (time.time() - start_time) / 60.0
                print(f"  [{t.date()}] day {i}/{len(live)}  {el:.1f}min  "
                      f"pos={len(pf.positions)}  equity={pf.equity():,.0f}", flush=True)
            members_t = self.universe.members_asof(t)
            spy_slice = cache.spy_upto(t)
            spy_close = float(spy_slice["Close"].iloc[-1]) if len(spy_slice) else np.nan

            # (1) mark held positions to today's close
            for permno in list(pf.positions):
                pf.mark(permno, cache.close(permno, t))

            # (2) regime gate
            gate, _spy_an, _breadth = regime.assess(t, cache, members_t, spy_slice, cfg)
            regime_on_streak = regime_on_streak + 1 if gate.get("should_generate_buys") else 0

            # (3) EXITS before entries (held names only -> cheap)
            for permno in list(pf.positions):
                pos = pf.positions[permno]

                d = self.price.delisting(permno)
                if d is not None and d[0] == t:                      # forced exit at delist return
                    pf.exit(permno, t, pos.last_price * (1.0 + d[1]), cost=0.0, reason="DELIST")
                    continue

                lo = cache.low(permno, t)
                cl = cache.close(permno, t)
                if cfg.stop_trigger == "low":
                    stop_hit = (not pd.isna(lo)) and lo <= pos.stop
                else:
                    stop_hit = (not pd.isna(cl)) and cl <= pos.stop
                if stop_hit:
                    if cfg.stop_trigger == "low":
                        op = cache.open(permno, t)
                        fill = min(op, pos.stop) if not pd.isna(op) else pos.stop
                    else:
                        fill = cl
                    pf.exit(permno, t, fill, cost=(spread / 2.0) * pos.shares, reason="STOP")
                    continue

                if cfg.exit_on_regime_flip and gate.get("risk_off"):
                    fill = cl if not pd.isna(cl) else pos.last_price
                    pf.exit(permno, t, fill, cost=(spread / 2.0) * pos.shares, reason="REGIME")
                    continue

                sell = signals.evaluate_sell(permno, t, cache, spy_slice, pos.last_phase, cfg)
                if sell is not None and sell.get("is_sell"):
                    fill = cl if not pd.isna(cl) else pos.last_price
                    pf.exit(permno, t, fill, cost=(spread / 2.0) * pos.shares, reason="SELL")
                    continue
                if sell is not None and sell.get("phase") is not None:
                    pos.last_phase = sell["phase"]

            # (4) ENTRIES (scan cadence + regime + capacity + free cash)
            days_since_scan += 1
            equity_now = pf.equity()
            if (days_since_scan >= cfg.scan_every_days
                    and gate.get("should_generate_buys")
                    and regime_on_streak >= cfg.regime_confirm_days       # whipsaw guard (re-entry lag)
                    and len(pf.positions) < cfg.max_positions
                    and pf.cash > cfg.min_free_cash_frac * equity_now):   # skip when ~fully invested
                days_since_scan = 0
                cands = [p for p in members_t
                         if p not in pf.positions and cache.buy_candidate_mask(p, t)]
                if cfg.candidate_cap is not None and len(cands) > cfg.candidate_cap:
                    cands = cands[:cfg.candidate_cap]

                scored = []
                for p in cands:
                    sig = signals.evaluate_buy(
                        p, t, cache, spy_slice, self.fundamentals.fundamentals_asof(p, t), cfg)
                    if sig is not None:
                        scored.append(sig)
                scored.sort(key=lambda s: s.get("score", 0.0), reverse=True)

                slots = cfg.max_positions - len(pf.positions)
                for sig in scored[:slots]:
                    permno = sig["permno"]
                    entry = cache.close(permno, t)
                    if pd.isna(entry) or entry <= 0:
                        continue
                    sig["entry_price"] = float(entry)
                    w = self.sizer.weight(sig, equity_now, len(pf.positions), cfg)
                    invest = min(w * equity_now, pf.cash)
                    if invest <= 0:
                        continue
                    shares = invest / entry
                    cost = (spread / 2.0) * shares
                    if invest + cost > pf.cash:                       # leave room for cost
                        invest = max(0.0, pf.cash - cost)
                        shares = invest / entry if entry > 0 else 0.0
                        cost = (spread / 2.0) * shares
                    if shares <= 0:
                        continue
                    stop = sig.get("stop_loss") or entry * 0.92
                    pf.enter(permno, t, entry, shares, stop, cost, "BUY", 2)
                    if len(pf.positions) >= cfg.max_positions:
                        break

            # (5) snapshot
            equity_t = pf.equity()
            net = (equity_t / prev_equity - 1.0) if prev_equity > 0 else 0.0
            if prev_spy is not None and not pd.isna(spy_close) and prev_spy > 0:
                spy_ret = spy_close / prev_spy - 1.0
            else:
                spy_ret = 0.0
            records.append({
                "Date": t, "equity": equity_t, "net": net, "spy_ret": spy_ret,
                "n_positions": len(pf.positions), "invested": pf.market_value(),
                "cash": pf.cash,
            })
            prev_equity = equity_t
            if not pd.isna(spy_close):
                prev_spy = spy_close

        daily = pd.DataFrame(records).set_index("Date").sort_index()
        report = build_report(daily, pf.trades, cfg)
        return {"daily": daily, "equity": daily["equity"], "report": report,
                "blotter": pd.DataFrame(pf.trades), "n_trades": len(pf.trades)}
