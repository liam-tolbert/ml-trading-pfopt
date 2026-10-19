"""The weekend hunt's eight base questions, answered from numbers instead of by Claude.

``read_form`` takes one name's ``detect_vcp`` result, the hunt's measured facts
(``hunt.reviewer.fact_sheet``) and its diagnostics row, and returns an answer in the
same shape Claude's answer has, so ``hunt.reviewer.verdict_from_form`` turns it into
PASS / PASS- / FAIL with the same table. It never writes a verdict itself.

What it cannot see it does not guess:

- ``pattern`` is ``vcp`` or ``no_base``. Cups, flat bases, double bottoms and high tight
  flags are parked (HANDOFF_HISTORY, Ideas).
- ``base_count`` is always ``cant_tell``: the base count failed its pre-registered check
  (HANDOFF §6.76).
- ``right_side`` answers ``v_shape`` only while ``V_SHAPE_ENABLED`` is True, which the
  pre-registered check in HANDOFF_HISTORY §6.109 decides.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .doctrine import NO_CHASE_PCT
from .vcp import EXPAND_RATIO

# Set by the pre-registered V check (HANDOFF_HISTORY §6.109). MUST NOT be turned on
# without a new pre-registered check that passes.
V_SHAPE_ENABLED = False
WILD_TR_PCT = 8.0          # a typical day this wide is wider than any sensible stop
HEAVY_DOWN_DAYS = 3        # heavy-volume down days in the base that, with more volume on
                           # down days than up days, read as distribution
GRIND_RATIO = 2.0          # a climb taking more than this × the bars of its fall is slow

QUESTIONS = ("pattern", "swings_match", "right_side", "final_area", "volume", "pivot",
             "character", "base_count")


def read_form(vcp: dict, facts: dict, row: dict) -> dict:
    """The eight answers for one name, with ``evidence`` (one line per question, from
    the numbers), ``swings_seen`` (always empty), ``sure`` (always True) and ``checks``:
    what should send the name to the user, empty when nothing should.

    ``vcp`` is a ``detect_vcp`` result; a key it lacks reads as unknown. ``facts`` needs
    ``heavy_down_days_in_base``, ``up_down_volume_in_base`` and ``dryup_ratio``; ``row``
    needs ``stop_room``. Missing values are read as unknown, never as a pass."""
    ans: Dict[str, str] = {}
    why: Dict[str, str] = {}

    n = int(vcp.get("contraction_count") or 0)
    ans["pattern"] = "vcp" if n >= 2 else "no_base"
    why["pattern"] = f"{n} contracting legs selected"

    zz = [float(g["drawdown_pct"]) for g in vcp.get("zz_legs") or []]
    ans["swings_match"] = "no" if vcp.get("swings_disagree") else "yes"
    why["swings_match"] = ("an earlier swing is deeper than the one before it"
                           if vcp.get("swings_disagree") else "the swings agree with the legs")

    v = vcp.get("v_shape") or {}
    if len(zz) >= 2 and zz[-1] > EXPAND_RATIO * zz[-2]:
        ans["right_side"] = "wider_leg"
        why["right_side"] = f"last swing {zz[-1]:.1f}% after {zz[-2]:.1f}%"
    elif V_SHAPE_ENABLED and v.get("flag"):
        ans["right_side"] = "v_shape"
        why["right_side"] = (f"{v['depth_pct']:.0f}% fall over {v['fall_bars']} days, back "
                             f"in {v['rise_bars']} without a pause")
    else:
        ans["right_side"] = "orderly_tightening"
        why["right_side"] = ("swings " + " -> ".join(f"{d:.1f}" for d in zz[-4:])
                             if zz else "no swings measured")

    area = vcp.get("final_area") or "not_formed"
    ans["final_area"] = area
    why["final_area"] = (f"{vcp.get('area_bars', 0)} days within the tight range"
                         if area != "not_formed" else "the last pullback has not turned")

    heavy = _num(facts.get("heavy_down_days_in_base"))
    updown = _num(facts.get("up_down_volume_in_base"))
    dry = _num(facts.get("dryup_ratio"))
    if heavy is not None and heavy >= HEAVY_DOWN_DAYS and updown is not None and updown < 1.0:
        ans["volume"] = "heavy_selling"
        why["volume"] = f"{heavy:.0f} heavy down days, up/down volume {updown:.2f}"
    elif dry is not None and dry < 1.0:
        ans["volume"] = "drying_up"
        why["volume"] = f"final area at {dry:.2f}x its average volume"
    else:
        ans["volume"] = "mixed"
        why["volume"] = (f"final area at {dry:.2f}x its average volume" if dry is not None
                         else "no dry-up read")

    top, area_high = _num(vcp.get("pivot_price")), _num(vcp.get("area_high"))
    if area == "not_formed" or area_high is None:
        ans["pivot"] = "none"
        why["pivot"] = "no final tight area yet"
    elif top is not None and top > area_high * (1.0 + NO_CHASE_PCT):
        # The buy zone runs NO_CHASE_PCT above the pivot: a base top beyond it is a
        # second level to clear inside any buy the area's high would trigger.
        ans["pivot"] = "several"
        why["pivot"] = f"area high {area_high:.2f} under the base top {top:.2f}"
    else:
        ans["pivot"] = "clear"
        why["pivot"] = f"area high {area_high:.2f}"

    day = _num(vcp.get("median_tr_pct"))
    room = _num(row.get("stop_room"))
    rec = [r for r in vcp.get("recoveries") or [] if r[0] > 0]
    slow = sum(1 for fall, rise in rec if rise > GRIND_RATIO * fall)
    if (day is not None and day >= WILD_TR_PCT) or (room is not None and room < 1.0):
        ans["character"] = "wild_swings"
        why["character"] = (f"typical day {day:.1f}%" if day is not None and day >= WILD_TR_PCT
                            else f"stop {room:.1f} typical days away")
    elif rec and slow * 2 > len(rec):
        ans["character"] = "grinding"
        why["character"] = f"{slow} of {len(rec)} climbs slower than {GRIND_RATIO:.0f}x the fall"
    else:
        ans["character"] = "quick_recoveries"
        why["character"] = (f"{slow} of {len(rec)} climbs slow" if rec
                            else "no complete swing to time")

    ans["base_count"] = "cant_tell"
    why["base_count"] = "not read"

    checks: List[str] = []
    if vcp.get("swings_disagree"):
        checks.append(f"an earlier swing is more than {EXPAND_RATIO:.2f}x the one before it, "
                      "and the selected legs leave it out")
    return {**ans, "evidence": why, "swings_seen": [], "sure": True, "checks": checks}


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f
