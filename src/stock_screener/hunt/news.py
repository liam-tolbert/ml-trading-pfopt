"""Recent headlines for the hunt's PASS names, from public RSS feeds.

The fetching is deterministic code; the judgment (``catalyst.json``) is the reviewer's,
made from these files alone. The unattended review has no web access, so a headline it
cannot find here does not exist for it. Two feeds per ticker, merged, de-duplicated by
title and ordered newest first: Yahoo Finance's per-ticker feed and a Google News search.
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import json
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import requests

NEWS_DIR = "news"
LIMIT = 12
TIMEOUT_S = 15
FEEDS = (
    ("Yahoo Finance",
     "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"),
    ("Google News",
     "https://news.google.com/rss/search?q={t}+stock&hl=en-US&gl=US&ceid=US:en"),
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (sepa-cockpit weekend hunt)"}
_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def _get(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT_S, headers=_HEADERS)
    r.raise_for_status()
    return r.content


def _clean(text: Optional[str], limit: int = 300) -> str:
    s = _SPACES.sub(" ", _html.unescape(_TAGS.sub(" ", text or ""))).strip()
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


def _date(text: Optional[str]) -> str:
    """An RSS pubDate as ``YYYY-MM-DD``; empty when unparseable."""
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return ""


def parse_feed(xml_bytes: bytes, default_publisher: str) -> List[dict]:
    """The items of one RSS document as ``{title, publisher, date, url, summary}``.

    A Google News title carries its publisher as a `` - Publisher`` suffix, which is
    dropped when the item names its source. A document that is not RSS yields ``[]``."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title = _clean(it.findtext("title"), 200)
        if not title:
            continue
        publisher = _clean(it.findtext("source"), 80) or default_publisher
        suffix = f" - {publisher}"
        if it.findtext("source") and title.endswith(suffix):
            title = title[:-len(suffix)].rstrip()
        out.append({"title": title, "publisher": publisher,
                    "date": _date(it.findtext("pubDate")),
                    "url": (it.findtext("link") or "").strip(),
                    "summary": _clean(it.findtext("description"))})
    return out


def _merge(items: Iterable[dict], limit: int) -> List[dict]:
    seen, out = set(), []
    for h in sorted(items, key=lambda h: h["date"], reverse=True):
        key = re.sub(r"[^a-z0-9]+", " ", h["title"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def fetch_news(tickers: Iterable[str], limit: int = LIMIT,
               fetch: Callable[[str], bytes] = _get) -> Dict[str, dict]:
    """``{ticker: {ticker, fetched_at, headlines, errors}}`` for every ticker.

    A feed that fails adds a line to ``errors`` and contributes nothing; a ticker whose
    feeds all fail has an empty ``headlines`` list. Never raises."""
    out: Dict[str, dict] = {}
    now = _dt.datetime.now().isoformat(timespec="seconds")
    for t in tickers:
        items, errors = [], []
        for publisher, url in FEEDS:
            try:
                items += parse_feed(fetch(url.format(t=t)), publisher)
            except Exception as e:           # a feed down MUST NOT stop the hunt
                errors.append(f"{publisher}: {type(e).__name__}: {str(e)[:120]}")
        out[t] = {"ticker": t, "fetched_at": now, "headlines": _merge(items, limit),
                  "errors": errors}
    return out


def write_news(hunt_path: Path, news: Dict[str, dict]) -> List[Path]:
    """Write one ``news/<ticker>.json`` per entry into ``hunt_path``. Returns the paths."""
    d = hunt_path / NEWS_DIR
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for t, rec in news.items():
        p = d / f"{t}.json"
        p.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(p)
    return paths
