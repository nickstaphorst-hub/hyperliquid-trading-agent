"""Hyperliquid trading agent: pure analysis core and paper-trading engine.

Everything in this module is deterministic and stdlib-only: parsing of
Hyperliquid public API responses, technical indicators, Fear & Greed and RSS
news parsing (same pattern as the crypto-market-scan agent), plus a complete
paper-trading account engine. Network calls live in the workflow activities
(``hyperliquid_agent.py``); this module only transforms data, so it is fully
unit-testable without any network access.

The paper engine approximates Hyperliquid perp mechanics:
- taker fee of 0.045% of notional on open and close;
- margin-based positions: opening locks ``margin`` in the position, closing
  returns ``margin + pnl - fee``;
- approximate liquidation price at ``entry * (1 -+ 0.9 / leverage)``;
- funding is not simulated (documented limitation);
- every order fills at the live mid price (paper, no slippage).
"""

from __future__ import annotations

import html
import math
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree as ET

# -------------------------------------------------------------------------- 
# Constants
# --------------------------------------------------------------------------

TAKER_FEE_RATE = 0.00045  # Hyperliquid taker fee, ~4.5 bps of notional
LIQ_BUFFER = 0.9  # fraction of the 1/leverage distance used as liq distance
MAX_NEWS_TITLE_LEN = 200
MAX_TRADES_KEPT = 50
PROMPT_CANDLES = 24  # candles per coin offered to the model

INTERVALS = ("15m", "1h", "4h", "1d")

# Free public RSS feeds (no API key). Fetched in an activity; parsed here.
NEWS_FEEDS: dict[str, str] = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
}

_SENTIMENT_POSITIVE = frozenset(
    {
        "surge", "surges", "surged", "surging",
        "soar", "soars", "soared", "soaring",
        "rally", "rallies", "rallied", "rallying",
        "bullish", "bull",
        "record", "highs", "ath",
        "gain", "gains", "gained", "gaining",
        "jump", "jumps", "jumped", "jumping",
        "boost", "boosts", "boosted", "boosting",
        "adoption", "adopts", "adopt",
        "approve", "approves", "approved", "approval",
        "inflow", "inflows",
        "upgrade", "upgrades", "upgraded",
        "partnership", "partnerships",
        "buy", "buys", "buying", "accumulation",
        "breakout", "breakouts",
        "recovery", "recovers", "recovered",
        "rise", "rises", "rising", "rose",
        "climb", "climbs", "climbing", "climbed",
        "uptrend", "green",
    }
)

_SENTIMENT_NEGATIVE = frozenset(
    {
        "crash", "crashes", "crashed", "crashing",
        "plunge", "plunges", "plunged", "plunging",
        "tumble", "tumbles", "tumbled", "tumbling",
        "slump", "slumps", "slumped", "slumping",
        "bearish", "bear",
        "loss", "losses",
        "hack", "hacks", "hacked", "hacking",
        "exploit", "exploits", "exploited",
        "breach", "breaches", "breached",
        "theft", "stolen",
        "sell-off", "selloff", "sell", "sells", "selling",
        "ban", "bans", "banned",
        "lawsuit", "sues", "sued",
        "fraud", "scam", "scams",
        "dump", "dumps", "dumped", "dumping",
        "fear", "fears",
        "liquidation", "liquidations",
        "outflow", "outflows",
        "warn", "warns", "warning", "warnings",
        "decline", "declines", "declined", "declining",
        "drop", "drops", "dropped", "dropping",
        "fall", "falls", "fell", "falling",
        "downtrend", "risk-off",
    }
)


# --------------------------------------------------------------------------
# Technical indicators
# --------------------------------------------------------------------------

def sma(values: list[float], period: int) -> float | None:
    """Simple moving average over the last ``period`` values."""
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder RSI over the last ``period`` closes."""
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def volatility_pct(closes: list[float], lookback: int = 30) -> float | None:
    """Stddev of the last ``lookback`` close-to-close returns, in percent."""
    if len(closes) < lookback + 1:
        return None
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(len(closes) - lookback, len(closes))
    ]
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(var) * 100.0


def classify_trend(price: float, sma_20: float | None, sma_50: float | None) -> str:
    """Dutch trend label based on price vs two SMAs."""
    if sma_20 is None or sma_50 is None:
        return "onbekend"
    if price > sma_20 > sma_50:
        return "stijgend"
    if price < sma_20 < sma_50:
        return "dalend"
    return "zijwaarts"


# --------------------------------------------------------------------------
# Hyperliquid public API parsing (POST /info, no key required)
# --------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    """Best-effort float conversion; None for junk instead of a raise."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def parse_all_mids(payload: dict[str, Any] | None) -> dict[str, float]:
    """``{"type":"allMids"}`` -> ``{coin: mid}``, only positive finite mids."""
    result: dict[str, float] = {}
    for key, value in (payload or {}).items():
        name = str(key)
        if name.startswith(("@", "#")) or not re.fullmatch(r"[A-Z0-9]{1,10}", name):
            continue  # spot pairs and other instruments; we only want perps
        mid = _to_float(value)
        if mid is not None and mid > 0:
            result[name] = mid
    return result


def parse_meta_and_ctxs(
    payload: list[Any] | None,
    coins: list[str],
) -> dict[str, dict[str, Any]]:
    """``{"type":"metaAndAssetCtxs"}`` -> per requested coin a market row.

    ``meta["universe"]`` and the ``assetCtxs`` list are index-aligned; both
    must exist for a coin to be included. Coins missing from the universe
    simply get no row (the caller treats them as untradeable).
    """
    if not isinstance(payload, list) or len(payload) < 2:
        return {}
    meta = payload[0] if isinstance(payload[0], dict) else {}
    ctxs = payload[1] if isinstance(payload[1], list) else []
    universe = meta.get("universe") or []
    rows: dict[str, dict[str, Any]] = {}
    wanted = {str(c).upper() for c in coins}
    for index, entry in enumerate(universe):
        name = str(entry.get("name", "")).upper()
        if name not in wanted or index >= len(ctxs):
            continue
        ctx = ctxs[index] if isinstance(ctxs[index], dict) else {}
        mark = _to_float(ctx.get("markPx"))
        prev = _to_float(ctx.get("prevDayPx"))
        day_change = None
        if mark is not None and prev is not None and prev > 0:
            day_change = (mark - prev) / prev * 100.0
        rows[name] = {
            "coin": name,
            "mark_px": mark,
            "mid_px": _to_float(ctx.get("midPx")),
            "funding": _to_float(ctx.get("funding")),
            "open_interest": _to_float(ctx.get("openInterest")),
            "day_ntl_vlm": _to_float(ctx.get("dayNtlVlm")),
            "prev_day_px": prev,
            "day_change_pct": day_change,
            "max_leverage": _to_int(entry.get("maxLeverage")),
        }
    return rows


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_candles(rows: list[Any] | None) -> list[dict[str, Any]]:
    """``candleSnapshot`` rows -> normalized candles, oldest first."""
    result: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        t = _to_int(row.get("t"))
        o = _to_float(row.get("o"))
        h = _to_float(row.get("h"))
        low = _to_float(row.get("l"))
        c = _to_float(row.get("c"))
        v = _to_float(row.get("v"))
        if None in (t, o, h, low, c) or v is None:
            continue
        result.append(
            {"t": t, "o": o, "h": h, "l": low, "c": c, "v": v}
        )
    result.sort(key=lambda candle: candle["t"])
    return result


def build_market_rows(
    mids: dict[str, float],
    meta_rows: dict[str, dict[str, Any]],
    candles: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Combine mids, meta and candles into one analysis row per coin.

    A coin needs at least a current mid price to be included; indicators are
    added whenever enough candles exist. All values are rounded for prompt
    and report use.
    """
    rows: list[dict[str, Any]] = []
    for coin in sorted(mids):
        price = mids[coin]
        meta = meta_rows.get(coin, {})
        candle_list = candles.get(coin, [])
        closes = [candle["c"] for candle in candle_list]
        sma_20 = sma(closes, 20)
        sma_50 = sma(closes, 50)
        rows.append(
            {
                "coin": coin,
                "price": price,
                "day_change_pct": meta.get("day_change_pct"),
                "funding": meta.get("funding"),
                "open_interest": meta.get("open_interest"),
                "day_ntl_vlm": meta.get("day_ntl_vlm"),
                "max_leverage": meta.get("max_leverage"),
                "rsi_14": round(rsi(closes), 1) if rsi(closes) is not None else None,
                "sma_20": round(sma_20, 4) if sma_20 is not None else None,
                "sma_50": round(sma_50, 4) if sma_50 is not None else None,
                "volatility_pct": (
                    round(volatility_pct(closes), 2)
                    if volatility_pct(closes) is not None
                    else None
                ),
                "trend": classify_trend(price, sma_20, sma_50),
                "candles": len(candle_list),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Fear & Greed (alternative.me) and news (RSS), same pattern as the scan agent
# --------------------------------------------------------------------------

def parse_fear_greed(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize the alternative.me Fear & Greed response; None if unusable."""
    data = (payload or {}).get("data") or []
    values: list[int] = []
    for entry in data:
        try:
            values.append(int(str(entry.get("value")).strip()))
        except (TypeError, ValueError):
            values.append(-1)
    if not data or values[0] < 0:
        return None

    def _value_at(index: int) -> int | None:
        if index >= len(values) or values[index] < 0:
            return None
        return values[index]

    value = values[0]
    yesterday = _value_at(1)
    last_week = _value_at(7)
    return {
        "value": value,
        "classification": str(data[0].get("value_classification", "onbekend")),
        "value_yesterday": yesterday,
        "value_last_week": last_week,
        "change_1d": value - yesterday if yesterday is not None else None,
        "change_7d": value - last_week if last_week is not None else None,
    }


def strip_html(text: str) -> str:
    """Remove HTML tags/entities and collapse whitespace in feed text."""
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    cleaned = html.unescape(cleaned)
    return " ".join(cleaned.split())


def score_headline_sentiment(title: str) -> int:
    """Deterministic keyword score for one headline, clamped to [-2, +2]."""
    tokens = re.findall(r"[a-z0-9'-]+", (title or "").lower())
    score = 0
    for token in tokens:
        if token in _SENTIMENT_POSITIVE:
            score += 1
        elif token in _SENTIMENT_NEGATIVE:
            score -= 1
    return max(-2, min(2, score))


def sentiment_label(score: int) -> str:
    """Dutch label for a headline score."""
    if score > 0:
        return "positief"
    if score < 0:
        return "negatief"
    return "neutraal"


def parse_rss_feed(xml_text: str, source: str) -> list[dict[str, Any]]:
    """Parse one RSS 2.0 feed body; malformed input never raises."""
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return []
    items: list[dict[str, Any]] = []
    for element in root.findall(".//item"):
        raw_title = strip_html(element.findtext("title") or "")
        if not raw_title:
            continue
        url = (element.findtext("link") or element.findtext("guid") or "").strip()
        published: str | None = None
        try:
            published = parsedate_to_datetime(
                (element.findtext("pubDate") or "").strip()
            ).isoformat()
        except (TypeError, ValueError):
            published = None
        items.append(
            {
                "title": raw_title[:MAX_NEWS_TITLE_LEN],
                "url": url,
                "published": published,
                "source": source,
            }
        )
    return items


def merge_news(
    feeds: dict[str, list[dict[str, Any]]],
    max_items: int = 10,
    lookback_hours: int = 48,
) -> list[dict[str, Any]]:
    """Deduplicate, age-filter (relative to newest item), sort and score."""
    merged: list[dict[str, Any]] = []
    for items in feeds.values():
        merged.extend(items)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in merged:
        key = re.sub(r"[^a-z0-9]+", "", item["title"].lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    unique.sort(key=lambda it: it.get("published") or "", reverse=True)

    newest_iso = next((it.get("published") for it in unique if it.get("published")), None)
    if newest_iso is not None:
        try:
            cutoff = datetime.fromisoformat(newest_iso) - timedelta(hours=lookback_hours)
            age_ok: list[dict[str, Any]] = []
            for item in unique:
                iso = item.get("published")
                if not iso or datetime.fromisoformat(iso) >= cutoff:
                    age_ok.append(item)
            unique = age_ok
        except ValueError:  # pragma: no cover - parse_rss_feed guarantees ISO
            pass

    result: list[dict[str, Any]] = []
    for item in unique[: max(1, max_items)]:
        score = score_headline_sentiment(item["title"])
        result.append(
            {
                "title": item["title"],
                "source": item.get("source", "onbekend"),
                "published": item.get("published"),
                "url": item.get("url", ""),
                "sentiment_score": score,
                "sentiment": sentiment_label(score),
            }
        )
    return result


def summarize_news_sentiment(news: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate the per-headline scores into one summary block."""
    if not news:
        return None
    scores = [int(item.get("sentiment_score", 0)) for item in news]
    positive = sum(1 for s in scores if s > 0)
    negative = sum(1 for s in scores if s < 0)
    neutral = len(scores) - positive - negative
    avg = round(sum(scores) / len(scores), 2)
    if avg >= 0.5:
        label = "positief"
    elif avg <= -0.5:
        label = "negatief"
    else:
        label = "neutraal"
    return {
        "average_score": avg,
        "label": label,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "total": len(scores),
    }


def coin_mention_counts(
    news: list[dict[str, Any]],
    coins: list[str],
) -> list[dict[str, Any]]:
    """Count which universe coins appear in the headlines (word boundary)."""
    if not news:
        return []
    titles = [item.get("title", "").lower() for item in news]
    counts: list[dict[str, Any]] = []
    for coin in coins:
        key = str(coin).lower()
        if len(key) < 3:
            continue
        regex = re.compile(r"\b" + re.escape(key) + r"(?:s|es)?\b")
        count = sum(1 for title in titles if regex.search(title))
        if count > 0:
            counts.append({"coin": str(coin), "count": count})
    counts.sort(key=lambda c: (-c["count"], c["coin"]))
    return counts


# --------------------------------------------------------------------------
# Paper account engine (pure, deterministic)
# --------------------------------------------------------------------------

def new_account(start_balance_usd: float, now_iso: str) -> dict[str, Any]:
    """Fresh paper account state (JSON-serializable, engine-owned)."""
    balance = float(start_balance_usd)
    return {
        "started_at": now_iso,
        "balance": balance,
        "positions": {},
        "trades": [],
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "day_key": day_key(now_iso),
        "day_start_equity": balance,
        "equity_high": balance,
        "cycles": 0,
        "last_cycle_at": None,
        "halted": False,
        "halt_reason": None,
    }


def day_key(now_iso: str) -> str:
    """UTC date (YYYY-MM-DD) of an ISO timestamp; marks the trading day."""
    try:
        return datetime.fromisoformat(now_iso).astimezone(timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        return "1970-01-01"


def liq_price_for(entry_price: float, side: str, leverage: int) -> float:
    """Approximate liquidation price at ~90% of the 1/leverage margin distance."""
    distance = entry_price * (LIQ_BUFFER / max(1, leverage))
    if side == "short":
        return entry_price + distance
    return entry_price - distance


def position_unrealized(position: dict[str, Any], price: float) -> float:
    """Unrealized PnL of one position at ``price`` (excludes fees/funding)."""
    direction = 1.0 if position["side"] == "long" else -1.0
    return (price - position["entry_price"]) * position["size"] * direction


def account_equity(account: dict[str, Any], prices: dict[str, float]) -> float:
    """Free balance plus margin plus unrealized PnL of open positions."""
    equity = account["balance"]
    for coin, position in account["positions"].items():
        price = prices.get(coin)
        equity += position["margin"]
        if price is not None:
            equity += position_unrealized(position, price)
    return equity


def available_balance(account: dict[str, Any]) -> float:
    """Balance not locked in positions."""
    return account["balance"]


def day_pnl_pct(account: dict[str, Any], prices: dict[str, float]) -> float:
    """Equity change since the start of the current trading day, percent."""
    start = account.get("day_start_equity") or 0.0
    if start <= 0:
        return 0.0
    return (account_equity(account, prices) - start) / start * 100.0


def circuit_breaker_tripped(
    account: dict[str, Any],
    prices: dict[str, float],
    daily_loss_limit_pct: float,
) -> bool:
    """True when today's equity loss reached the configured limit."""
    return day_pnl_pct(account, prices) <= -abs(float(daily_loss_limit_pct))


def coin_max_leverage(meta_rows: dict[str, dict[str, Any]], coin: str) -> int | None:
    """Max leverage Hyperliquid allows for a coin, from meta; None unknown."""
    value = (meta_rows.get(coin) or {}).get("max_leverage")
    return int(value) if value else None


def validate_action(
    action: dict[str, Any],
    account: dict[str, Any],
    prices: dict[str, float],
    params: dict[str, Any],
    meta_rows: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    """Deterministic pre-trade risk check; returns (ok, reason).

    Every rule is enforced by the engine, never trusted to the model: the
    model proposes, this function disposes. ``params`` carries
    ``max_leverage``, ``max_margin_fraction``, ``max_open_positions``,
    ``daily_loss_limit_pct`` and ``universe``.
    """
    kind = str(action.get("action", "")).lower()
    coin = str(action.get("coin", "")).upper()

    if kind == "none":
        return False, "geen actie"
    if coin not in params["universe"]:
        return False, f"coin {coin} zit niet in het universum"
    if coin not in prices or prices[coin] <= 0:
        return False, f"geen geldige prijs voor {coin}"

    if kind == "close":
        if coin not in account["positions"]:
            return False, f"geen open positie in {coin} om te sluiten"
        return True, "ok"

    if kind != "open":
        return False, f"onbekende actie: {kind}"

    side = str(action.get("direction", "")).lower()
    if side not in ("long", "short"):
        return False, f"richting moet long of short zijn, got: {side}"
    if coin in account["positions"]:
        return False, f"er staat al een positie open in {coin}"
    if len(account["positions"]) >= int(params["max_open_positions"]):
        return False, f"max {params['max_open_positions']} open posities bereikt"
    if account.get("halted"):
        return False, f"nieuwe entries geblokkeerd: {account.get('halt_reason', 'circuit breaker')}"

    leverage = action.get("leverage")
    if not isinstance(leverage, int) or isinstance(leverage, bool):
        return False, "leverage moet een geheel getal zijn"
    if leverage < 1:
        return False, "leverage moet minimaal 1 zijn"
    cap = int(params["max_leverage"])
    coin_cap = coin_max_leverage(meta_rows, coin)
    if coin_cap is not None:
        cap = min(cap, coin_cap)
    if leverage > cap:
        return False, f"leverage {leverage} boven limiet {cap} voor {coin}"

    margin_fraction = action.get("margin_fraction")
    if not isinstance(margin_fraction, (int, float)) or isinstance(margin_fraction, bool):
        return False, "margin_fraction moet een getal zijn"
    margin_fraction = float(margin_fraction)
    if margin_fraction <= 0 or margin_fraction > float(params["max_margin_fraction"]):
        return False, (
            f"margin_fraction {margin_fraction} buiten (0, {params['max_margin_fraction']}]"
        )

    if circuit_breaker_tripped(account, prices, params["daily_loss_limit_pct"]):
        return False, "dagverlieslimiet bereikt: geen nieuwe entries vandaag"

    equity = account_equity(account, prices)
    margin_usd = margin_fraction * equity
    if margin_usd > available_balance(account):
        return False, "vrije balans dekt de marge niet"

    stop = action.get("stop_loss")
    if stop is not None:
        stop = _to_float(stop)
        if stop is None or stop <= 0:
            return False, "stop_loss moet een positief getal zijn"
        if side == "long" and stop >= prices[coin]:
            return False, "stop_loss voor een long moet onder de prijs liggen"
        if side == "short" and stop <= prices[coin]:
            return False, "stop_loss voor een short moet boven de prijs liggen"
    target = action.get("take_profit")
    if target is not None:
        target = _to_float(target)
        if target is None or target <= 0:
            return False, "take_profit moet een positief getal zijn"
        if side == "long" and target <= prices[coin]:
            return False, "take_profit voor een long moet boven de prijs liggen"
        if side == "short" and target >= prices[coin]:
            return False, "take_profit voor een short moet onder de prijs liggen"
    return True, "ok"


def _record_trade(account: dict[str, Any], trade: dict[str, Any]) -> None:
    """Append a trade and trim the log to the last MAX_TRADES_KEPT entries."""
    account["trades"].append(trade)
    if len(account["trades"]) > MAX_TRADES_KEPT:
        del account["trades"][:-MAX_TRADES_KEPT]


def open_position(
    account: dict[str, Any],
    action: dict[str, Any],
    price: float,
    now_iso: str,
) -> dict[str, Any]:
    """Execute a validated open at ``price``; mutates the account.

    ``action["_equity"]`` is injected by :func:`apply_actions` so sizing uses
    the up-to-date equity at execution time.
    """
    coin = str(action["coin"]).upper()
    side = str(action["direction"]).lower()
    leverage = int(action["leverage"])
    margin = float(action["margin_fraction"]) * float(action["_equity"])
    notional = margin * leverage
    size = notional / price
    fee = notional * TAKER_FEE_RATE
    account["balance"] -= margin + fee
    account["fees_paid"] += fee
    stop = _to_float(action.get("stop_loss"))
    target = _to_float(action.get("take_profit"))
    position = {
        "coin": coin,
        "side": side,
        "size": size,
        "entry_price": price,
        "leverage": leverage,
        "margin": margin,
        "liq_price": liq_price_for(price, side, leverage),
        "stop_loss": stop,
        "take_profit": target,
        "opened_at": now_iso,
    }
    account["positions"][coin] = position
    _record_trade(
        account,
        {
            "coin": coin,
            "side": side,
            "action": "open",
            "price": price,
            "size": size,
            "notional": notional,
            "leverage": leverage,
            "margin": margin,
            "pnl": None,
            "fee": fee,
            "at": now_iso,
            "reason": str(action.get("rationale", ""))[:300],
        },
    )
    return {"type": "open", "coin": coin, "side": side, "price": price,
            "size": size, "leverage": leverage, "margin": margin, "fee": fee}


def close_position(
    account: dict[str, Any],
    coin: str,
    price: float,
    now_iso: str,
    reason: str,
) -> dict[str, Any]:
    """Close a position at ``price`` and settle margin + PnL - fee."""
    position = account["positions"].get(coin)
    if position is None:
        return {"type": "skipped", "coin": coin, "reason": "positie bestaat niet"}
    pnl = position_unrealized(position, price)
    notional = abs(price * position["size"])
    fee = notional * TAKER_FEE_RATE
    account["balance"] += position["margin"] + pnl - fee
    account["fees_paid"] += fee
    account["realized_pnl"] += pnl
    del account["positions"][coin]
    _record_trade(
        account,
        {
            "coin": coin,
            "side": position["side"],
            "action": reason,
            "price": price,
            "size": position["size"],
            "notional": notional,
            "leverage": position["leverage"],
            "margin": position["margin"],
            "pnl": round(pnl, 2),
            "fee": fee,
            "at": now_iso,
            "reason": position.get("close_reason", ""),
        },
    )
    return {"type": "close", "coin": coin, "price": price, "pnl": pnl,
            "fee": fee, "reason": reason}


def apply_actions(
    account: dict[str, Any],
    actions: list[dict[str, Any]],
    prices: dict[str, float],
    now_iso: str,
    params: dict[str, Any],
    meta_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and execute the model's proposed actions for one cycle.

    Closes are processed before opens (frees margin), each action is
    validated independently, and rejected actions become "rejected" events
    instead of raising. ``_equity`` is injected per action so open sizing
    uses the up-to-date equity.
    """
    events: list[dict[str, Any]] = []
    ordered = sorted(
        actions,
        key=lambda a: 0 if str(a.get("action", "")).lower() == "close" else 1,
    )
    for action in ordered:
        ok, reason = validate_action(action, account, prices, params, meta_rows)
        if not ok:
            if str(action.get("action", "")).lower() == "none":
                continue
            events.append(
                {"type": "rejected", "coin": str(action.get("coin", "")),
                 "action": str(action.get("action", "")), "reason": reason}
            )
            continue
        coin = str(action["coin"]).upper()
        if str(action["action"]).lower() == "close":
            events.append(
                close_position(account, coin, prices[coin], now_iso, "close")
            )
        else:
            action = dict(action)
            action["_equity"] = account_equity(account, prices)
            events.append(
                open_position(account, action, prices[coin], now_iso)
            )
    return events


def process_forced_exits(
    account: dict[str, Any],
    prices: dict[str, float],
    now_iso: str,
) -> list[dict[str, Any]]:
    """Check every open position against liq, stop and target at current price.

    - Liquidation: price crossed the approximate liq price -> the full
      margin is lost (pnl = -margin), trade action "liquidated".
    - Stop loss / take profit: closed at the stop/target level.
    """
    events: list[dict[str, Any]] = []
    for coin in sorted(list(account["positions"])):
        position = account["positions"].get(coin)
        if position is None:
            continue
        price = prices.get(coin)
        if price is None:
            continue
        long = position["side"] == "long"
        liq = position["liq_price"]
        stop = position.get("stop_loss")
        target = position.get("take_profit")
        if (long and price <= liq) or (not long and price >= liq):
            # Liquidation: the position's margin is lost entirely. The locked
            # margin simply never returns to the balance (no extra fee).
            account["balance"] += 0.0
            account["realized_pnl"] -= position["margin"]
            del account["positions"][coin]
            _record_trade(
                account,
                {
                    "coin": coin,
                    "side": position["side"],
                    "action": "liquidated",
                    "price": liq,
                    "size": position["size"],
                    "notional": abs(liq * position["size"]),
                    "leverage": position["leverage"],
                    "margin": position["margin"],
                    "pnl": round(-position["margin"], 2),
                    "fee": 0.0,
                    "at": now_iso,
                    "reason": "liquidatieprijs benaderd (gesimuleerd)",
                },
            )
            events.append(
                {"type": "liquidated", "coin": coin, "price": liq,
                 "pnl": -position["margin"], "reason": "liquidated"}
            )
            continue
        if stop is not None and ((long and price <= stop) or (not long and price >= stop)):
            account["positions"][coin]["close_reason"] = "stop-loss geraakt"
            events.append(close_position(account, coin, stop, now_iso, "stop"))
            continue
        if target is not None and (
            (long and price >= target) or (not long and price <= target)
        ):
            account["positions"][coin]["close_reason"] = "take-profit geraakt"
            events.append(close_position(account, coin, target, now_iso, "target"))
    return events


def roll_day_if_needed(
    account: dict[str, Any],
    now_iso: str,
    prices: dict[str, float],
) -> bool:
    """Start a new trading day at a UTC date change: reset the day baseline.

    Also lifts the daily-loss halt: a new day gets a fresh loss limit.
    """
    key = day_key(now_iso)
    if key == account["day_key"]:
        return False
    account["day_key"] = key
    account["day_start_equity"] = account_equity(account, prices)
    if account.get("halted"):
        account["halted"] = False
        account["halt_reason"] = None
    return True


def update_circuit_breaker(
    account: dict[str, Any],
    prices: dict[str, float],
    daily_loss_limit_pct: float,
) -> bool:
    """Trip the entries-halt when the daily loss limit is hit; idempotent."""
    if account.get("halted"):
        return True
    if circuit_breaker_tripped(account, prices, daily_loss_limit_pct):
        account["halted"] = True
        account["halt_reason"] = (
            f"dagverlieslimiet {-abs(float(daily_loss_limit_pct)):.0f}% bereikt"
        )
        return True
    return False


# --------------------------------------------------------------------------
# Trade history and learning (deterministic, in-context)
# --------------------------------------------------------------------------

MAX_JOURNAL_TRADES = 12  # closed trades shown in the prompt journal
MAX_LESSONS = 4  # deterministic coaching rules kept in the prompt

_EXIT_LABELS = {
    "close": "model-close",
    "stop": "stop-loss",
    "target": "take-profit",
    "liquidated": "LIQUIDATIE",
}


def pair_round_trips(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each opening trade with its closing trade per coin.

    Trades with ``action == "open"`` open a round trip; the first trade of
    the same coin that carries a ``pnl`` (close, stop, target, liquidated)
    closes it. Closes without a recorded open are kept with ``entry=None``
    so no information is silently dropped.
    """
    pending: dict[str, list[dict[str, Any]]] = {}
    pairs: list[dict[str, Any]] = []
    for trade in trades or []:
        action = str(trade.get("action", ""))
        coin = str(trade.get("coin", ""))
        if action == "open":
            pending.setdefault(coin, []).append(trade)
        elif trade.get("pnl") is not None:
            entry = pending.get(coin)
            if entry:
                pairs.append({"entry": entry.pop(0), "exit": trade})
            else:
                pairs.append({"entry": None, "exit": trade})
    return pairs


def performance_stats(account: dict[str, Any]) -> dict[str, Any]:
    """Deterministic performance metrics over closed round-trip trades.

    All numbers come straight from the trade log; no model judgement is
    involved. Win/loss counts use the recorded PnL (fees excluded, they are
    reported separately through ``account['fees_paid']``).
    """
    pairs = pair_round_trips(account.get("trades", []))
    closed = [p for p in pairs if p["entry"] is not None]
    wins = [p for p in closed if float(p["exit"]["pnl"]) > 0]
    losses = [p for p in closed if float(p["exit"]["pnl"]) < 0]

    def _side_breakdown(side: str) -> dict[str, Any]:
        sel = [p for p in closed if p["entry"]["side"] == side]
        return {
            "count": len(sel),
            "wins": sum(1 for p in sel if float(p["exit"]["pnl"]) > 0),
            "pnl": round(sum(float(p["exit"]["pnl"]) for p in sel), 2),
        }

    def _coin_breakdown() -> dict[str, dict[str, Any]]:
        by_coin: dict[str, dict[str, Any]] = {}
        for p in closed:
            coin = p["entry"]["coin"]
            slot = by_coin.setdefault(coin, {"count": 0, "wins": 0, "pnl": 0.0})
            slot["count"] += 1
            if float(p["exit"]["pnl"]) > 0:
                slot["wins"] += 1
            slot["pnl"] = round(slot["pnl"] + float(p["exit"]["pnl"]), 2)
        return by_coin

    def _streaks() -> tuple[int, int]:
        current = longest = 0
        for p in closed:
            if float(p["exit"]["pnl"]) < 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return current, longest

    exits: dict[str, int] = {}
    for p in closed:
        label = str(p["exit"].get("action", "close"))
        exits[label] = exits.get(label, 0) + 1

    loss_streak_now, longest_loss_streak = _streaks()
    best = max(closed, key=lambda p: float(p["exit"]["pnl"]), default=None)
    worst = min(closed, key=lambda p: float(p["exit"]["pnl"]), default=None)

    def _avg_lev(trades_subset: list[dict[str, Any]]) -> float | None:
        levs = [p["exit"].get("leverage") or p["entry"].get("leverage")
                for p in trades_subset]
        levs = [lev for lev in levs if lev]
        return sum(levs) / len(levs) if levs else None

    return {
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100.0, 1) if closed else None,
        "total_pnl": round(sum(float(p["exit"]["pnl"]) for p in closed), 2),
        "avg_win": round(sum(float(p["exit"]["pnl"]) for p in wins) / len(wins), 2)
        if wins else None,
        "avg_loss": round(sum(float(p["exit"]["pnl"]) for p in losses) / len(losses), 2)
        if losses else None,
        "by_side": {"long": _side_breakdown("long"), "short": _side_breakdown("short")},
        "by_coin": _coin_breakdown(),
        "exits": exits,
        "loss_streak_now": loss_streak_now,
        "longest_loss_streak": longest_loss_streak,
        "best": best,
        "worst": worst,
        "avg_win_leverage": _avg_lev(wins),
        "avg_loss_leverage": _avg_lev(losses),
    }


def learning_lessons(stats: dict[str, Any]) -> list[str]:
    """Deterministic coaching rules derived from the performance stats.

    Small, conservative and fully rule-based: the model receives these as
    hard-won context, not as new risk limits (those stay in validate_action).
    """
    lessons: list[str] = []
    if not stats or not stats.get("closed"):
        return lessons
    if stats["loss_streak_now"] >= 3:
        lessons.append(
            f"Verliesreeks van {stats['loss_streak_now']} trades op een rij: wees "
            "extra selectief en overweeg een kleinere margin_fraction tot de reeks gebroken is."
        )
    for side in ("long", "short"):
        block = stats["by_side"][side]
        if block["count"] >= 3 and block["pnl"] < 0:
            lessons.append(
                f"{side}-trades hebben tot nu toe netto verlies opgeleverd "
                f"({block['count']} trades, {_fmt_signed(block['pnl'])} USD): vraag "
                f"een sterker bevestigd signaal voordat je weer {side} opent."
            )
    if stats["exits"].get("liquidated"):
        lessons.append(
            f"Er {is_zijn(stats['exits']['liquidated'])} liquidatie(s) geweest: houd "
            "meer afstand tot de benaderde liquidatieprijs (lagere leverage of ruimere stop)."
        )
    if stats["closed"] >= 5 and stats["win_rate"] is not None and stats["win_rate"] < 40:
        lessons.append(
            f"Win-rate is {stats['win_rate']:.0f}% over {stats['closed']} trades: verscherp "
            "je entry-criteria en wacht op duidelijke bevestiging."
        )
    if (
        stats["avg_win_leverage"] is not None
        and stats["avg_loss_leverage"] is not None
        and stats["avg_loss_leverage"] > stats["avg_win_leverage"]
        and stats["closed"] >= 3
    ):
        lessons.append(
            f"Verliezende trades liepen gemiddeld op {stats['avg_loss_leverage']:.1f}x leverage "
            f"tegen {stats['avg_win_leverage']:.1f}x bij winnaars: verlaag leverage bij twijfel."
        )
    return lessons[:MAX_LESSONS]


def is_zijn(count: int) -> str:
    """Dutch agreement helper: 'is 1' vs 'zijn 2+'."""
    return "is" if count == 1 else "zijn"


def _journal_line(pair: dict[str, Any]) -> str:
    """One-line journal entry for a closed round trip (newest last)."""
    entry, exit_trade = pair["entry"], pair["exit"]
    pnl = float(exit_trade["pnl"])
    margin = float(entry.get("margin") or 0.0)
    pnl_pct = (pnl / margin * 100.0) if margin > 0 else None
    label = _EXIT_LABELS.get(str(exit_trade.get("action", "close")), "close")
    extra = str(exit_trade.get("reason") or "")
    rationale = str(entry.get("reason") or "")[:120]
    parts = [
        f"{exit_trade.get('at', '?')} {entry['coin']} {entry['side']} "
        f"{exit_trade.get('leverage') or entry.get('leverage')}x: "
        f"in @ {entry.get('price')}, uit @ {exit_trade.get('price')} ({label}",
        f"pnl {_fmt_signed(pnl)} USD"
        + (f" ({pnl_pct:+.0f}% op marge)" if pnl_pct is not None else ""),
    ]
    if extra:
        parts.append(f"reden: {extra[:80]}")
    if rationale:
        parts.append(f"entry-idee: {rationale}")
    return " | ".join(parts)


def learning_block(account: dict[str, Any]) -> str:
    """Performance history + journal + lessons as a prompt block.

    Returns an empty string when no round trip has closed yet: there is
    nothing to learn from yet, so the prompt stays compact.
    """
    stats = performance_stats(account)
    if not stats["closed"]:
        return ""
    pairs = pair_round_trips(account.get("trades", []))
    closed = [p for p in pairs if p["entry"] is not None]
    sides = ", ".join(
        f"{side} {block['count']} ({block['wins']} winst, {_fmt_signed(block['pnl'])} USD)"
        for side, block in stats["by_side"].items()
        if block["count"]
    )
    coins = ", ".join(
        f"{coin} {block['count']}x ({_fmt_signed(block['pnl'])} USD)"
        for coin, block in sorted(stats["by_coin"].items())
    )
    exits = ", ".join(
        f"{count}x {_EXIT_LABELS.get(label, label)}"
        for label, count in sorted(stats["exits"].items())
    )
    lines = [
        f"- gesloten trades: {stats['closed']} | win-rate "
        f"{stats['win_rate']:.0f}% | netto resultaat {_fmt_signed(stats['total_pnl'])} USD (excl. fees)",
        f"- per richting: {sides or 'geen'}",
        f"- per coin: {coins or 'geen'}",
        f"- exits: {exits or 'geen'}",
        f"- verliesreeks nu: {stats['loss_streak_now']} | langste: {stats['longest_loss_streak']}",
        "JOURNAAL (nieuwste eerst):",
    ]
    for pair in list(reversed(closed))[:MAX_JOURNAL_TRADES]:
        lines.append(f"  - {_journal_line(pair)}")
    lessons = learning_lessons(stats)
    if lessons:
        lines.append("LESSEN (deterministisch afgeleid uit je trades):")
        for lesson in lessons:
            lines.append(f"  - {lesson}")
    return "\n".join(lines)


def learning_report_lines(account: dict[str, Any]) -> list[str]:
    """Compact performance summary + lessons for the Markdown cycle report."""
    stats = performance_stats(account)
    if not stats["closed"]:
        return []
    lines = [
        f"- **Gesloten trades:** {stats['closed']} | **Win-rate:** {stats['win_rate']:.0f}% "
        f"| **Netto resultaat:** {_fmt_signed(stats['total_pnl'])} USD (excl. fees, "
        f"fees totaal {_fmt_num(account.get('fees_paid', 0.0), 2)} USD)",
        f"- **Exits:** "
        + ", ".join(
            f"{count}x {_EXIT_LABELS.get(label, label)}"
            for label, count in sorted(stats["exits"].items())
        )
        + f" | verliesreeks nu: {stats['loss_streak_now']}, langste: {stats['longest_loss_streak']}",
    ]
    lessons = learning_lessons(stats)
    if lessons:
        lines.append("**Lessen voor volgende trades:**")
        lines.extend(f"- {lesson}" for lesson in lessons)
    return lines


# --------------------------------------------------------------------------
# Prompt building and report rendering
# --------------------------------------------------------------------------

def _fmt_price(value: float | int | None) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value:,.0f}".replace(",", ".")
    if value >= 1:
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{value:.6f}"


def _fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def _fmt_num(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_signed(value: float | int | None) -> str:
    if value is None:
        return "-"
    if value > 0:
        return "+" + _fmt_num(value, 2)
    return _fmt_num(value, 2)


def _fmt_news_date(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%d-%m %H:%M") + " UTC"
    except ValueError:
        return "-"


def compact_market_for_prompt(rows: list[dict[str, Any]], candles: dict[str, list[dict[str, Any]]]) -> str:
    """Compact market block for the model prompt."""
    lines = []
    for row in rows:
        lines.append(
            f"- {row['coin']}: prijs {row['price']}, 24u {_fmt_pct(row['day_change_pct'])}, "
            f"trend {row['trend']}, RSI {row['rsi_14'] if row['rsi_14'] is not None else '-'}, "
            f"SMA20 {row['sma_20']}, SMA50 {row['sma_50']}, vol {row['volatility_pct']}%, "
            f"funding {row['funding']}, OI {row['open_interest']}, "
            f"max leverage {row['max_leverage']}"
        )
        candle_list = candles.get(row["coin"], [])
        if candle_list:
            recent = candle_list[-PROMPT_CANDLES:]
            closes = ",".join(f"{c['c']:.4g}" for c in recent)
            lines.append(f"  laatste {len(recent)} closes: {closes}")
    return "\n".join(lines)


def account_state_for_prompt(
    account: dict[str, Any],
    prices: dict[str, float],
) -> str:
    """Current paper account state as a prompt block."""
    equity = account_equity(account, prices)
    day_pnl = day_pnl_pct(account, prices)
    lines = [
        f"- vrije balans: {_fmt_num(account['balance'], 2)} USD",
        f"- equity (incl. open posities): {_fmt_num(equity, 2)} USD",
        f"- dagresultaat: {_fmt_pct(day_pnl)}",
        f"- gerealiseerd totaal: {_fmt_signed(account['realized_pnl'])} USD",
        f"- betaalde fees totaal: {_fmt_num(account['fees_paid'], 2)} USD",
        f"- entries geblokkeerd: {'JA - ' + str(account['halt_reason']) if account.get('halted') else 'nee'}",
        f"- open posities: {len(account['positions'])}",
    ]
    for coin, pos in sorted(account["positions"].items()):
        price = prices.get(coin)
        unrealized = position_unrealized(pos, price) if price else 0.0
        lines.append(
            f"  - {coin} {pos['side']}: size {_fmt_num(pos['size'], 4)}, entry "
            f"{pos['entry_price']}, lev {pos['leverage']}x, marge {_fmt_num(pos['margin'], 2)}, "
            f"liq ~{pos['liq_price']}, uPnL {_fmt_signed(unrealized)}"
            + (f", SL {pos['stop_loss']}" if pos.get("stop_loss") else "")
            + (f", TP {pos['take_profit']}" if pos.get("take_profit") else "")
        )
    recent = account["trades"][-8:]
    if recent:
        lines.append("- laatste trades:")
        for trade in recent:
            lines.append(
                f"  - {trade['at']} {trade['action']} {trade['coin']} {trade['side']} "
                f"@ {trade['price']} (pnl {trade['pnl']})"
            )
    return "\n".join(lines)


def events_block(events: list[dict[str, Any]]) -> str:
    """Human-readable one-line summary per engine event."""
    lines = []
    for event in events:
        kind = event.get("type")
        if kind == "open":
            lines.append(
                f"- OPEN {event['coin']} {event['side']} @ {_fmt_price(event['price'])} "
                f"({event['leverage']}x, marge {_fmt_num(event['margin'], 2)} USD)"
            )
        elif kind == "close":
            lines.append(
                f"- CLOSE {event['coin']} @ {_fmt_price(event['price'])} "
                f"(pnl {_fmt_signed(event.get('pnl'))} USD, reden {event.get('reason')})"
            )
        elif kind == "rejected":
            lines.append(
                f"- GEWEIGERD {event.get('action')} {event.get('coin')}: {event.get('reason')}"
            )
        elif kind == "skipped":
            lines.append(f"- OVERGESLAGEN {event.get('coin')}: {event.get('reason')}")
        else:
            lines.append(f"- {event}")
    return "\n".join(lines)


def build_cycle_prompt(
    rows: list[dict[str, Any]],
    candles: dict[str, list[dict[str, Any]]],
    account: dict[str, Any],
    prices: dict[str, float],
    params: dict[str, Any],
    fear_greed: dict[str, Any] | None,
    news: list[dict[str, Any]],
    news_sentiment: dict[str, Any] | None,
    coin_mentions: list[dict[str, Any]],
) -> str:
    """Full Dutch cycle prompt for the Mistral model."""
    fg_block = "-"
    if fear_greed:
        fg_block = (
            f"{fear_greed['value']}/100 ({fear_greed['classification']}), "
            f"1d {_fmt_signed(fear_greed.get('change_1d'))}, "
            f"7d {_fmt_signed(fear_greed.get('change_7d'))}"
        )
    news_lines = (
        "\n".join(
            f"- [{item['sentiment']}] ({item['source']}, {_fmt_news_date(item['published'])}) {item['title']}"
            for item in news
        )
        or "geen koppen"
    )
    ns_block = "-"
    if news_sentiment:
        ns_block = (
            f"{news_sentiment['label']} (gem. score {news_sentiment['average_score']}, "
            f"{news_sentiment['positive']} positief / {news_sentiment['negative']} negatief / "
            f"{news_sentiment['neutral']} neutraal van {news_sentiment['total']})"
        )
    mentions_block = (
        ", ".join(f"{m['coin']} ({m['count']}x)" for m in coin_mentions) or "geen"
    )
    return f"""Je bent een crypto trading-agent die een papieren (paper) account beheert op Hyperliquid perpetuals. Beoordeel per cyclus de markt en beslis welke acties het account nu uitvoert. Je stuurt NIETS live: een deterministische engine valideert en simuleert elke actie tegen de risicoregels.

MARKTDATA (Hyperliquid, live):
{compact_market_for_prompt(rows, candles)}

MARKT-SENTIMENT (Fear & Greed Index):
{fg_block}

NIEUWSKOPPEN (laatste ~48 uur, met deterministische sentiment-score):
{news_lines}

NIEUWS-SAMENVATTING: {ns_block}
MUNTVERMELDINGEN: {mentions_block}

ACCOUNTSTATUS (paper):
{account_state_for_prompt(account, prices)}

PRESTATIEVERLEDEN (leer van je eigen trades):
{learning_block(account) or "nog geen gesloten trades; er valt nog niets uit eigen historie te leren"}

RISICOREGELS (door de engine afgedwongen, voorstellen buiten deze regels worden geweigerd):
1. Max leverage per positie: {params['max_leverage']}x (of lager als de coin minder toestaat).
2. Max marge per trade: {int(params['max_margin_fraction'] * 100)}% van de equity (margin_fraction).
3. Max {params['max_open_positions']} open posities tegelijk, 1 per coin.
4. Dagverlieslimiet {params['daily_loss_limit_pct']}%: daarna blokkeert de engine nieuwe entries tot de volgende UTC-dag.
5. Taker fee 0,045% bij open en close; liquidatieprijs is een benadering op ~{int(LIQ_BUFFER * 100)}% van de 1/leverage-marge.
6. Risicoprofiel: {params['risk_profile']}. Funding wordt in de simulatie niet meegenomen.
7. Sentiment en nieuws zijn contextsignalen: speculeer niet buiten de gegeven data.

BESLISSCHEMA (antwoord uitsluitend als JSON conform het schema):
- Beslis per coin in het universum: "open" (long of short), "close" (alleen als er een positie open staat), of "none".
- Wees selectief: alleen een "open" als setup, trend en sentiment samenwijzen; een lege actielijst is een prima uitkomst.
- Geef bij elke open een stop_loss en take_profit (long: SL onder entry, TP boven entry; short omgekeerd), en een korte rationale in het Nederlands.
- Sluit posities die hun thesis verliezen of bij bedreigend nieuws/technisch breuk.
- Neem het prestatieverleden mee: herhaal aanpak die winst opleverde, pas aan wat verlies opleverde, en volg de lessen tenzij de verse marktdata duidelijk anders zegt.
- Houd rekening met de accountstatus: margin_fraction is een fractie van de equity (bijv. 0.05 = 5%)."""


def render_cycle_report(
    now_iso: str,
    rows: list[dict[str, Any]],
    account: dict[str, Any],
    prices: dict[str, float],
    params: dict[str, Any],
    fear_greed: dict[str, Any] | None,
    news: list[dict[str, Any]],
    news_sentiment: dict[str, Any] | None,
    decision: dict[str, Any],
    events: list[dict[str, Any]],
    model: str,
) -> str:
    """Dutch Markdown report of one completed agent cycle."""
    equity = account_equity(account, prices)
    day_pnl = day_pnl_pct(account, prices)
    lines: list[str] = []
    lines.append("# Hyperliquid Trading Agent - cyclusrapport")
    lines.append("")
    lines.append(f"**Tijd:** {now_iso} | **Modus:** paper | **Model:** {model} | **Profiel:** {params['risk_profile']} | **Cyclus:** {account['cycles']}")
    lines.append("")
    lines.append("## Markt")
    lines.append("")
    lines.append("| Coin | Prijs | 24u | Trend | RSI | Funding | Max lev |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['coin']} | {_fmt_price(row['price'])} | {_fmt_pct(row['day_change_pct'])} "
            f"| {row['trend']} | {row['rsi_14'] if row['rsi_14'] is not None else '-'} "
            f"| {row['funding'] if row['funding'] is not None else '-'} "
            f"| {row['max_leverage'] if row['max_leverage'] is not None else '-'} |"
        )
    lines.append("")
    if fear_greed:
        lines.append("## Sentiment")
        lines.append("")
        lines.append(
            f"Fear & Greed Index: **{fear_greed['value']}/100** ({fear_greed['classification']}), "
            f"1d {_fmt_signed(fear_greed.get('change_1d'))}, 7d {_fmt_signed(fear_greed.get('change_7d'))}."
        )
        lines.append("")
    if news:
        lines.append("## Nieuws (laatste ~48 uur)")
        lines.append("")
        if news_sentiment:
            lines.append(
                f"Samenvatting: **{news_sentiment['label']}** (gem. score {news_sentiment['average_score']}, "
                f"{news_sentiment['positive']} positief / {news_sentiment['negative']} negatief / "
                f"{news_sentiment['neutral']} neutraal)."
            )
            lines.append("")
        lines.append("| Kop | Bron | Sentiment |")
        lines.append("|---|---|---|")
        for item in news:
            title = item["title"].replace("|", "\\|")
            lines.append(f"| {title} | {item['source']} | {item['sentiment']} |")
        lines.append("")
    lines.append("## Account (paper)")
    lines.append("")
    lines.append(
        f"- **Vrije balans:** {_fmt_num(account['balance'], 2)} USD | "
        f"**Equity:** {_fmt_num(equity, 2)} USD | "
        f"**Dagresultaat:** {_fmt_pct(day_pnl)}"
    )
    lines.append(
        f"- **Gerealiseerd totaal:** {_fmt_signed(account['realized_pnl'])} USD | "
        f"**Fees betaald:** {_fmt_num(account['fees_paid'], 2)} USD | "
        f"**Posities open:** {len(account['positions'])}"
    )
    if account.get("halted"):
        lines.append(f"- ⚠️ **Entries geblokkeerd:** {account.get('halt_reason')}")
    lines.append("")
    learning_lines = learning_report_lines(account)
    if learning_lines:
        lines.append("## Prestatieverleden (leren van trades)")
        lines.append("")
        lines.extend(learning_lines)
        lines.append("")
    if account["positions"]:
        lines.append("| Coin | Richting | Size | Entry | Lev | Marge | Liq ~ | uPnL | SL | TP |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for coin, pos in sorted(account["positions"].items()):
            price = prices.get(coin)
            unrealized = position_unrealized(pos, price) if price else 0.0
            lines.append(
                f"| {coin} | {pos['side']} | {_fmt_num(pos['size'], 4)} | {_fmt_price(pos['entry_price'])} "
                f"| {pos['leverage']}x | {_fmt_num(pos['margin'], 2)} | {_fmt_price(pos['liq_price'])} "
                f"| {_fmt_signed(unrealized)} | {pos.get('stop_loss') or '-'} | {pos.get('take_profit') or '-'} |"
            )
        lines.append("")
    lines.append("## Beslissingen")
    lines.append("")
    lines.append(f"**Marktbeeld:** {decision.get('market_view', '-')}")
    lines.append("")
    lines.append(decision.get("rationale", ""))
    lines.append("")
    taken = events_block(events) or "- geen acties deze cyclus"
    lines.append("**Uitgevoerd door de engine:**")
    lines.append("")
    lines.append(taken)
    lines.append("")
    if decision.get("key_risks"):
        lines.append("**Belangrijkste risico's:**")
        lines.append("")
        for risk in decision["key_risks"]:
            lines.append(f"- {risk}")
        lines.append("")
    lines.append("---")
    lines.append(
        "*Paper-simulatie op basis van publieke Hyperliquid-data. Geen financieel advies; "
        "heuristische liquidatie- en fee-benadering, geen funding-simulatie.*"
    )
    return "\n".join(lines)
