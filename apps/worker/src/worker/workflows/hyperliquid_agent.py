"""Hyperliquid trading agent: a durable, stateful paper-trading workflow.

Runs as a single long-lived execution on a managed deployment. Every cycle it
pulls live Hyperliquid market data (public info API, no key), Fear & Greed and
news headlines, shows a Mistral model the full market + paper-account state,
and lets a deterministic risk engine validate and simulate the model's
proposed actions against the account.

Everything here is PAPER ONLY: orders are simulated at live mid prices on an
in-account balance. Live trading is deliberately not implemented; asking for
``trading_mode="live"`` raises a clear error. No exchange credentials exist
anywhere in this code.

State lives on the workflow class (``self.account``), persisted by the
workflow engine across cycles and restarts. The paper account can be inspected
with the ``get_account``/``get_last_cycle`` queries and reset with the
``reset_account`` update.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from pydantic import BaseModel, Field

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    UserMessage,
    chat_parse_to_model,
)

from worker.workflows import hl_analysis

RISK_PROFILES = ("conservative", "balanced", "aggressive")
TRADING_MODES = ("paper",)  # live is deliberately not supported (yet)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


class AgentInput(BaseModel):
    """Configuration for the whole trading-agent run."""

    universe: list[str] = Field(default=["BTC", "ETH", "SOL"], min_length=1, max_length=5)
    interval: str = "4h"
    candles_limit: int = Field(default=180, ge=60, le=500)
    model: str = "mistral-medium-latest"
    risk_profile: str = "balanced"
    report_language: str = "nl"
    news_max_items: int = Field(default=10, ge=0, le=25)
    max_leverage: int = Field(default=5, ge=1, le=5)
    max_margin_fraction: float = Field(default=0.10, ge=0.01, le=0.25)
    max_open_positions: int = Field(default=3, ge=1, le=5)
    daily_loss_limit_pct: float = Field(default=5.0, ge=1.0, le=25.0)
    start_balance_usd: float = Field(default=1000.0, ge=100.0, le=1_000_000.0)
    cycle_seconds: int = Field(default=3600, ge=300, le=86_400)
    trading_mode: str = "paper"


class TradeAction(BaseModel):
    """One proposed action; validated and simulated by the risk engine."""

    action: str = Field(description="'open', 'close' of 'none'")
    coin: str = Field(description="coin uit het universum, bijv. BTC")
    direction: str = Field(default="long", description="'long' of 'short'")
    leverage: int = Field(default=1, ge=1, le=25)
    margin_fraction: float = Field(
        default=0.05, description="fractie van de equity als marge, bijv. 0.05 = 5%"
    )
    stop_loss: float | None = None
    take_profit: float | None = None
    rationale: str = ""


class CycleDecision(BaseModel):
    """Structured model output for one agent cycle."""

    market_view: str
    rationale: str
    key_risks: list[str] = Field(default_factory=list)
    actions: list[TradeAction] = Field(default_factory=list)


@workflows.activity()
async def now_utc() -> str:
    """Wall clock, only ever read in an activity (the workflow body stays deterministic)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


async def _hl_post(client, body: dict) -> dict | list | None:
    """One POST to the Hyperliquid info API with retry on transient errors."""
    last_error = "no attempt made"
    for attempt in range(4):
        resp = await client.post(HL_INFO_URL, json=body)
        if resp.status_code == 200:
            return resp.json()
        last_error = f"HTTP {resp.status_code}"
        if resp.status_code in _RETRYABLE_STATUS:
            await asyncio.sleep(1.5 * (2**attempt))
            continue
        break
    raise RuntimeError(f"Hyperliquid info failed ({body.get('type')}): {last_error}")


@workflows.activity()
async def fetch_hl_market(coins: list[str], interval: str, candles_limit: int) -> dict:
    """Fetch mids, meta+ctxs and candles for the universe from Hyperliquid.

    Coins whose candle request fails are still tradable on their mid price;
    a missing candle series only means no technical indicators for that coin.
    """
    import time

    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        mids_payload = await _hl_post(client, {"type": "allMids"})
        meta_payload = await _hl_post(client, {"type": "metaAndAssetCtxs"})

        interval_ms = _INTERVAL_MS.get(interval)
        if interval_ms is None:
            raise ValueError(f"interval moet een van {sorted(_INTERVAL_MS)} zijn")
        start_ms = int(time.time() * 1000) - candles_limit * interval_ms
        candles: dict[str, list] = {}
        for coin in coins:
            try:
                payload = await _hl_post(
                    client,
                    {
                        "type": "candleSnapshot",
                        "req": {
                            "coin": coin,
                            "interval": interval,
                            "startTime": start_ms,
                        },
                    },
                )
            except RuntimeError:
                continue  # no indicators for this coin, not fatal
            if isinstance(payload, list):
                candles[coin] = payload

    return {
        "mids": mids_payload,
        "meta": meta_payload,
        "candles": candles,
    }


@workflows.activity()
async def fetch_fear_greed() -> dict | None:
    """Fear & Greed Index (alternative.me); None when unreachable (non-fatal)."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                resp = await client.get(
                    "https://api.alternative.me/fng/", params={"limit": 8}
                )
            except httpx.HTTPError:
                break
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in _RETRYABLE_STATUS:
                await asyncio.sleep(1.5 * (2**attempt))
                continue
            break
    return None


@workflows.activity()
async def fetch_news_feeds() -> dict[str, str]:
    """Raw RSS XML per feed; unreachable feeds are simply absent."""
    import httpx

    feeds: dict[str, str] = {}
    async with httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": "hyperliquid-trading-agent/1.0 (+rss)"}
    ) as client:
        for source, url in hl_analysis.NEWS_FEEDS.items():
            try:
                resp = await client.get(url)
            except httpx.HTTPError:
                continue
            if resp.status_code == 200 and resp.text:
                feeds[source] = resp.text
    return feeds


@workflows.workflow.define(
    name="hyperliquid-trading-agent",
    workflow_display_name="Hyperliquid Trading Agent",
    # The platform default execution_timeout is 1 hour; this agent is a
    # long-running loop, so it opts into a full year. Note the execution
    # history cap (~51,200 events) is only reached after ~1,5 year at the
    # default 4h cadence; when that horizon gets closer, add the
    # Continue-As-New pattern to reset the history while carrying the
    # account state forward.
    execution_timeout=timedelta(days=365),
    workflow_description=(
        "Duurzame paper-trading-agent voor Hyperliquid perpetuals met leverage "
        "(max 5x, max 10% marge per trade, max 3 posities, dagverlieslimiet). "
        "Haalt elke cyclus live Hyperliquid-data, Fear & Greed en cryptonieuws, "
        "laat een Mistral-model acties voorstellen en simuleert de orders op "
        "een paper-account via een deterministische risico-engine. Alleen "
        "paper; live traden is bewust niet geimplementeerd. Geen financieel advies."
    ),
)
class HyperliquidTradingAgent:
    def __init__(self):
        self.account: dict | None = None
        self.last_cycle: dict | None = None
        self.started_at: str | None = None
        self.last_error: str | None = None

    @workflows.workflow.query(name="get_account")
    def get_account(self) -> dict | None:
        """Current paper account state: balance, positions, trades, halt status."""
        return self.account

    @workflows.workflow.query(name="get_last_cycle")
    def get_last_cycle(self) -> dict | None:
        """Summary + Markdown report of the most recent completed cycle."""
        return self.last_cycle

    @workflows.workflow.update(name="reset_account")
    async def reset_account(self, start_balance_usd: float = 1000.0) -> dict:
        """Reset the paper account to a fresh balance (positions are dropped)."""
        now = await now_utc()
        self.account = hl_analysis.new_account(float(start_balance_usd), now)
        self.last_cycle = None
        return {"ok": True, "account": self.account}

    @workflows.workflow.update(name="set_entries_halted")
    def set_entries_halted(self, halted: bool, reason: str = "handmatig") -> dict:
        """Manually block (or re-allow) new entries; closes always remain possible."""
        self.account["halted"] = bool(halted)
        self.account["halt_reason"] = reason if halted else None
        return {"ok": True, "halted": self.account["halted"]}

    async def _run_cycle(self, inp: AgentInput) -> None:
        """One full trading cycle: data -> forced exits -> model -> engine -> report."""
        now = await now_utc()
        params = {
            "universe": [c.upper() for c in inp.universe],
            "max_leverage": inp.max_leverage,
            "max_margin_fraction": inp.max_margin_fraction,
            "max_open_positions": inp.max_open_positions,
            "daily_loss_limit_pct": inp.daily_loss_limit_pct,
            "risk_profile": inp.risk_profile,
        }
        account = self.account
        account["cycles"] += 1
        account["last_cycle_at"] = now

        market = await fetch_hl_market(params["universe"], inp.interval, inp.candles_limit)
        mids = hl_analysis.parse_all_mids(market["mids"])
        meta_rows = hl_analysis.parse_meta_and_ctxs(market["meta"], params["universe"])
        candles = {
            coin: hl_analysis.parse_candles(rows)
            for coin, rows in market["candles"].items()
        }
        prices = {coin: mids[coin] for coin in params["universe"] if coin in mids}
        if not prices:
            raise RuntimeError("Geen bruikbare Hyperliquid-mids voor het universum.")

        # 1. New trading day? Reset the day baseline and lift a stale halt.
        hl_analysis.roll_day_if_needed(account, now, prices)

        # 2. Forced exits on open positions (liq, stop, target) at current prices.
        forced_events = hl_analysis.process_forced_exits(account, prices, now)

        # 3. Market context: sentiment and news.
        fear_greed_raw = await fetch_fear_greed()
        fear_greed = hl_analysis.parse_fear_greed(fear_greed_raw) if fear_greed_raw else None
        feeds_xml = await fetch_news_feeds()
        news = hl_analysis.merge_news(
            {
                source: hl_analysis.parse_rss_feed(xml_text, source)
                for source, xml_text in feeds_xml.items()
            },
            max_items=inp.news_max_items,
        )
        news_sentiment = hl_analysis.summarize_news_sentiment(news)
        coin_mentions = hl_analysis.coin_mention_counts(news, params["universe"])

        rows = hl_analysis.build_market_rows(prices, meta_rows, candles)

        # 4. Model decides; the engine validates and simulates.
        prompt = hl_analysis.build_cycle_prompt(
            rows, candles, account, prices, params, fear_greed, news, news_sentiment, coin_mentions
        )
        request = ChatCompletionRequest(model=inp.model, messages=[UserMessage(content=prompt)])
        decision = await chat_parse_to_model(CycleDecision, request)

        events = forced_events + hl_analysis.apply_actions(
            account,
            [a.model_dump() for a in decision.actions],
            prices,
            now,
            params,
            meta_rows,
        )
        hl_analysis.update_circuit_breaker(account, prices, inp.daily_loss_limit_pct)

        report_markdown = hl_analysis.render_cycle_report(
            now_iso=now,
            rows=rows,
            account=account,
            prices=prices,
            params=params,
            fear_greed=fear_greed,
            news=news,
            news_sentiment=news_sentiment,
            decision=decision.model_dump(),
            events=events,
            model=inp.model,
        )
        equity = hl_analysis.account_equity(account, prices)
        account["equity_high"] = max(account.get("equity_high", equity), equity)
        self.last_cycle = {
            "cycle": account["cycles"],
            "at": now,
            "equity": round(equity, 2),
            "day_pnl_pct": round(hl_analysis.day_pnl_pct(account, prices), 2),
            "open_positions": len(account["positions"]),
            "events": events,
            "market_view": decision.market_view,
            "report_markdown": report_markdown,
        }
        self.last_error = None

    @workflows.workflow.entrypoint
    async def run(self, input: AgentInput) -> None:
        if input.trading_mode not in TRADING_MODES:
            raise ValueError(
                "Deze agent draait alleen in paper-modus. Live traden is "
                "bewust niet geimplementeerd; niets in deze code kan orders "
                "plaatsen of bevat exchange-credentials."
            )
        if input.risk_profile not in RISK_PROFILES:
            raise ValueError(
                f"risk_profile moet een van {RISK_PROFILES} zijn, got: {input.risk_profile}"
            )
        if input.interval not in hl_analysis.INTERVALS:
            raise ValueError(
                f"interval moet een van {hl_analysis.INTERVALS} zijn, got: {input.interval}"
            )

        now = await now_utc()
        self.started_at = now
        self.account = hl_analysis.new_account(input.start_balance_usd, now)
        self.last_cycle = None
        self.last_error = None

        # The agent never dies: a failing cycle (API down, model error) is
        # recorded and the loop simply tries again after the next sleep.
        while True:
            try:
                await self._run_cycle(input)
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                self.last_error = f"{type(exc).__name__}: {exc}"
            await workflows.workflow.sleep(input.cycle_seconds)
