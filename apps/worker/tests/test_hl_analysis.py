"""Unit tests for the Hyperliquid trading agent pure core (stdlib-only).

Run from apps/worker/tests:  python3 -m unittest test_hl_analysis.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from worker.workflows import hl_analysis  # noqa: E402


def make_candles(closes):
    """Candle list from a sequence of closes: o=l=c, h=c*1.01, v=100."""
    return [
        {"t": 1_700_000_000_000 + i * 14_400_000, "o": c, "h": c * 1.01,
         "l": c * 0.99, "c": c, "v": 100.0}
        for i, c in enumerate(closes)
    ]


PARAMS = {
    "universe": ["BTC", "ETH", "SOL"],
    "max_leverage": 5,
    "max_margin_fraction": 0.10,
    "max_open_positions": 3,
    "daily_loss_limit_pct": 5.0,
    "risk_profile": "balanced",
}

META = {
    "BTC": {"coin": "BTC", "max_leverage": 40, "day_change_pct": 1.5,
            "funding": 0.000006, "open_interest": 38000.0},
    "ETH": {"coin": "ETH", "max_leverage": 25},
    "SOL": {"coin": "SOL", "max_leverage": 20},
}

PRICES = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}


def fresh_account():
    return hl_analysis.new_account(1000.0, "2026-11-14T10:00:00+00:00")


class IndicatorTests(unittest.TestCase):
    def test_sma(self):
        self.assertAlmostEqual(hl_analysis.sma([1, 2, 3, 4], 2), 3.5)
        self.assertIsNone(hl_analysis.sma([1], 2))
        self.assertIsNone(hl_analysis.sma([1, 2], 0))

    def test_rsi_all_gains(self):
        closes = [float(i) for i in range(1, 25)]
        self.assertAlmostEqual(hl_analysis.rsi(closes), 100.0)

    def test_rsi_needs_data(self):
        self.assertIsNone(hl_analysis.rsi([1.0] * 5))

    def test_volatility_zero_when_flat(self):
        self.assertAlmostEqual(hl_analysis.volatility_pct([5.0] * 40), 0.0)

    def test_volatility_needs_lookback(self):
        self.assertIsNone(hl_analysis.volatility_pct([1.0, 2.0]))

    def test_classify_trend(self):
        self.assertEqual(hl_analysis.classify_trend(110, 100, 90), "stijgend")
        self.assertEqual(hl_analysis.classify_trend(80, 100, 110), "dalend")
        self.assertEqual(hl_analysis.classify_trend(95, 100, 90), "zijwaarts")
        self.assertEqual(hl_analysis.classify_trend(95, None, 90), "onbekend")


class HyperliquidParsingTests(unittest.TestCase):
    def test_parse_all_mids_filters_spot_and_junk(self):
        payload = {"BTC": "100.0", "ETH": "50", "@1": "10", "#123": "1", "FOO": "-3", "B2B": "x"}
        self.assertEqual(hl_analysis.parse_all_mids(payload), {"BTC": 100.0, "ETH": 50.0})

    def test_parse_all_mids_empty(self):
        self.assertEqual(hl_analysis.parse_all_mids(None), {})

    def test_parse_meta_and_ctxs(self):
        meta = {"universe": [
            {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
            {"name": "ETH", "maxLeverage": 25},
        ]}
        ctxs = [
            {"markPx": "101.0", "prevDayPx": "100.0", "funding": "0.00001",
             "openInterest": "12.5", "midPx": "101.0", "dayNtlVlm": "1e9"},
            {"markPx": "49", "prevDayPx": "50", "funding": "0", "openInterest": "1"},
        ]
        rows = hl_analysis.parse_meta_and_ctxs([meta, ctxs], ["BTC", "ETH"])
        self.assertIn("BTC", rows)
        self.assertEqual(rows["BTC"]["max_leverage"], 40)
        self.assertAlmostEqual(rows["BTC"]["day_change_pct"], 1.0)
        self.assertAlmostEqual(rows["ETH"]["day_change_pct"], -2.0)
        self.assertNotIn("SOL", rows)

    def test_parse_meta_and_ctxs_bad_shape(self):
        self.assertEqual(hl_analysis.parse_meta_and_ctxs(None, ["BTC"]), {})
        self.assertEqual(hl_analysis.parse_meta_and_ctxs(["x"], ["BTC"]), {})

    def test_parse_candles_normalizes_and_sorts(self):
        rows = [
            {"t": "200", "o": "2", "h": "3", "l": "1", "c": "2.5", "v": "10"},
            {"t": "100", "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "5"},
            {"t": "junk", "o": "1", "h": "1", "l": "1", "c": "1", "v": "1"},
        ]
        parsed = hl_analysis.parse_candles(rows)
        self.assertEqual([c["t"] for c in parsed], [100, 200])
        self.assertIsInstance(parsed[0]["c"], float)

    def test_build_market_rows_includes_indicators(self):
        candles = {"BTC": make_candles([float(i) for i in range(10, 110)])}
        rows = hl_analysis.build_market_rows({"BTC": 109.0}, META, candles)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["coin"], "BTC")
        self.assertIsNotNone(row["rsi_14"])
        self.assertIsNotNone(row["sma_20"])
        self.assertEqual(row["trend"], "stijgend")
        self.assertEqual(row["candles"], 100)


class FearGreedTests(unittest.TestCase):
    def test_parse_fear_greed(self):
        payload = {"data": [
            {"value": "13", "value_classification": "Extreme Fear"},
            {"value": "20", "value_classification": "Fear"},
        ]}
        fg = hl_analysis.parse_fear_greed(payload)
        self.assertEqual(fg["value"], 13)
        self.assertEqual(fg["change_1d"], -7)
        self.assertIsNone(fg["change_7d"])

    def test_parse_fear_greed_unusable(self):
        self.assertIsNone(hl_analysis.parse_fear_greed(None))
        self.assertIsNone(hl_analysis.parse_fear_greed({"data": []}))


RSS_SAMPLE = """<?xml version="1.0"?>
<rss><channel>
<item><title>BTC surges to record highs</title><link>https://x.test/1</link>
<pubDate>Sat, 14 Nov 2026 10:00:00 GMT</pubDate></item>
<item><title>Exchange hack causes crash fears</title><link>https://x.test/2</link>
<pubDate>Sat, 14 Nov 2026 09:00:00 GMT</pubDate></item>
<item><title>Quiet day in the markets</title><link>https://x.test/3</link>
<pubDate>Fri, 13 Nov 2026 09:00:00 GMT</pubDate></item>
</channel></rss>"""


class NewsTests(unittest.TestCase):
    def test_parse_rss_feed(self):
        items = hl_analysis.parse_rss_feed(RSS_SAMPLE, "test")
        self.assertEqual(len(items), 3)
        self.assertTrue(items[0]["published"].startswith("2026-11-14"))

    def test_parse_rss_feed_malformed(self):
        self.assertEqual(hl_analysis.parse_rss_feed("not xml <<<", "test"), [])

    def test_merge_news_scores_and_dedupes(self):
        items = hl_analysis.parse_rss_feed(RSS_SAMPLE, "test")
        merged = hl_analysis.merge_news({"a": items, "b": items}, max_items=10)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["sentiment"], "positief")
        self.assertEqual(merged[1]["sentiment"], "negatief")

    def test_score_headline_sentiment_clamps(self):
        self.assertEqual(hl_analysis.score_headline_sentiment("surge rally bull gains"), 2)
        self.assertEqual(hl_analysis.score_headline_sentiment("crash plunge dump loss"), -2)

    def test_summarize_news_sentiment(self):
        summary = hl_analysis.summarize_news_sentiment([
            {"sentiment_score": 1}, {"sentiment_score": -1}, {"sentiment_score": 0},
        ])
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["label"], "neutraal")

    def test_coin_mention_counts_word_boundary(self):
        news = [{"title": "BTC rises, SOL crashes"}, {"title": "solely irrelevant"}]
        counts = hl_analysis.coin_mention_counts(news, ["BTC", "SOL"])
        self.assertEqual(counts, [{"coin": "BTC", "count": 1}, {"coin": "SOL", "count": 1}])


class PaperEngineTests(unittest.TestCase):
    def test_new_account(self):
        account = fresh_account()
        self.assertEqual(account["balance"], 1000.0)
        self.assertEqual(account["positions"], {})
        self.assertEqual(account["day_key"], "2026-11-14")
        self.assertFalse(account["halted"])

    def test_liq_price_long_and_short(self):
        self.assertAlmostEqual(hl_analysis.liq_price_for(100.0, "long", 5), 82.0)
        self.assertAlmostEqual(hl_analysis.liq_price_for(100.0, "short", 5), 118.0)

    def test_position_unrealized(self):
        pos = {"side": "long", "entry_price": 100.0, "size": 2.0}
        self.assertAlmostEqual(hl_analysis.position_unrealized(pos, 110.0), 20.0)
        pos["side"] = "short"
        self.assertAlmostEqual(hl_analysis.position_unrealized(pos, 110.0), -20.0)

    def test_account_equity(self):
        account = fresh_account()
        account["positions"]["BTC"] = {
            "side": "long", "entry_price": 100.0, "size": 2.0, "margin": 50.0,
        }
        # balance was reduced by the margin in a real open; simulate that:
        account["balance"] = 950.0
        self.assertAlmostEqual(hl_analysis.account_equity(account, {"BTC": 105.0}), 1010.0)


class ValidateActionTests(unittest.TestCase):
    def test_valid_open(self):
        account = fresh_account()
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 5, "margin_fraction": 0.05}
        ok, reason = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertTrue(ok, reason)

    def test_rejects_unknown_coin(self):
        account = fresh_account()
        action = {"action": "open", "coin": "DOGE", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.05}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_bad_direction(self):
        account = fresh_account()
        action = {"action": "open", "coin": "BTC", "direction": "sideways",
                  "leverage": 2, "margin_fraction": 0.05}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_leverage_over_cap(self):
        account = fresh_account()
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 6, "margin_fraction": 0.05}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_leverage_over_coin_cap(self):
        account = fresh_account()
        low_meta = {"BTC": {"coin": "BTC", "max_leverage": 3}}
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 5, "margin_fraction": 0.05}
        ok, reason = hl_analysis.validate_action(
            action, account, PRICES, PARAMS | {"max_leverage": 10}, low_meta
        )
        self.assertFalse(ok)
        self.assertIn("3", reason)  # capped by the coin's max leverage

    def test_rejects_margin_fraction_over_limit(self):
        account = fresh_account()
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.15}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_zero_margin_fraction(self):
        account = fresh_account()
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.0}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_duplicate_position(self):
        account = fresh_account()
        account["positions"]["BTC"] = {"side": "long", "entry_price": 100.0,
                                       "size": 1.0, "margin": 50.0, "leverage": 5}
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.05}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_when_halted(self):
        account = fresh_account()
        account["halted"] = True
        account["halt_reason"] = "test"
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.05}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_close_without_position(self):
        account = fresh_account()
        ok, _ = hl_analysis.validate_action(
            {"action": "close", "coin": "BTC"}, account, PRICES, PARAMS, META
        )
        self.assertFalse(ok)

    def test_close_with_position_ok(self):
        account = fresh_account()
        account["positions"]["BTC"] = {"side": "long", "entry_price": 100.0,
                                       "size": 1.0, "margin": 50.0, "leverage": 5}
        ok, _ = hl_analysis.validate_action(
            {"action": "close", "coin": "BTC"}, account, PRICES, PARAMS, META
        )
        self.assertTrue(ok)

    def test_rejects_stop_on_wrong_side(self):
        account = fresh_account()
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.05, "stop_loss": 105.0}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_take_profit_on_wrong_side(self):
        account = fresh_account()
        action = {"action": "open", "coin": "BTC", "direction": "short",
                  "leverage": 2, "margin_fraction": 0.05, "take_profit": 105.0}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_when_insufficient_balance(self):
        account = fresh_account()
        account["balance"] = 1.0
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.05}
        ok, _ = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)

    def test_rejects_open_when_circuit_breaker_tripped(self):
        account = fresh_account()
        # simulate a 6% loss today: day_start 1000, equity now 940
        account["day_start_equity"] = 1000.0
        account["balance"] = 940.0
        action = {"action": "open", "coin": "BTC", "direction": "long",
                  "leverage": 2, "margin_fraction": 0.05}
        ok, reason = hl_analysis.validate_action(action, account, PRICES, PARAMS, META)
        self.assertFalse(ok)
        self.assertIn("dagverlieslimiet", reason)


class ApplyActionsTests(unittest.TestCase):
    def test_open_and_close_roundtrip(self):
        account = fresh_account()
        events = hl_analysis.apply_actions(
            account,
            [{"action": "open", "coin": "BTC", "direction": "long",
              "leverage": 5, "margin_fraction": 0.10,
              "stop_loss": 90.0, "take_profit": 120.0, "rationale": "test"}],
            PRICES, "2026-11-14T11:00:00+00:00", PARAMS, META,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "open")
        pos = account["positions"]["BTC"]
        # margin 10% of 1000 equity = 100, notional 500, size 5 at price 100
        self.assertAlmostEqual(pos["margin"], 100.0)
        self.assertAlmostEqual(pos["size"], 5.0)
        self.assertAlmostEqual(pos["liq_price"], 82.0)
        self.assertAlmostEqual(pos["entry_price"], 100.0)
        # balance: 1000 - margin 100 - fee (500 * 0.00045 = 0.225)
        self.assertAlmostEqual(account["balance"], 899.775)
        self.assertAlmostEqual(account["fees_paid"], 0.225)

        # price moves to 110: close realizes +50 minus fee
        prices = {"BTC": 110.0}
        events = hl_analysis.apply_actions(
            account, [{"action": "close", "coin": "BTC"}],
            prices, "2026-11-14T12:00:00+00:00", PARAMS, META,
        )
        self.assertEqual(events[0]["type"], "close")
        self.assertAlmostEqual(events[0]["pnl"], 50.0)
        # balance: 899.775 + 100 + 50 - fee (550*0.00045 = 0.2475)
        self.assertAlmostEqual(account["balance"], 1049.5275)
        self.assertAlmostEqual(account["realized_pnl"], 50.0)
        self.assertEqual(account["positions"], {})
        self.assertEqual(len(account["trades"]), 2)

    def test_rejected_action_becomes_event(self):
        account = fresh_account()
        events = hl_analysis.apply_actions(
            account,
            [{"action": "open", "coin": "BTC", "direction": "long",
              "leverage": 50, "margin_fraction": 0.05}],
            PRICES, "2026-11-14T11:00:00+00:00", PARAMS, META,
        )
        self.assertEqual(events[0]["type"], "rejected")
        self.assertEqual(account["positions"], {})
        self.assertEqual(account["trades"], [])

    def test_none_actions_are_silent(self):
        account = fresh_account()
        events = hl_analysis.apply_actions(
            account,
            [{"action": "none", "coin": "BTC"}],
            PRICES, "2026-11-14T11:00:00+00:00", PARAMS, META,
        )
        self.assertEqual(events, [])

    def test_closes_processed_before_opens(self):
        account = fresh_account()
        account["positions"]["BTC"] = {
            "side": "long", "entry_price": 100.0, "size": 5.0,
            "margin": 100.0, "leverage": 5, "liq_price": 82.0,
            "stop_loss": None, "take_profit": None, "opened_at": "x",
        }
        account["balance"] = 900.0
        actions = [
            {"action": "open", "coin": "ETH", "direction": "long",
             "leverage": 2, "margin_fraction": 0.10},
            {"action": "close", "coin": "BTC"},
        ]
        events = hl_analysis.apply_actions(
            account, actions, PRICES, "2026-11-14T11:00:00+00:00", PARAMS, META,
        )
        types = [e["type"] for e in events]
        self.assertEqual(types, ["close", "open"])
        self.assertIn("ETH", account["positions"])
        self.assertNotIn("BTC", account["positions"])


class ForcedExitTests(unittest.TestCase):
    def _account_with_position(self, side="long", entry=100.0, lev=5, stop=None, target=None):
        account = fresh_account()
        account["positions"]["BTC"] = {
            "side": side, "entry_price": entry, "size": 5.0,
            "margin": 100.0, "leverage": lev,
            "liq_price": hl_analysis.liq_price_for(entry, side, lev),
            "stop_loss": stop, "take_profit": target, "opened_at": "x",
        }
        account["balance"] = 900.0
        return account

    def test_liquidation_loses_margin(self):
        account = self._account_with_position()
        events = hl_analysis.process_forced_exits(account, {"BTC": 80.0}, "t")
        self.assertEqual(events[0]["type"], "liquidated")
        self.assertAlmostEqual(events[0]["pnl"], -100.0)
        # balance unchanged: the locked margin is simply gone
        self.assertAlmostEqual(account["balance"], 900.0)
        self.assertAlmostEqual(account["realized_pnl"], -100.0)
        self.assertNotIn("BTC", account["positions"])
        self.assertEqual(account["trades"][-1]["action"], "liquidated")

    def test_stop_loss_close(self):
        account = self._account_with_position(stop=90.0)
        events = hl_analysis.process_forced_exits(account, {"BTC": 89.0}, "t")
        self.assertEqual(events[0]["type"], "close")
        self.assertEqual(events[0]["reason"], "stop")
        self.assertAlmostEqual(events[0]["price"], 90.0)
        self.assertAlmostEqual(events[0]["pnl"], -50.0)

    def test_take_profit_close(self):
        account = self._account_with_position(target=120.0)
        events = hl_analysis.process_forced_exits(account, {"BTC": 121.0}, "t")
        self.assertEqual(events[0]["type"], "close")
        self.assertEqual(events[0]["reason"], "target")
        self.assertAlmostEqual(events[0]["pnl"], 100.0)

    def test_short_take_profit(self):
        account = self._account_with_position(side="short", target=90.0)
        events = hl_analysis.process_forced_exits(account, {"BTC": 89.0}, "t")
        self.assertEqual(events[0]["reason"], "target")
        self.assertAlmostEqual(events[0]["pnl"], 50.0)

    def test_no_exit_inside_range(self):
        account = self._account_with_position(stop=90.0, target=120.0)
        events = hl_analysis.process_forced_exits(account, {"BTC": 100.0}, "t")
        self.assertEqual(events, [])
        self.assertIn("BTC", account["positions"])


class DayAndBreakerTests(unittest.TestCase):
    def test_roll_day_resets_baseline_and_unhalts(self):
        account = fresh_account()
        account["halted"] = True
        account["halt_reason"] = "dagverlieslimiet -5% bereikt"
        account["balance"] = 900.0
        rolled = hl_analysis.roll_day_if_needed(
            account, "2026-11-15T00:30:00+00:00", {}
        )
        self.assertTrue(rolled)
        self.assertEqual(account["day_key"], "2026-11-15")
        self.assertAlmostEqual(account["day_start_equity"], 900.0)
        self.assertFalse(account["halted"])

    def test_roll_day_same_day_noop(self):
        account = fresh_account()
        self.assertFalse(
            hl_analysis.roll_day_if_needed(account, "2026-11-14T23:00:00+00:00", {})
        )

    def test_circuit_breaker_trips(self):
        account = fresh_account()
        account["day_start_equity"] = 1000.0
        account["balance"] = 949.0  # -5.1% on the day, no positions
        tripped = hl_analysis.update_circuit_breaker(account, {}, 5.0)
        self.assertTrue(tripped)
        self.assertTrue(account["halted"])
        self.assertIn("dagverlieslimiet", account["halt_reason"])
        # idempotent
        self.assertTrue(hl_analysis.update_circuit_breaker(account, {}, 5.0))

    def test_circuit_breaker_stays_calm_within_limit(self):
        account = fresh_account()
        account["day_start_equity"] = 1000.0
        account["balance"] = 960.0
        self.assertFalse(hl_analysis.update_circuit_breaker(account, {}, 5.0))
        self.assertFalse(account["halted"])


class PromptAndReportTests(unittest.TestCase):
    def _cycle_context(self):
        account = fresh_account()
        candles = {"BTC": make_candles([float(i) for i in range(80, 120)])}
        rows = hl_analysis.build_market_rows(PRICES, META, candles)
        fg = {"value": 13, "classification": "Extreme Fear",
              "change_1d": -7, "change_7d": -12}
        news = [{"title": "BTC surges", "source": "test", "published": "2026-11-14T10:00:00+00:00",
                 "url": "u", "sentiment_score": 1, "sentiment": "positief"}]
        ns = {"average_score": 1.0, "label": "positief", "positive": 1,
              "negative": 0, "neutral": 0, "total": 1}
        mentions = [{"coin": "BTC", "count": 1}]
        return account, candles, rows, fg, news, ns, mentions

    def test_build_cycle_prompt_blocks(self):
        account, candles, rows, fg, news, ns, mentions = self._cycle_context()
        prompt = hl_analysis.build_cycle_prompt(
            rows, candles, account, PRICES, PARAMS, fg, news, ns, mentions
        )
        for block in ("MARKTDATA", "MARKT-SENTIMENT", "NIEUWSKOPPEN",
                      "ACCOUNTSTATUS", "RISICOREGELS", "BESLISSCHEMA"):
            self.assertIn(block, prompt)
        self.assertIn("13/100", prompt)
        self.assertIn("BTC surges", prompt)
        self.assertIn("vrije balans", prompt)

    def test_account_state_for_prompt_shows_position(self):
        account = fresh_account()
        account["positions"]["BTC"] = {
            "side": "long", "entry_price": 100.0, "size": 5.0,
            "margin": 100.0, "leverage": 5, "liq_price": 82.0,
            "stop_loss": 90.0, "take_profit": 120.0, "opened_at": "x",
        }
        account["balance"] = 900.0
        block = hl_analysis.account_state_for_prompt(account, PRICES)
        self.assertIn("BTC long", block)
        self.assertIn("open posities: 1", block)

    def test_render_cycle_report(self):
        account, candles, rows, fg, news, ns, mentions = self._cycle_context()
        decision = {
            "market_view": "voorzichtig bullish",
            "rationale": "RSI laag, sentiment herstellend.",
            "key_risks": ["liquidatierisico bij 5x"],
            "actions": [],
        }
        report = hl_analysis.render_cycle_report(
            now_iso="2026-11-14T12:00:00+00:00", rows=rows, account=account,
            prices=PRICES, params=PARAMS, fear_greed=fg, news=news,
            news_sentiment=ns, decision=decision,
            events=[{"type": "open", "coin": "BTC", "side": "long", "price": 100.0,
                     "size": 5.0, "leverage": 5, "margin": 100.0}],
            model="mistral-medium-latest",
        )
        self.assertIn("# Hyperliquid Trading Agent - cyclusrapport", report)
        self.assertIn("## Markt", report)
        self.assertIn("## Sentiment", report)
        self.assertIn("## Nieuws", report)
        self.assertIn("## Account (paper)", report)
        self.assertIn("## Beslissingen", report)
        self.assertIn("OPEN BTC long", report)
        self.assertIn("Geen financieel advies", report)

    def test_events_block_rejected(self):
        block = hl_analysis.events_block([
            {"type": "rejected", "action": "open", "coin": "BTC", "reason": "te veel leverage"},
        ])
        self.assertIn("GEWEIGERD", block)


class LearningTests(unittest.TestCase):
    """In-context learning: stats, journal, lessons, prompt and report blocks."""

    def _trade(self, action, coin, side, pnl, at, lev=5, margin=100.0, reason=""):
        return {"coin": coin, "side": side, "action": action, "price": 100.0,
                "size": 1.0, "notional": 100.0, "leverage": lev, "margin": margin,
                "pnl": pnl, "fee": 0.0, "at": at, "reason": reason}

    def _account_with_history(self):
        account = fresh_account()
        account["trades"] = [
            self._trade("open", "BTC", "long", None, "t1", reason="oversold bounce"),
            self._trade("close", "BTC", "long", 50.0, "t2"),
            self._trade("open", "BTC", "long", None, "t3", lev=3, reason="trendvolg"),
            self._trade("target", "BTC", "long", 10.0, "t4", lev=3),
            self._trade("open", "ETH", "short", None, "t5", reason="weerstand"),
            self._trade("stop", "ETH", "short", -20.0, "t6"),
            self._trade("open", "ETH", "short", None, "t7"),
            self._trade("close", "ETH", "short", -30.0, "t8"),
            self._trade("open", "ETH", "short", None, "t9"),
            self._trade("liquidated", "ETH", "short", -5.0, "t10"),
        ]
        return account

    def test_pair_round_trips(self):
        trades = [
            self._trade("open", "BTC", "long", None, "t1"),
            self._trade("open", "ETH", "short", None, "t2"),
            self._trade("close", "BTC", "long", 50.0, "t3"),
            self._trade("stop", "ETH", "short", -20.0, "t4"),
            self._trade("liquidated", "SOL", "long", -100.0, "t5"),
        ]
        pairs = hl_analysis.pair_round_trips(trades)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["entry"]["coin"], "BTC")
        self.assertEqual(pairs[0]["exit"]["pnl"], 50.0)
        self.assertEqual(pairs[1]["entry"]["coin"], "ETH")
        self.assertIsNone(pairs[2]["entry"])
        self.assertEqual(pairs[2]["exit"]["pnl"], -100.0)

    def test_stats_empty_account(self):
        stats = hl_analysis.performance_stats(fresh_account())
        self.assertEqual(stats["closed"], 0)
        self.assertIsNone(stats["win_rate"])
        self.assertEqual(hl_analysis.learning_lessons(stats), [])
        self.assertEqual(hl_analysis.learning_block(fresh_account()), "")
        self.assertEqual(hl_analysis.learning_report_lines(fresh_account()), [])

    def test_stats_mixed_history(self):
        stats = hl_analysis.performance_stats(self._account_with_history())
        self.assertEqual(stats["closed"], 5)
        self.assertEqual(stats["wins"], 2)
        self.assertEqual(stats["losses"], 3)
        self.assertAlmostEqual(stats["win_rate"], 40.0)
        self.assertAlmostEqual(stats["total_pnl"], 5.0)
        self.assertAlmostEqual(stats["avg_win"], 30.0)
        self.assertAlmostEqual(stats["avg_loss"], -18.33)
        self.assertEqual(stats["by_side"]["long"], {"count": 2, "wins": 2, "pnl": 60.0})
        self.assertEqual(stats["by_side"]["short"], {"count": 3, "wins": 0, "pnl": -55.0})
        self.assertEqual(stats["by_coin"]["BTC"], {"count": 2, "wins": 2, "pnl": 60.0})
        self.assertEqual(stats["by_coin"]["ETH"], {"count": 3, "wins": 0, "pnl": -55.0})
        self.assertEqual(stats["exits"], {"close": 2, "stop": 1, "target": 1, "liquidated": 1})
        self.assertEqual(stats["loss_streak_now"], 3)
        self.assertEqual(stats["longest_loss_streak"], 3)
        self.assertAlmostEqual(stats["avg_win_leverage"], 4.0)
        self.assertAlmostEqual(stats["avg_loss_leverage"], 5.0)

    def test_lessons_rules_fire(self):
        stats = hl_analysis.performance_stats(self._account_with_history())
        lessons = hl_analysis.learning_lessons(stats)
        self.assertEqual(len(lessons), 4)
        self.assertIn("Verliesreeks van 3", lessons[0])
        self.assertIn("short-trades", lessons[1])
        self.assertIn("netto verlies", lessons[1])
        self.assertIn("liquidatie", lessons[2].lower())
        self.assertIn("Verliezende trades", lessons[3])
        self.assertIn("verlaag leverage", lessons[3])

    def test_lessons_win_rate_rule(self):
        trades = []
        for i in range(6):
            trades.append(self._trade("open", "BTC", "long", None, f"o{i}"))
            pnl = 10.0 if i == 0 else -10.0
            trades.append(self._trade("close", "BTC", "long", pnl, f"c{i}"))
        account = fresh_account()
        account["trades"] = trades
        stats = hl_analysis.performance_stats(account)
        lessons = hl_analysis.learning_lessons(stats)
        self.assertTrue(any("Win-rate" in lesson for lesson in lessons))

    def test_learning_block_content(self):
        block = hl_analysis.learning_block(self._account_with_history())
        self.assertIn("gesloten trades: 5", block)
        self.assertIn("win-rate 40%", block)
        self.assertIn("JOURNAAL", block)
        self.assertIn("LESSEN", block)
        self.assertIn("BTC long", block)
        self.assertIn("ETH short", block)
        self.assertIn("entry-idee: oversold bounce", block)
        self.assertIn("LIQUIDATIE", block)
        self.assertIn("+50% op marge", block)

    def test_prompt_includes_learning(self):
        account = self._account_with_history()
        candles = {"BTC": make_candles([float(i) for i in range(80, 120)])}
        rows = hl_analysis.build_market_rows(PRICES, META, candles)
        prompt = hl_analysis.build_cycle_prompt(
            rows, candles, account, PRICES, PARAMS, None, [], None, []
        )
        self.assertIn("PRESTATIEVERLEDEN", prompt)
        self.assertIn("JOURNAAL", prompt)
        self.assertIn("prestatieverleden mee", prompt)

    def test_prompt_without_history_shows_placeholder(self):
        candles = {"BTC": make_candles([float(i) for i in range(80, 120)])}
        rows = hl_analysis.build_market_rows(PRICES, META, candles)
        prompt = hl_analysis.build_cycle_prompt(
            rows, candles, fresh_account(), PRICES, PARAMS, None, [], None, []
        )
        self.assertIn("PRESTATIEVERLEDEN", prompt)
        self.assertIn("nog geen gesloten trades", prompt)

    def test_report_includes_learning(self):
        account = self._account_with_history()
        candles = {"BTC": make_candles([float(i) for i in range(80, 120)])}
        rows = hl_analysis.build_market_rows(PRICES, META, candles)
        decision = {"market_view": "-", "rationale": "", "key_risks": [], "actions": []}
        report = hl_analysis.render_cycle_report(
            now_iso="2026-11-14T12:00:00+00:00", rows=rows, account=account,
            prices=PRICES, params=PARAMS, fear_greed=None, news=[],
            news_sentiment=None, decision=decision, events=[], model="m",
        )
        self.assertIn("## Prestatieverleden", report)
        self.assertIn("Win-rate", report)
        self.assertIn("Lessen voor volgende trades", report)

    def test_report_without_history_has_no_learning_section(self):
        candles = {"BTC": make_candles([float(i) for i in range(80, 120)])}
        rows = hl_analysis.build_market_rows(PRICES, META, candles)
        decision = {"market_view": "-", "rationale": "", "key_risks": [], "actions": []}
        report = hl_analysis.render_cycle_report(
            now_iso="2026-11-14T12:00:00+00:00", rows=rows, account=fresh_account(),
            prices=PRICES, params=PARAMS, fear_greed=None, news=[],
            news_sentiment=None, decision=decision, events=[], model="m",
        )
        self.assertNotIn("Prestatieverleden", report)


    def test_next_cycle_delay_seconds_slots(self):
        delay = hl_analysis.next_cycle_delay_seconds("2026-11-15T10:00:00+00:00", 5)
        self.assertAlmostEqual(delay, 300.0)
        delay = hl_analysis.next_cycle_delay_seconds("2026-11-15T10:30:00+00:00", 5)
        self.assertAlmostEqual(delay, 2100.0)
        # exactly at the slot -> next hour
        delay = hl_analysis.next_cycle_delay_seconds("2026-11-15T10:05:00+00:00", 5)
        self.assertAlmostEqual(delay, 3600.0)
        # ends within a minute of the next slot -> skip to the following hour
        delay = hl_analysis.next_cycle_delay_seconds("2026-11-15T10:04:59+00:00", 5)
        self.assertAlmostEqual(delay, 3601.0)
        # naive timestamp is treated as UTC
        delay = hl_analysis.next_cycle_delay_seconds("2026-11-15T10:00:00", 5)
        self.assertAlmostEqual(delay, 300.0)
        # broken input falls back to one hour
        self.assertAlmostEqual(hl_analysis.next_cycle_delay_seconds("geen-tijd", 5), 3600.0)

    def test_next_cycle_delay_seconds_clamps_minute(self):
        delay = hl_analysis.next_cycle_delay_seconds("2026-11-15T10:00:00+00:00", 99)
        self.assertAlmostEqual(delay, 3540.0)  # treated as :59
        delay = hl_analysis.next_cycle_delay_seconds("2026-11-15T10:00:00+00:00", -3)
        self.assertAlmostEqual(delay, 3600.0)  # treated as :00 -> next hour


    def test_guard_sleep_steps_lands_on_slot(self):
        # exact multiple: last step ends precisely on the slot
        self.assertEqual(hl_analysis.guard_sleep_steps(3600, 900), [900, 900, 900, 900])
        # remainder becomes the final step, never a guard past the slot
        self.assertEqual(hl_analysis.guard_sleep_steps(1860, 900), [900, 900, 60])
        # wait shorter than one guard interval: single unguarded sleep
        self.assertEqual(hl_analysis.guard_sleep_steps(500, 900), [500])
        # guard disabled
        self.assertEqual(hl_analysis.guard_sleep_steps(3600, 0), [3600])
        # sums always equal the target
        for target, guard in ((2745, 900), (3601, 900), (7200, 14400), (120, 60)):
            self.assertAlmostEqual(sum(hl_analysis.guard_sleep_steps(target, guard)), target)




class ShadowLadderTests(unittest.TestCase):
    """Upgrade B in shadow-modus: de ladder meet, grijpt nooit in."""

    def _open(self, account, direction="long", stop=None, target=None, margin_fraction=0.10):
        events = hl_analysis.apply_actions(
            account,
            [{"action": "open", "coin": "BTC", "direction": direction,
              "leverage": 5, "margin_fraction": margin_fraction,
              "stop_loss": stop, "take_profit": target, "rationale": "shadow-test"}],
            PRICES, "2026-11-14T11:00:00+00:00", PARAMS, META,
        )
        self.assertEqual(events[0]["type"], "open")
        return account["positions"]["BTC"]

    def _tracker(self, account):
        return account["shadow_b"]["positions"]["BTC"]

    def test_ladder_phases_be_trailing_and_shadow_exit_long(self):
        account = fresh_account()
        self._open(account, stop=99.0, target=105.0)
        # entry 100, SL 99 -> sl_dist 1.0, size 5 -> 1R = 5 USD
        tracker = self._tracker(account)
        self.assertAlmostEqual(tracker["r_unit_usd"], 5.0)
        self.assertEqual(tracker["phase"], 0)
        self.assertEqual(tracker["shadow_stop"], 99.0)

        # +0.5R: geen faseverandering
        hl_analysis.shadow_b_update(account, {"BTC": 100.5}, "t1")
        self.assertEqual(tracker["phase"], 0)
        self.assertAlmostEqual(tracker["mfe_r"], 0.5)

        # +1.2R: BE-lock op entry + rondje fees (100.09)
        hl_analysis.shadow_b_update(account, {"BTC": 101.2}, "t2")
        self.assertEqual(tracker["phase"], 1)
        self.assertAlmostEqual(tracker["shadow_stop"], 100.09, places=4)

        # +3R: trailing start en direct getraild op top - 1R
        hl_analysis.shadow_b_update(account, {"BTC": 103.0}, "t3")
        self.assertEqual(tracker["phase"], 2)
        self.assertAlmostEqual(tracker["shadow_stop"], 102.0)

        # top 104.5 -> trail gaat mee naar 103.5
        hl_analysis.shadow_b_update(account, {"BTC": 104.5}, "t4")
        self.assertAlmostEqual(tracker["shadow_stop"], 103.5)

        # terug naar 103.0 raakt de trail: schaduw-exit op 103.5 (r=3.5)
        events = hl_analysis.shadow_b_update(account, {"BTC": 103.0}, "t5")
        self.assertEqual(tracker["shadow_exit"]["price"], 103.5)
        self.assertAlmostEqual(tracker["shadow_exit"]["r"], 3.5)
        self.assertTrue(any(e["type"] == "shadow-exit" for e in events))

        # de ECHTE positie is onaangeroerd: zelfde SL, geen extra trades
        self.assertEqual(account["positions"]["BTC"]["stop_loss"], 99.0)
        self.assertEqual(len(account["trades"]), 1)

        # echte close op 102.5 -> journaal vergeleelt ladder (3.5R) vs echt (2.5R)
        hl_analysis.close_position(account, "BTC", 102.5, "t6", "close")
        journal = account["shadow_b"]["journal"]
        self.assertEqual(len(journal), 1)
        entry = journal[0]
        self.assertEqual(entry["comparison"], "ladder-eerst")
        self.assertAlmostEqual(entry["real_r"], 2.5)
        self.assertAlmostEqual(entry["shadow_r"], 3.5)
        self.assertAlmostEqual(entry["delta_r"], 1.0)

        summary = hl_analysis.shadow_b_summary(account)
        self.assertEqual(summary["resolved"], 1)
        self.assertAlmostEqual(summary["delta_r"], 1.0)
        # tracker opgeruimd
        self.assertNotIn("BTC", account["shadow_b"]["positions"])

    def test_ladder_mirrors_for_short(self):
        account = fresh_account()
        self._open(account, direction="short", stop=101.5, target=95.0)
        tracker = self._tracker(account)
        sl_dist = 1.5

        # short: gunst is dalen; entry 100, prijs 98.5 -> +1R -> BE-lock
        hl_analysis.shadow_b_update(account, {"BTC": 98.5}, "t1")
        self.assertEqual(tracker["phase"], 1)
        self.assertAlmostEqual(tracker["shadow_stop"], 100.0 - 100.0 * hl_analysis.TAKER_FEE_RATE * 2, places=4)

        # prijs 95 -> +3.33R: trailing, stop op fav + 1R afstand
        hl_analysis.shadow_b_update(account, {"BTC": 95.0}, "t2")
        self.assertEqual(tracker["phase"], 2)
        self.assertAlmostEqual(tracker["shadow_stop"], 95.0 + sl_dist)

        # terug naar 96.6 raakt de trail: schaduw-exit op 96.5
        hl_analysis.shadow_b_update(account, {"BTC": 96.6}, "t3")
        self.assertAlmostEqual(tracker["shadow_exit"]["price"], 96.5)
        self.assertAlmostEqual(tracker["shadow_exit"]["r"], (100.0 - 96.5) / sl_dist, places=3)
        self.assertIn("BTC", account["positions"])  # echt blijft open

    def test_shadow_stop_follows_actual_sl_until_first_r(self):
        account = fresh_account()
        self._open(account, stop=99.0)
        # dip naar 98.9 raakt de (schaduw-)SL gewoon, nog in fase 0
        events = hl_analysis.shadow_b_update(account, {"BTC": 98.9}, "t1")
        tracker = self._tracker(account)
        self.assertEqual(tracker["shadow_exit"]["reason"], "shadow-stop")
        self.assertAlmostEqual(tracker["shadow_exit"]["r"], -1.0)
        # en de echte stop-close journalt netjes met delta 0
        hl_analysis.process_forced_exits(account, {"BTC": 98.9}, "t2")
        entry = account["shadow_b"]["journal"][0]
        self.assertEqual(entry["reason"], "stop")
        self.assertAlmostEqual(entry["real_r"], -1.0)
        self.assertAlmostEqual(entry["delta_r"], 0.0)

    def test_no_sl_means_dormant_ladder(self):
        account = fresh_account()
        self._open(account, stop=None)
        tracker = self._tracker(account)
        self.assertIsNone(tracker["r_unit_usd"])
        hl_analysis.shadow_b_update(account, {"BTC": 130.0}, "t1")
        self.assertEqual(tracker["phase"], 0)
        self.assertIsNone(tracker["shadow_exit"])
        hl_analysis.close_position(account, "BTC", 110.0, "t2", "close")
        entry = account["shadow_b"]["journal"][0]
        self.assertEqual(entry["comparison"], "geen-SL: ladder inactief")
        summary = hl_analysis.shadow_b_summary(account)
        self.assertEqual(summary["resolved"], 0)
        self.assertEqual(summary["trades"], 1)

    def test_real_exit_while_ladder_active_is_unresolved(self):
        account = fresh_account()
        self._open(account, stop=99.0, target=105.0)
        hl_analysis.shadow_b_update(account, {"BTC": 100.5}, "t1")  # +0.5R, geen fase
        hl_analysis.close_position(account, "BTC", 100.5, "t2", "close")
        entry = account["shadow_b"]["journal"][0]
        self.assertEqual(entry["comparison"], "echte-exit-eerst")
        self.assertIsNone(entry["delta_r"])

    def test_liquidation_journals_without_crash(self):
        account = fresh_account()
        # SL ver onder de liq-prijs zodat de liquidatie-tak eerst slaat
        self._open(account, stop=80.0)
        hl_analysis.process_forced_exits(account, {"BTC": 81.0}, "t1")
        self.assertNotIn("BTC", account["positions"])
        entry = account["shadow_b"]["journal"][0]
        self.assertEqual(entry["reason"], "liquidated")
        self.assertAlmostEqual(entry["real_r"], (82.0 - 100.0) / 20.0, places=3)

    def test_report_lines_and_purity_for_model(self):
        account = fresh_account()
        self._open(account, stop=99.0)
        hl_analysis.shadow_b_update(account, {"BTC": 101.2}, "t1")
        lines = hl_analysis.shadow_b_report_lines(account, {"BTC": 101.2})
        self.assertTrue(any("Shadow-exitladder" in ln for ln in lines))
        self.assertTrue(any("BE-lock" in ln for ln in lines))
        # de model-prompt blijft schoon: geen shadow-tekst in de promptblokken
        prompt_block = hl_analysis.account_state_for_prompt(account, {"BTC": 101.2})
        self.assertNotIn("shadow", prompt_block.lower())
        # lege shadow-state -> geen sectie
        self.assertEqual(hl_analysis.shadow_b_report_lines(fresh_account(), PRICES), [])

    def test_shadow_update_never_mutates_trading_state(self):
        account = fresh_account()
        self._open(account, stop=99.0, target=105.0)
        before_balance = account["balance"]
        before_realized = account["realized_pnl"]
        before_sl = account["positions"]["BTC"]["stop_loss"]
        hl_analysis.shadow_b_update(account, {"BTC": 104.0}, "t1")
        hl_analysis.shadow_b_update(account, {"BTC": 101.5}, "t2")
        self.assertEqual(account["balance"], before_balance)
        self.assertEqual(account["realized_pnl"], before_realized)
        self.assertEqual(account["positions"]["BTC"]["stop_loss"], before_sl)
        self.assertEqual(len(account["trades"]), 1)
        self.assertEqual(account["positions"]["BTC"]["take_profit"], 105.0)


class SeedTests(unittest.TestCase):
    """Seed-import/-export: historie terugzetten en rond Bowler bewaren."""

    def _account_with_roundtrip(self):
        account = fresh_account()
        hl_analysis.apply_actions(
            account,
            [{"action": "open", "coin": "BTC", "direction": "long",
              "leverage": 5, "margin_fraction": 0.10,
              "stop_loss": 99.0, "take_profit": 105.0,
              "rationale": "seed-test long"}],
            PRICES, "2026-11-14T11:00:00+00:00", PARAMS, META,
        )
        hl_analysis.close_position(account, "BTC", 102.5, "2026-11-14T12:00:00+00:00", "close")
        return account

    def test_seed_roundtrip_preserves_everything(self):
        account = self._account_with_roundtrip()
        account["cycles"] = 42
        account["halted"] = True
        account["halt_reason"] = "test"
        # shadow-meeting deed mee: 1 journalentry via een echte open+close
        journal_before = list(account["shadow_b"]["journal"])
        seed = hl_analysis.seed_from_account(account)
        seeded = hl_analysis.account_from_seed(seed, "2026-11-15T09:00:00+00:00")

        self.assertEqual(seeded["balance"], account["balance"])
        self.assertEqual(seeded["realized_pnl"], account["realized_pnl"])
        self.assertEqual(seeded["fees_paid"], account["fees_paid"])
        self.assertEqual(seeded["cycles"], 42)
        self.assertEqual(seeded["started_at"], account["started_at"])
        self.assertEqual(seeded["halted"], True)
        self.assertEqual(seeded["halt_reason"], "test")
        self.assertEqual(seeded["trades"], account["trades"])
        self.assertEqual(seeded["positions"], {})
        self.assertEqual(seeded["shadow_b"]["journal"], journal_before)
        self.assertEqual(
            hl_analysis.performance_stats(seeded), hl_analysis.performance_stats(account)
        )
        self.assertEqual(
            hl_analysis.learning_block(seeded), hl_analysis.learning_block(account)
        )
        # dagbasis van de actieve dag blijft overgedragen bij een roundtrip
        self.assertEqual(seeded["day_key"], "2026-11-15")
        self.assertEqual(seeded["day_start_equity"], account["day_start_equity"])

    def test_seed_manual_history_import(self):
        seed = {
            "version": 1,
            "balance": 990.29,
            "realized_pnl": -5.67,
            "fees_paid": 4.04,
            "cycles": 173,
            "started_at": "2026-11-15T12:10:00Z",
            "equity_high": 1004.56,
            "trades": [
                {"coin": "BTC", "side": "long", "action": "open", "price": 100.0,
                 "size": 5.0, "notional": 500.0, "leverage": 5, "margin": 100.0,
                 "fee": 0.225, "at": "2026-11-16T10:05:00+00:00",
                 "reason": "hersteld uit rapport na herstart"},
                {"coin": "BTC", "side": "long", "action": "target", "price": 101.2,
                 "size": 5.0, "notional": 506.0, "leverage": 5, "margin": 100.0,
                 "pnl": 6.0, "fee": 0.227, "at": "2026-11-16T13:05:00+00:00",
                 "reason": "take-profit geraakt"},
            ],
            "shadow_journal": [
                {"coin": "BTC", "side": "long",
                 "opened_at": "2026-11-16T10:05:00+00:00",
                 "closed_at": "2026-11-16T13:05:00+00:00",
                 "reason": "target", "r_unit_usd": 5.0, "mfe_r": 1.4, "mae_r": -0.2,
                 "phase_at_close": 1, "shadow_stop_at_close": 100.09,
                 "real_exit_price": 101.2, "real_r": 1.2, "shadow_exit": None,
                 "shadow_r": None, "delta_r": None, "comparison": "echte-exit-eerst"},
            ],
        }
        account = hl_analysis.account_from_seed(seed, "2026-11-23T06:20:00+00:00")
        stats = hl_analysis.performance_stats(account)
        self.assertEqual(stats["closed"], 1)
        self.assertEqual(stats["wins"], 1)
        self.assertAlmostEqual(stats["total_pnl"], 6.0)
        self.assertEqual(stats["exits"], {"target": 1})
        self.assertEqual(account["cycles"], 173)
        self.assertEqual(account["started_at"], "2026-11-15T12:10:00+00:00")  # Z genormaliseerd
        self.assertAlmostEqual(account["equity_high"], 1004.56)
        self.assertAlmostEqual(account["balance"], 990.29)
        self.assertEqual(len(account["shadow_b"]["journal"]), 1)
        summary = hl_analysis.shadow_b_summary(account)
        self.assertEqual(summary["trades"], 1)
        self.assertEqual(summary["resolved"], 0)  # echte-exit-eerst = unresolved

    def test_account_from_seed_rejects_incomplete_or_bad_rows(self):
        base = {
            "version": 1, "balance": 1000.0, "realized_pnl": 0.0, "fees_paid": 0.0,
        }
        # balance/realized_pnl/fees_paid zijn verplicht
        with self.assertRaises(ValueError):
            hl_analysis.account_from_seed({"version": 1}, "t")
        # ongeldige actie
        bad_action = dict(base, trades=[
            {"coin": "BTC", "side": "long", "action": "buy", "price": 100.0,
             "size": 1.0, "leverage": 5, "fee": 0.0, "at": "2026-11-16T10:05:00+00:00"},
        ])
        with self.assertRaises(ValueError):
            hl_analysis.account_from_seed(bad_action, "t")
        # ongeldige timestamp
        bad_time = dict(base, trades=[
            {"coin": "BTC", "side": "long", "action": "open", "price": 100.0,
             "size": 1.0, "leverage": 5, "fee": 0.0, "at": "gisteren"},
        ])
        with self.assertRaises(ValueError):
            hl_analysis.account_from_seed(bad_time, "t")
        # close-rij zonder pnl
        no_pnl = dict(base, trades=[
            {"coin": "BTC", "side": "long", "action": "open", "price": 100.0,
             "size": 1.0, "leverage": 5, "fee": 0.0, "at": "2026-11-16T10:05:00+00:00"},
            {"coin": "BTC", "side": "long", "action": "close", "price": 101.0,
             "size": 1.0, "leverage": 5, "fee": 0.0, "at": "2026-11-16T11:05:00+00:00"},
        ])
        with self.assertRaises(ValueError):
            hl_analysis.account_from_seed(no_pnl, "t")
        # onbekende seed-versie
        with self.assertRaises(ValueError):
            hl_analysis.account_from_seed(dict(base, version=99), "t")
        # geen object
        with self.assertRaises(ValueError):
            hl_analysis.account_from_seed(["geen", "object"], "t")

    def test_seed_never_carries_positions(self):
        account = self._account_with_roundtrip()
        seed = hl_analysis.seed_from_account(account)
        self.assertNotIn("positions", seed)
        seeded = hl_analysis.account_from_seed(seed, "t")
        self.assertEqual(seeded["positions"], {})


if __name__ == "__main__":
    unittest.main()
