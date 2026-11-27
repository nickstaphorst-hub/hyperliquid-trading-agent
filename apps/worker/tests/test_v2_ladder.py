"""Unit tests voor de v2-uitbreidingen (live trailing-ladder + harde regels).

Run from apps/worker/tests:  python3 -m unittest test_v2_ladder.py
Alles stdlib-only, geen netwerk.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from worker.workflows import hl_analysis  # noqa: E402

NOW = "2026-11-27T12:00:00+00:00"

META = {
    "BTC": {"coin": "BTC", "max_leverage": 40},
}

V2_PARAMS = {
    "universe": ["BTC", "ETH", "SOL"],
    "max_leverage": 5,
    "max_margin_fraction": 0.10,
    "max_open_positions": 3,
    "daily_loss_limit_pct": 5.0,
    "risk_profile": "balanced",
    "exit_engine": "ladder",
    "ladder_cfg": {"be_lock_r": 1.0, "trail_start_r": 2.0, "trail_dist_r": 1.0, "tp_r": 0.0},
    "hard_risk_rules": True,
    "min_rr": 1.2,
    "fee_gain_multiple": 3.0,
    "margin_throttle_streak": 3,
}

V1_PARAMS = {k: v for k, v in V2_PARAMS.items() if k not in (
    "exit_engine", "ladder_cfg", "hard_risk_rules", "min_rr",
    "fee_gain_multiple", "margin_throttle_streak",
)}


def fresh_account():
    return hl_analysis.new_account(1000.0, NOW)


def open_btc(account, params, sl=95.0, tp=110.0, margin=0.05, leverage=2):
    """Open een BTC long via de gewone engine-route (met alle injecties)."""
    actions = [{
        "action": "open", "coin": "BTC", "direction": "long",
        "leverage": leverage, "margin_fraction": margin,
        "stop_loss": sl, "take_profit": tp, "rationale": "test",
    }]
    events = hl_analysis.apply_actions(
        account, actions, {"BTC": 100.0}, NOW, params, META
    )
    return events


class LadderEngineTests(unittest.TestCase):
    """De live ladder: fase-upgrades, trail-updates en ladder-stops."""

    def setUp(self):
        self.account = fresh_account()
        open_btc(self.account, V2_PARAMS)  # entry 100, SL 95 -> 1R = 5

    def test_open_creates_ladder_state(self):
        pos = self.account["positions"]["BTC"]
        state = pos["ladder_state"]
        self.assertEqual(state["phase"], 0)
        self.assertAlmostEqual(state["sl_dist"], 5.0)
        self.assertEqual(state["fav_price"], 100.0)
        self.assertGreater(state["be_price"], 100.0)  # entry + rondje fees

    def test_no_ladder_state_in_v1_mode(self):
        account = fresh_account()
        open_btc(account, V1_PARAMS)
        self.assertNotIn("ladder_state", account["positions"]["BTC"])

    def test_be_lock_moves_stop(self):
        events = hl_analysis.ladder_manage_exits(
            self.account, {"BTC": 105.0}, NOW, V2_PARAMS
        )
        self.assertTrue(any(e["type"] == "ladder-be-lock" for e in events))
        pos = self.account["positions"]["BTC"]
        self.assertEqual(pos["stop_loss"], self.account["positions"]["BTC"]["ladder_state"]["be_price"])

    def test_trailing_and_ladder_stop_close(self):
        # +1R: BE-lock, +3R: trailing start, dan terug naar de trail (110)
        hl_analysis.ladder_manage_exits(self.account, {"BTC": 105.0}, NOW, V2_PARAMS)
        events = hl_analysis.ladder_manage_exits(
            self.account, {"BTC": 115.0}, NOW, V2_PARAMS
        )
        self.assertTrue(any(e["type"] == "ladder-trailing-start" for e in events))
        self.assertTrue(any(e["type"] == "ladder-trail-update" for e in events))
        self.assertAlmostEqual(
            self.account["positions"]["BTC"]["stop_loss"], 110.0
        )
        events = hl_analysis.ladder_manage_exits(
            self.account, {"BTC": 110.0}, NOW, V2_PARAMS
        )
        self.assertTrue(any(e["type"] == "close" for e in events))
        self.assertNotIn("BTC", self.account["positions"])
        last = self.account["trades"][-1]
        self.assertEqual(last["action"], "ladder-trailing-stop")
        # close op 110: +10 per unit op entry 100, size 2 (notional 200)
        self.assertAlmostEqual(last["pnl"], 10.0, places=4)

    def test_be_close_logs_ladder_be_reason(self):
        hl_analysis.ladder_manage_exits(self.account, {"BTC": 105.0}, NOW, V2_PARAMS)
        be = self.account["positions"]["BTC"]["stop_loss"]
        events = hl_analysis.ladder_manage_exits(
            self.account, {"BTC": be}, NOW, V2_PARAMS
        )
        self.assertTrue(any(e["type"] == "close" for e in events))
        self.assertEqual(self.account["trades"][-1]["action"], "ladder-be-lock")

    def test_ladder_inactive_in_v1_mode(self):
        events = hl_analysis.ladder_manage_exits(
            self.account, {"BTC": 105.0}, NOW, V1_PARAMS
        )
        self.assertEqual(events, [])
        self.assertEqual(self.account["positions"]["BTC"]["stop_loss"], 95.0)

    def test_ladder_respects_config_knobs(self):
        params = {**V2_PARAMS, "ladder_cfg": {
            "be_lock_r": 0.5, "trail_start_r": 1.0, "trail_dist_r": 0.5, "tp_r": 0.0
        }}
        hl_analysis.ladder_manage_exits(self.account, {"BTC": 103.0}, NOW, params)
        state = self.account["positions"]["BTC"]["ladder_state"]
        self.assertEqual(state["phase"], 1)  # +0.6R >= 0.5R -> BE-lock

    def test_ladder_report_lines(self):
        lines = hl_analysis.ladder_report_lines(
            self.account, {"BTC": 102.0}
        )
        self.assertTrue(any("Trailing-ladder" in line for line in lines))
        self.assertTrue(any("BTC" in line for line in lines))
        self.assertEqual(hl_analysis.ladder_report_lines(fresh_account(), {}), [])


class TpOverrideTests(unittest.TestCase):
    """tp_r-knop: de engine legt een R-gebaseerd TP over het model heen."""

    def test_tp_r_overrides_model_tp(self):
        account = fresh_account()
        params = {**V2_PARAMS, "ladder_cfg": {**V2_PARAMS["ladder_cfg"], "tp_r": 1.5}}
        open_btc(account, params, tp=120.0)
        self.assertAlmostEqual(account["positions"]["BTC"]["take_profit"], 107.5)

    def test_tp_r_zero_keeps_model_tp(self):
        account = fresh_account()
        open_btc(account, V2_PARAMS, tp=110.0)
        self.assertAlmostEqual(account["positions"]["BTC"]["take_profit"], 110.0)


class HardRiskRuleTests(unittest.TestCase):
    """Harde v2-regels in validate_action: R:R, fee-throttle, cooldown, marge."""

    def validate(self, account, action, params=None, prices=None):
        return hl_analysis.validate_action(
            action, account, prices or {"BTC": 100.0}, params or V2_PARAMS, META
        )

    def test_min_rr_rejects(self):
        account = fresh_account()
        ok, reason = self.validate(account, {
            "action": "open", "coin": "BTC", "direction": "long",
            "leverage": 2, "margin_fraction": 0.05,
            "stop_loss": 95.0, "take_profit": 100.5,
        })
        self.assertFalse(ok)
        self.assertIn("R:R", reason)

    def test_missing_sl_or_tp_rejected_with_hard_rules(self):
        account = fresh_account()
        ok, reason = self.validate(account, {
            "action": "open", "coin": "BTC", "direction": "long",
            "leverage": 2, "margin_fraction": 0.05, "take_profit": 110.0,
        })
        self.assertFalse(ok)
        self.assertIn("stop_loss", reason)
        ok, reason = self.validate(account, {
            "action": "open", "coin": "BTC", "direction": "long",
            "leverage": 2, "margin_fraction": 0.05, "stop_loss": 95.0,
        })
        self.assertFalse(ok)
        self.assertIn("take_profit", reason)

    def test_missing_tp_allowed_without_hard_rules(self):
        account = fresh_account()
        params = {**V2_PARAMS, "hard_risk_rules": False}
        ok, _ = self.validate(account, {
            "action": "open", "coin": "BTC", "direction": "long",
            "leverage": 2, "margin_fraction": 0.05, "stop_loss": 95.0,
        }, params)
        self.assertTrue(ok)

    def test_fee_throttle_rejects_tiny_edge(self):
        # tp_dist 0.1 op prijs 100 -> 0.1% van notional, onder de
        # 3x round-trip fees (0.27%): churn, netto verlies.
        account = fresh_account()
        ok, reason = self.validate(account, {
            "action": "open", "coin": "BTC", "direction": "long",
            "leverage": 2, "margin_fraction": 0.05,
            "stop_loss": 99.95, "take_profit": 100.1,
        })
        self.assertFalse(ok)
        self.assertIn("fee-throttle", reason)

    def test_cooldown_after_stop(self):
        account = fresh_account()
        open_btc(account, V2_PARAMS)
        hl_analysis.close_position(account, "BTC", 95.0, NOW, "stop")
        self.assertEqual(account["last_stop_cycle"], account["cycles"])
        ok, reason = self.validate(account, {
            "action": "open", "coin": "ETH", "direction": "long",
            "leverage": 1, "margin_fraction": 0.05,
            "stop_loss": 45.0, "take_profit": 56.5,
        }, prices={"BTC": 100.0, "ETH": 50.0})
        self.assertFalse(ok)
        self.assertIn("cooldown", reason)
        # volgende cyclus mag weer
        account["cycles"] += 1
        prices = {"BTC": 100.0, "ETH": 50.0}
        ok, _ = hl_analysis.validate_action(
            {"action": "open", "coin": "ETH", "direction": "long",
             "leverage": 1, "margin_fraction": 0.05,
             "stop_loss": 45.0, "take_profit": 56.5},
            account, prices, V2_PARAMS, META,
        )
        self.assertTrue(ok)

    def test_margin_throttle_halves_cap_at_streak(self):
        account = fresh_account()
        for i in range(3):  # drie verliezende round-trips
            account["trades"].extend([
                {"coin": "BTC", "side": "long", "action": "open", "price": 100.0,
                 "size": 1.0, "notional": 100.0, "leverage": 1, "margin": 50.0,
                 "pnl": None, "fee": 0.045, "at": NOW, "reason": "x"},
                {"coin": "BTC", "side": "long", "action": "stop", "price": 99.0,
                 "size": 1.0, "notional": 99.0, "leverage": 1, "margin": 50.0,
                 "pnl": -1.0, "fee": 0.045, "at": NOW, "reason": "x"},
            ])
        ok, reason = self.validate(account, {
            "action": "open", "coin": "BTC", "direction": "long",
            "leverage": 2, "margin_fraction": 0.08,
            "stop_loss": 95.0, "take_profit": 110.0,
        })
        self.assertFalse(ok)
        self.assertIn("0.05", reason)  # 0.10 gehalveerd
        # zonder harde regels is 0.08 gewoon toegestaan
        params = {**V2_PARAMS, "hard_risk_rules": False}
        ok, _ = self.validate(account, {
            "action": "open", "coin": "BTC", "direction": "long",
            "leverage": 2, "margin_fraction": 0.08,
            "stop_loss": 95.0, "take_profit": 110.0,
        }, params)
        self.assertTrue(ok)


class LessonEffectivenessTests(unittest.TestCase):
    """Les-effectiviteit: actieve lessen komen in de trade-row van de entry."""

    def test_lessons_recorded_with_hard_rules(self):
        account = fresh_account()
        for i in range(3):
            account["trades"].extend([
                {"coin": "BTC", "side": "long", "action": "open", "price": 100.0,
                 "size": 1.0, "notional": 100.0, "leverage": 1, "margin": 50.0,
                 "pnl": None, "fee": 0.045, "at": NOW, "reason": "x"},
                {"coin": "BTC", "side": "long", "action": "close", "price": 99.0,
                 "size": 1.0, "notional": 99.0, "leverage": 1, "margin": 50.0,
                 "pnl": -1.0, "fee": 0.045, "at": NOW, "reason": "x"},
            ])
        account["cycles"] = 1  # cooldown de volgende cyclus
        events = open_btc(account, V2_PARAMS)
        self.assertEqual(events[0]["type"], "open")
        self.assertIn("lessons", self.account_row(account))
        self.assertTrue(self.account_row(account)["lessons"])

    def account_row(self, account):
        return account["trades"][-1]

    def test_no_lessons_without_hard_rules(self):
        account = fresh_account()
        params = {**V2_PARAMS, "hard_risk_rules": False}
        open_btc(account, params)
        self.assertNotIn("lessons", account["trades"][-1])

    def test_seed_roundtrip_preserves_lessons_and_ladder_actions(self):
        account = fresh_account()
        params = {**V2_PARAMS, "hard_risk_rules": False}
        open_btc(account, params)
        account["trades"].append({
            "coin": "BTC", "side": "long", "action": "ladder-trailing-stop",
            "price": 110.0, "size": 2.0, "notional": 220.0, "leverage": 2,
            "margin": 100.0, "pnl": 20.0, "fee": 0.099, "at": NOW, "reason": "trail",
        })
        seed = hl_analysis.seed_from_account(account)
        seed["trades"][0]["lessons"] = ["les een", "les twee"]
        restored = hl_analysis.account_from_seed(seed, NOW, 1000.0)
        self.assertEqual(restored["trades"][0]["lessons"], ["les een", "les twee"])
        self.assertEqual(restored["trades"][-1]["action"], "ladder-trailing-stop")
        self.assertAlmostEqual(restored["trades"][-1]["pnl"], 20.0)


class PromptAndConfigTests(unittest.TestCase):
    """De v2-prompt-regels en de config-normalisatie."""

    def test_exit_rules_lines_empty_in_v1(self):
        self.assertEqual(hl_analysis.exit_rules_lines(V1_PARAMS), "")

    def test_exit_rules_lines_mention_ladder_and_hard_rules(self):
        text = hl_analysis.exit_rules_lines(V2_PARAMS)
        self.assertIn("trailing-ladder", text)
        self.assertIn("R:R 1.2", text)
        self.assertIn("fee", text)

    def test_ladder_cfg_from_params_defaults(self):
        cfg = hl_analysis.ladder_cfg_from_params({})
        self.assertEqual(cfg["be_lock_r"], 1.0)
        self.assertEqual(cfg["trail_start_r"], 2.0)
        self.assertEqual(cfg["trail_dist_r"], 1.0)
        self.assertEqual(cfg["tp_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
