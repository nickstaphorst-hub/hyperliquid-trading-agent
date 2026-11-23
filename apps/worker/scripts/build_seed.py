"""Bouwt de seed-payload (historie-import) voor de herstart van de agent.

Reconstrueert het paper-account uit de beëindigde run: balance, 12 gesloten
trades, lessen-context en het shadow-journaal (T10-12, verbatim bekend).
Reconstructies zijn in de reason-velden gemarkeerd als hersteld.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worker.workflows import hl_analysis  # noqa: E402

TAKER_FEE = 0.00045
HERSTELD = "hersteld uit cyclusrapporten na herstart"


def rows(coin, side, open_at, close_at, entry, exitprice, margin, lev, pnl, action,
         reason_open, reason_close):
    size = round(margin * lev / entry, 4)
    fees_open = round(margin * lev * TAKER_FEE, 4)
    notional = round(margin * lev, 4)
    close_notional = round(abs(exitprice * size), 4)
    return [
        {"coin": coin, "side": side, "action": "open", "price": entry, "size": size,
         "notional": notional, "leverage": lev, "margin": margin, "pnl": None,
         "fee": fees_open, "at": open_at, "reason": reason_open},
        {"coin": coin, "side": side, "action": action, "price": exitprice, "size": size,
         "notional": close_notional, "leverage": lev, "margin": margin,
         "pnl": round(pnl, 2), "fee": round(close_notional * TAKER_FEE, 4),
         "at": close_at, "reason": reason_close},
    ]


trades = []
# T1: BTC long, TP-reeks maandag
trades += rows("BTC", "long", "2026-11-16T10:05:00+00:00", "2026-11-16T13:05:00+00:00",
               64.838, 65.43, 99.66, 5, 4.56, "target",
               f"Contrair long BTC na crash-weekend: zwaar oversold, Extreme Fear. ({HERSTELD})",
               "take-profit geraakt")
# T2: BTC long, model-close
trades += rows("BTC", "long", "2026-11-16T14:05:00+00:00", "2026-11-16T15:05:00+00:00",
               64.396, 64.1535, 100.40, 5, -1.89, "close",
               f"Vervolglong BTC op momentum rond de middag (RSI 46.3). ({HERSTELD})",
               "model-close: momentum verzwakt onder de 4h-SMA's")
# T3: SOL long
trades += rows("SOL", "long", "2026-11-17T16:05:00+00:00", "2026-11-17T21:05:00+00:00",
               72.69, 71.7736, 100.10, 5, -6.31, "close",
               f"Long SOL op dagkracht, breakout boven recent hoog. ({HERSTELD})",
               "model-close: breakout faalt, SOL verliest relatiekracht")
# T4: SOL short
trades += rows("SOL", "short", "2026-11-18T01:05:00+00:00", "2026-11-18T06:05:00+00:00",
               74.73, 75.2731, 49.80, 5, -1.81, "close",
               f"Short SOL na uitputting van de +4.2% nachtelijke rally. ({HERSTELD})",
               "model-close: cover na tegendruk-verlies")
# T5: BTC long, woensdagnacht-dip
trades += rows("BTC", "long", "2026-11-18T03:05:00+00:00", "2026-11-18T07:05:00+00:00",
               62.578, 63.7615, 49.64, 5, 4.69, "close",
               f"Long BTC op oversold-nachttop, reclaim van steun. ({HERSTELD})",
               "model-close: winst veiliggesteld na 7%-schommeling")
# T6: BTC long
trades += rows("BTC", "long", "2026-11-18T09:05:00+00:00", "2026-11-18T12:05:00+00:00",
               63.606, 63.8259, 49.70, 5, 0.86, "close",
               f"Vervolglong BTC op ochtendmomentum. ({HERSTELD})",
               "model-close: break-out zonder vervolg")
# T7: BTC long
trades += rows("BTC", "long", "2026-11-18T12:05:00+00:00", "2026-11-18T14:05:00+00:00",
               63.731, 63.7977, 49.70, 5, 0.26, "close",
               f"Scalp-long rond middaguur-weerstand, klein formaat. ({HERSTELD})",
               "model-close: geen vervolg")
# T8: BTC long
trades += rows("BTC", "long", "2026-11-18T14:05:00+00:00", "2026-11-18T16:05:00+00:00",
               63.770, 63.954, 49.50, 5, 0.72, "close",
               f"Long-co\u00efncident met de middagschommeling. ({HERSTELD})",
               "model-close: momentum stokt v\u00f3\u00f3r het sluiten")
# T9: BTC long, donderdag middag (exact uit journaal)
trades += rows("BTC", "long", "2026-11-19T15:05:00+00:00", "2026-11-19T16:05:00+00:00",
               63.780, 63.0815, 99.84, 5, -5.47, "close",
               f"Long BTC opmiddagse dip-reclaim. ({HERSTELD})",
               "model-close: rebound houdt geen stand")
# T10: BTC long, stop (reconstructed sizing, exit exact)
trades += rows("BTC", "long", "2026-11-19T16:06:33+00:00", "2026-11-19T17:06:48+00:00",
               63.431, 62.50, 31.17, 5, -2.29, "stop",
               f"Re-entry long na stoploopse dip van donderdag. ({HERSTELD})",
               "stop-loss geraakt")
# T11: BTC long, zaterdag (exit + getallen exact uit journaal)
trades += rows("BTC", "long", "2026-11-21T15:06:44+00:00", "2026-11-21T16:07:41+00:00",
               60374.5, 59843.5, 98.995, 5, -4.35, "close",
               "Contraire long op BTC: RSI 36.3 (oversold) in Extreme Fear (12/100)",
               "model-close: bounce op steun blijft uit")
# T12: BTC long, zondag (exit exact, winstreeks verbroken)
trades += rows("BTC", "long", "2026-11-22T09:07:01+00:00", "2026-11-22T10:06:52+00:00",
               59723.5, 61024.5, 49.32, 5, 5.37, "close",
               f"Long BTC op zondagochtend-momentumreclaim onder de 60k. ({HERSTELD})",
               "model-close: +5 USD bereikt, winst veiliggesteld")

shadow_journal = [
    {"coin": "BTC", "side": "long",
     "opened_at": "2026-11-19T16:06:33+00:00", "closed_at": "2026-11-19T17:06:48+00:00",
     "reason": "stop", "r_unit_usd": 2.2872, "mfe_r": 0.084, "mae_r": -0.412,
     "phase_at_close": 0, "shadow_stop_at_close": 62.5,
     "real_exit_price": 62.5, "real_r": -1.0, "shadow_exit": None,
     "shadow_r": None, "delta_r": None, "comparison": "echte-exit-eerst"},
    {"coin": "BTC", "side": "long",
     "opened_at": "2026-11-21T15:06:44+00:00", "closed_at": "2026-11-21T16:07:41+00:00",
     "reason": "close", "r_unit_usd": 7.1695, "mfe_r": 0.0, "mae_r": -0.652,
     "phase_at_close": 0, "shadow_stop_at_close": 59460,
     "real_exit_price": 59843.5, "real_r": -0.607, "shadow_exit": None,
     "shadow_r": None, "delta_r": None, "comparison": "echte-exit-eerst"},
    {"coin": "BTC", "side": "long",
     "opened_at": "2026-11-22T09:07:01+00:00", "closed_at": "2026-11-22T10:06:52+00:00",
     "reason": "close", "r_unit_usd": 5.0455, "mfe_r": 0.864, "mae_r": 0.0,
     "phase_at_close": 0, "shadow_stop_at_close": 58500,
     "real_exit_price": 61024.5, "real_r": None, "shadow_exit": None,
     "shadow_r": None, "delta_r": None, "comparison": "echte-exit-eerst"},
]

seed = {
    "version": 1,
    "start_balance": 1000.0,
    "balance": 990.2942,
    "realized_pnl": -5.6699,
    "fees_paid": 4.0360,
    "cycles": 173,
    "started_at": "2026-11-15T12:10:00+00:00",
    "equity_high": 1004.56,
    "halted": False,
    "halt_reason": None,
    "trades": trades,
    "shadow_journal": shadow_journal,
}

account = hl_analysis.account_from_seed(seed, "2026-11-23T06:20:00+00:00")
stats = hl_analysis.performance_stats(account)
print("=== Seed validatie OK ===")
print(f"closed={stats['closed']} wins={stats['wins']} losses={stats['losses']} "
      f"winrate={stats['win_rate']}% total_pnl={stats['total_pnl']}")
print(f"exits={stats['exits']}")
print(f"loss_streak_now={stats['loss_streak_now']} longest={stats['longest_loss_streak']}")
print(f"by_side={stats['by_side']}")
print(f"by_coin={stats['by_coin']}")
print("--- Lessen die het model de eerste cyclus ziet ---")
for les in hl_analysis.learning_lessons(stats):
    print(f"- {les}")
print("--- Journal-jes (laatste 3) ---")
for line in hl_analysis.learning_block(account).splitlines()[-8:]:
    print(line)
print(f"--- Shadow journaal: {len(account['shadow_b']['journal'])} entries ---")
print(hl_analysis.shadow_b_summary(account))

out = Path(__file__).resolve().parents[1] / "seed_payload.json"
out.write_text(json.dumps(seed, indent=1), encoding="utf-8")
print(f"geschreven: {out} ({out.stat().st_size} bytes)")
