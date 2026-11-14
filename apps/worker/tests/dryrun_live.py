"""Live dry-run of the pure core against real APIs (no model call)."""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worker.workflows import hl_analysis  # noqa: E402


def post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hl-agent-dryrun/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


coins = ["BTC", "ETH", "SOL"]
mids_payload = post("https://api.hyperliquid.xyz/info", {"type": "allMids"})
meta_payload = post("https://api.hyperliquid.xyz/info", {"type": "metaAndAssetCtxs"})
candles = {}
for coin in coins:
    start = int(time.time() * 1000) - 180 * 4 * 3600 * 1000
    rows = post("https://api.hyperliquid.xyz/info", {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": "4h", "startTime": start}})
    candles[coin] = hl_analysis.parse_candles(rows)

mids = hl_analysis.parse_all_mids(mids_payload)
meta_rows = hl_analysis.parse_meta_and_ctxs(meta_payload, coins)
prices = {c: mids[c] for c in coins if c in mids}
print("PRICES:", prices)
print("META BTC:", {k: v for k, v in meta_rows.get("BTC", {}).items()})
print("CANDLES BTC:", len(candles.get("BTC", [])))

fg_raw = json.loads(get("https://api.alternative.me/fng/?limit=8"))
fg = hl_analysis.parse_fear_greed(fg_raw)
print("F&G:", fg)

feeds = {}
for source, url in hl_analysis.NEWS_FEEDS.items():
    try:
        feeds[source] = get(url)
    except Exception as exc:  # noqa: BLE001
        print("feed fail:", source, exc)
news = hl_analysis.merge_news({s: hl_analysis.parse_rss_feed(x, s) for s, x in feeds.items()})
ns = hl_analysis.summarize_news_sentiment(news)
mentions = hl_analysis.coin_mention_counts(news, coins)
print("NEWS:", len(news), "items;", ns)
print("MENTIONS:", mentions)

params = {"universe": coins, "max_leverage": 5, "max_margin_fraction": 0.10,
          "max_open_positions": 3, "daily_loss_limit_pct": 5.0, "risk_profile": "balanced"}
account = hl_analysis.new_account(1000.0, "2026-11-14T12:00:00+00:00")
rows = hl_analysis.build_market_rows(prices, meta_rows, candles)

# simulated model decision: long BTC 3x, short none, one invalid action
actions = [
    {"action": "open", "coin": "BTC", "direction": "long", "leverage": 3,
     "margin_fraction": 0.05, "stop_loss": prices["BTC"] * 0.97,
     "take_profit": prices["BTC"] * 1.06, "rationale": "droogrun"},
    {"action": "open", "coin": "ETH", "direction": "long", "leverage": 9,
     "margin_fraction": 0.05, "rationale": "te veel leverage, moet geweigerd worden"},
]
events = hl_analysis.apply_actions(account, actions, prices, "2026-11-14T12:00:00+00:00", params, meta_rows)
print("EVENTS:", json.dumps(events, indent=1)[:600])

prompt = hl_analysis.build_cycle_prompt(rows, candles, account, prices, params, fg, news, ns, mentions)
print("PROMPT LEN:", len(prompt))
assert "MARKTDATA" in prompt and "ACCOUNTSTATUS" in prompt

forced = hl_analysis.process_forced_exits(account, prices, "2026-11-14T12:01:00+00:00")
print("FORCED:", forced)
hl_analysis.update_circuit_breaker(account, prices, 5.0)

decision = {"market_view": "droogrun", "rationale": "valideren van de rapportrenderer.",
            "key_risks": ["liquidatierisico bij leverage"], "actions": []}
report = hl_analysis.render_cycle_report(
    "2026-11-14T12:00:00+00:00", rows, account, prices, params, fg, news, ns,
    decision, events, "mistral-medium-latest")
out = Path(__file__).parent / "dryrun_report.md"
out.write_text(report)
print("REPORT LEN:", len(report), "->", out)
for section in ("## Markt", "## Sentiment", "## Nieuws", "## Account (paper)",
                "## Beslissingen", "OPEN BTC long", "GEWEIGERD"):
    assert section in report, section
print("ALL DRY-RUN CHECKS PASSED")
