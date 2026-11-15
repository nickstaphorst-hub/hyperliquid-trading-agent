# Hyperliquid Trading Agent

Duurzame **paper-trading-agent** voor Hyperliquid perpetuals, gebouwd als stateful
[Mistral Workflow](https://docs.mistral.ai). De agent draait als één langlopende
executie op een managed deployment, houdt een papieren (paper) account bij in
workflow-state en doorloopt elke cyclus: marktdata ophalen → posities beoordelen →
model-beslissing → deterministische risico-engine → rapport.

> ⚠️ **Alleen paper trading.** Live orders plaatsen is bewust niet geïmplementeerd.
> Er bestaan nergens exchange-credentials in deze code. Geen financieel advies.

## Hoe het werkt

Elke cyclus: standaard uitgelijnd op een vast muurklokslot (elk uur op `:05` na het uur, via `align_to_hour=True` + `cycle_minute`), zodat het schema niet afhangt van de cyclusduur en de feeds (en bij de 4-uursgrenzen de net gesloten 4h-candles) freshest zijn. Met `align_to_hour=False` geldt de vaste slaap `cycle_seconds`:

1. **Marktdata** (publiek, geen API-key nodig):
   - Hyperliquid `allMids` + `metaAndAssetCtxs` (prijs, funding, open interest, max leverage per coin) en `candleSnapshot` (4h candles) voor het universum (standaard BTC/ETH/SOL).
   - Fear & Greed Index (alternative.me) en RSS-nieuws (Cointelegraph, Decrypt, CoinDesk) met deterministische sentiment-scores, net als de crypto-market-scan-agent.
2. **Forced exits**: open posities worden eerst gecheckt tegen (gesimuleerde) liquidatieprijs, stop-loss en take-profit. **Guard-mode:** zolang er posities open staan, checkt een lichte loop (geen model-aanroep, alleen een allMids-request) elke `guard_seconds` (standaard 900 s = 15 min, 0 = uit) de SL/TP/liquidatie-triggers, dus tussen de uurscycli door.
3. **Model-beslissing**: een Mistral-model krijgt markt, sentiment, nieuws én de volledige accountstatus en stelt maximaal N acties voor (`open`/`close`/`none` met leverage, margin-fraction, SL/TP en rationale).
4. **Risico-engine** (deterministisch, in `hl_analysis.py`) valideert élke actie en simuleert de fill tegen de live mid-prijs. Voorstellen buiten de regels worden geweigerd en als zodanig gerapporteerd.

### Risicoregels (engine-afgedwongen)

| Regel | Standaard |
|---|---|
| Max leverage per positie | 5x (of lager als de coin minder toestaat) |
| Max marge per trade (fraction van equity) | 10% |
| Max open posities (1 per coin) | 3 |
| Dagverlieslimiet (circuit breaker, blokkeert nieuwe entries) | -5% per UTC-dag |
| Taker fee per open/close | 0,045% van notional |
| Liquidatie | benadering: entry ∓ 90% van 1/leverage-marge |

Funding wordt in de simulatie niet meegenomen; dat staat in het rapport vermeld.

### Leren van trades (in-context)

De agent leert van zijn eigen (paper-)historie:

- Elke cyclus berekent de engine deterministisch **prestatie-statistieken** uit het handelsjournaal: win-rate, gemiddelde winst/verlies, resultaat per richting en per coin, exit-redenen (model-close, stop-loss, take-profit, liquidatie) en verliesreeksen.
- Deze statistieken, een compact **journaal van de laatste gesloten trades** (inclusief de oorspronkelijke entry-rationale) en automatisch afgeleide **lessen** (bijv. verliesreeks, verliesgevende richting, liquidatie, te hoge leverage bij verliezers) komen elke cyclus in de prompt als blok `PRESTATIEVERLEDEN`.
- Het model krijgt de instructie om te leren van winnaars en verliezers; de lessen zijn context, geen nieuwe limieten: de harde risicoregels blijven onveranderd engine-afgedwongen.
- Het cyclusrapport bevat een sectie **Prestatieverleden** met dezelfde cijfers en lessen.

Zodra er genoeg trades verzameld zijn, kan de gelogde trade-historie (via `get_account`) dienen als dataset voor fine-tuning van het model; dat is bewust nog niet geautomatiseerd.

## Structuur

```
apps/worker/src/worker/workflows/
├── hl_analysis.py        # pure core: parsers, indicators, sentiment, paper-engine, prompt, rapport
└── hyperliquid_agent.py  # stateful workflow: activities (HTTP), loop, queries, updates
apps/worker/tests/
└── test_hl_analysis.py   # 56 stdlib-only unittests
```

Queries op de draaiende agent: `get_account` (balans, posities, trades, halt-status) en
`get_last_cycle` (laatste cyclusrapport in Markdown). Updates: `reset_account`
(paper-account opnieuw initialiseren) en `set_entries_halted` (handmatig entries blokkeren).

## Lokaal testen

```bash
cd apps/worker/tests
python3 -m unittest test_hl_analysis.py        # 56 tests, geen netwerk nodig
python3 dryrun_live.py                         # live dry-run tegen Hyperliquid/F&G/RSS (geen model)
```

## Inputs (executie)

`universe` (BTC/ETH/SOL), `interval` (4h), `candles_limit` (180), `model`
(mistral-medium-latest), `risk_profile` (balanced), `max_leverage` (5),
`max_margin_fraction` (0.10), `max_open_positions` (3), `daily_loss_limit_pct` (5),
`start_balance_usd` (1000), `cycle_seconds` (3600), `align_to_hour`/`cycle_minute` (elk uur op :05), `guard_seconds` (900, 0 = uit), `trading_mode` (alleen "paper").

## Deployment

Managed deployment vanaf `main` met `deploy/docker/Dockerfile.worker`
(build-directory `.`), daarna de workflow één keer starten via `execute_workflow`;
de loop draait daarna vanzelf door. Zie de canvas-rapportages in de chat voor de
laatste status.

**Run-lifetime:** de workflow-definitie heeft `execution_timeout=timedelta(days=365)`
opt-in; de platform-default is 1 uur, wat de oneindige loop anders halverwege de
tweede cyclus doodt. Na ~365 dagen (of eerder bij een veel snellere cadence) moet
de executie opnieuw gestart worden; de execution-history-limiet (~51.200 events)
ligt bij de 4-urige cadence ruim boven die horizon. Het paper-account begint na
een herstart weer op de startbalance; trade-historie herleeft binnen één executie.
