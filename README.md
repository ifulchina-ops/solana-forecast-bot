# Monte Carlo rank forecaster — pump.fun top 5

Entry for the make.fun bounty "Predict the top 5 new Solana coins by market
cap, 24 hours out" (`pumpfun-top5-forecast-01`).

`main.py` (Python 3.13, standard library only, no network) reads the house's
10-minute market-cap snapshots of pump.fun coins from `/input` and writes one
forecast to `/output/forecast.json`: per-place probabilities for the 5 largest
coins at each of the seven answer reads, the top 5 in order, and per-place
market-cap bin probabilities. It is fully deterministic (fixed seed): two runs
on the same history write byte-identical output.

## Method

- **Candidates**: the last mark's ranked coins (up to 200).
- **Per-coin model**: decay-weighted log-cap trend (half-life 30 min, shrunk
  toward the cross-sectional median drift), per-coin volatility (shrunk toward
  the median), a crash-jump hazard (higher for coins still on their bonding
  curve), and a graduation rate for curve coins (their cap is bounded until
  they graduate). Drift decays with horizon (half-life 1.5 h): momentum does
  not persist 13 hours. A small secular fade (-0.05 log10/hour) corrects the
  survivorship bias in observed drifts. Simulated caps pass through a soft
  upper cap (knee 10,000 SOL, asymptote ~316,000 SOL) so long-horizon
  diffusion cannot put mass on absurd caps.
- **Phantom newcomers**: coins born after the cut cannot be named, but they
  take places and push the top-5 cap levels up. Each simulation draws a
  Poisson number of phantoms at a rate estimated from the history itself
  (newcomers entering the top 5 per hour, last-half transitions, shrunk
  toward a 6/hour prior), with caps near the current place-1 level. Phantoms
  compete in the ranking, which shrinks known coins' place probabilities
  honestly and widens the size distributions; their share stays in each
  place's unnamed mass.
- **Aggregation**: a fixed-seed Monte Carlo (~2,500 draws per coin per read)
  ranks the candidates at each of the seven answer reads; place frequencies
  and place-cap histograms are averaged over reads (each read is scored on
  its own marginals). A coin is named at a place only when its probability
  beats what it would receive as part of the unnamed share. Probabilities
  are rounded down to 6 decimals; every place's sum stays at most 1.

## Run

```
python3 main.py
```

Outside the grader's container, `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR`
stand in for `/input` and `/output`:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runtime on the kit's practice input: ~5 s (limit 300 s). Output is ~9 KB
(limit 1,048,576 bytes).

## Honest limitations

- The practice history covers 1 hour; the final forecasts 7–13 hours out.
  The long-horizon behaviour (drift decay, crash hazard, phantom rate) is
  designed from the mechanism, not measured: it cannot be validated on the
  practice data, and the practice backtest below measures short-horizon
  skill only.
- No forecast can name a coin born after the cut. At the practice's last
  read, 3 of the top 5 were unnameable from the history; over 7–13 hours
  that share is likely larger. The model leaves most of each place unnamed
  at long horizons, which is honest but caps the achievable score.
- The house AI model forecast's score is unknown; the prize bar is the
  higher of the house baseline's and the house model's final scores.

## Validation (local, on the kit's practice history)

- A Python port of the kit's scorer (`grader/forecast/ranked.ts`)
  reproduces the rules' worked example exactly (-38.445453 / -34.255798 /
  mean -36.350626, agreement 13 / 15).
- Within-history backtest (train on early marks, score against later marks
  as pseudo-answer reads; same scorer): the model beats the kit's starter
  (house baseline) on all 4 slices, by +0.52 / +8.01 / +5.95 / +7.55 nats,
  mean **+5.51 nats** (model -34.76 vs baseline -40.27).
- Determinism: two runs on the same input produce byte-identical output.
- Long-horizon smoke test (reads 7–13 h out): place sums fall to
  0.58 / 0.31 / 0.19 / 0.15 / 0.11 (phantoms take most places), place-1
  sizes centre near 10^4.5 SOL, no mass on the ≥10^6 SOL bin.
