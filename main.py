#!/usr/bin/env python3
"""Monte Carlo rank forecaster for make.fun pumpfun-top5-forecast-01.

Reads /input/history.ndjson (10-minute market-cap snapshots of pump.fun coins
launched since the opening) and /input/task.json, writes /output/forecast.json:
for each of the 5 places, place probabilities per coin, the top 5 in order,
and per-place market-cap bin probabilities.

Method (deterministic, stdlib only, no network, fixed seed):
  - Each candidate coin (the last mark's ranked rows) gets a decay-weighted
    log-cap trend (shrunk toward the cross-sectional median drift), a
    per-coin volatility (shrunk toward the median), a crash-jump hazard
    (pool coins lower, curve coins higher), and a graduation rate for coins
    still on their bonding curve (their cap is bounded until they graduate).
  - Drift decays with horizon (half-life 1.5 h): momentum does not persist
    13 hours. Volatility grows with sqrt(horizon).
  - A fixed-seed Monte Carlo draws each coin's log market cap at each of the
    seven answer reads, ranks the candidates, and counts place frequencies
    and place-cap histograms. Because each read is scored on its own
    marginals, the per-read distributions are averaged into one forecast.
  - Coins born after the cut cannot be named, but they take places and push
    the top-5 cap levels up, so each sim also draws "phantom" newcomers
    (a Poisson count at a rate estimated from the history, caps near the
    current place-1 level). Phantoms compete in the ranking, which shrinks
    known coins' place probabilities honestly and widens the size
    distributions; their share stays in each place's unnamed mass, which
    the scorer spreads over the chart's unnamed coins.
  - A coin is only named at a place when its probability beats what it
    would receive as part of the unnamed share.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
MICRO = 1_000_000

# --- model constants (fixed in advance, not tuned on the practice answer) ---
DRIFT_HALF_LIFE_H = 1.5      # momentum decay half-life, hours
DRIFT_CLAMP = (-0.6, 0.9)    # per-hour log10 drift clamp
DRIFT_SHRINK_K = 2.0         # shrinkage weight toward median drift
VOL_FLOOR_H = 0.25           # per-hour log10 volatility floor
VOL_CAP_H = 0.9              # per-hour log10 volatility cap
VOL_SHRINK_K = 2.0           # shrinkage weight toward median vol
FADE_PER_H = -0.05           # secular bleed of every coin, per hour (not
                             # decayed: the median pump.fun coin bleeds out;
                             # observed drifts are survivorship-biased up)
# Soft upper cap on simulated log-caps: diffusion over 13 h would otherwise
# put mass on absurd caps (a coin cannot reach 10^6 SOL in a day). The cap
# only bends values above the knee (10,000 SOL) toward an asymptote of
# ~316,000 SOL and leaves the bulk of the distribution untouched.
SOFT_CAP_KNEE = 4.0
SOFT_CAP_SCALE = 1.5
HAZARD_POOL_H = 0.15         # crash jumps per hour, graduated coins
HAZARD_CURVE_H = 0.40        # crash jumps per hour, curve coins
JUMP_LO, JUMP_HI = 0.8, 2.0  # crash jump size, log10
CURVE_CAP = 2.70             # log10 SOL cap of a coin still on its curve
GRAD_COMPLETE_H = 0.6        # graduation rate/hr for a completed curve
GRAD_RISING_H = 0.25         # graduation rate/hr for a rising curve
GRAD_OTHER_H = 0.08          # graduation rate/hr otherwise
# Phantom newcomers: coins born after the cut cannot be named, but they take
# places and push the top-5 cap levels up. Their arrival rate and cap
# distribution are estimated from the history itself (last-half transitions,
# so the opening ramp-up does not dominate).
PHANTOM_RATE_FLOOR = 3.0     # newcomers into the top 5, per hour
PHANTOM_RATE_CAP = 12.0
PHANTOM_DELTA_DEFAULT = -0.1  # newcomer cap vs the last place-1 cap, log10
PHANTOM_SD_DEFAULT = 0.35
PHANTOM_MAX = 25             # phantoms drawn per sim (bounds the max-of-N rise)
MAX_ROWS_PER_PLACE = 60      # rows named per place (limit is 100)
MIN_NAMED_P = 0.004          # smallest named place probability
MIN_BIN_P = 0.0005           # smallest named bin probability
LOG_FLOOR = 0.0              # log10 SOL floor for a simulated cap
TARGET_DRAWS = 2_500_000     # simulation budget (coin x read x sim)
MIN_SIMS, MAX_SIMS = 800, 4000


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def logcap_sol(market_cap_lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    return math.log10(int(market_cap_lamports)) - 9


def bin_of(r, edges):
    """Bin of value r under strictly ascending edges (how many edges <= r)."""
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) >> 1
        if edges[mid] <= r:
            lo = mid + 1
        else:
            hi = mid
    return lo


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def soft_cap(y):
    """Bend log-caps above the knee toward SOFT_CAP_KNEE + SOFT_CAP_SCALE."""
    if y <= SOFT_CAP_KNEE:
        return y
    d = y - SOFT_CAP_KNEE
    return SOFT_CAP_KNEE + d / (1.0 + d / SOFT_CAP_SCALE)


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    output_name = task.get("output", "forecast.json")
    reads = [parse_time(r) for r in task["reads"]]

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    marks.sort(key=lambda m: parse_time(m["at"]))

    t_last = parse_time(marks[-1]["at"])
    horizons = [max((r - t_last) / 3600.0, 0.05) for r in reads]

    # --- per-coin series: mint -> list of (t_hours_since_first, logcap, venue, complete) ---
    t0 = parse_time(marks[0]["at"])
    series = {}
    last_row = {}
    for m in marks:
        t = (parse_time(m["at"]) - t0) / 3600.0
        for row in m["rows"]:
            mint = row["mint"]
            y = logcap_sol(row["market_cap_lamports"])
            series.setdefault(mint, []).append(
                (t, y, row.get("venue", "curve"), bool(row.get("complete")))
            )
            last_row[mint] = row

    last_mark = marks[-1]["rows"]
    last_mark.sort(key=lambda r: r["rank"])
    candidates = [r["mint"] for r in last_mark]
    if len(candidates) < places:
        raise SystemExit("the last mark ranks fewer coins than the places")

    # --- cross-sectional priors from the candidates' own histories ---
    slopes, vols = [], []
    for mint in candidates:
        pts = series[mint]
        if len(pts) >= 2:
            (t1, y1, _, _), (t2, y2, _, _) = pts[0], pts[-1]
            if t2 > t1:
                slopes.append((y2 - y1) / (t2 - t1))
        if len(pts) >= 3:
            diffs = [
                (pts[i + 1][1] - pts[i][1]) / math.sqrt(max(pts[i + 1][0] - pts[i][0], 1e-9))
                for i in range(len(pts) - 1)
            ]
            mu = sum(diffs) / len(diffs)
            var = sum((d - mu) ** 2 for d in diffs) / len(diffs)
            vols.append(math.sqrt(var))
    median_slope = median(slopes) if slopes else 0.0
    median_vol = median(vols) if vols else 0.30
    median_vol = min(max(median_vol, VOL_FLOOR_H), VOL_CAP_H)

    # --- per-coin parameters ---
    params = {}
    for mint in candidates:
        pts = series[mint]
        row = last_row[mint]
        y_last = pts[-1][1]
        venue = row.get("venue", "curve")
        complete = bool(row.get("complete"))
        # decay-weighted linear trend (half-life 30 min), shrunk to the median
        t_end = pts[-1][0]
        sw = sx = sy = sxx = sxy = 0.0
        for (t, y, _, _) in pts:
            w = 0.5 ** ((t_end - t) / 0.5)
            sw += w
            sx += w * t
            sy += w * y
            sxx += w * t * t
            sxy += w * t * y
        denom = sw * sxx - sx * sx
        if len(pts) >= 2 and abs(denom) > 1e-12:
            slope = (sw * sxy - sx * sy) / denom
        else:
            slope = 0.0
        drift = (sw * slope + DRIFT_SHRINK_K * median_slope) / (sw + DRIFT_SHRINK_K)
        drift = min(max(drift, DRIFT_CLAMP[0]), DRIFT_CLAMP[1])
        # volatility from residuals around the fitted line, shrunk to the median
        if len(pts) >= 3:
            intercept = (sy - slope * sx) / sw
            resid = [y - (intercept + slope * t) for (t, y, _, _) in pts]
            diffs = [resid[i + 1] - resid[i] for i in range(len(resid) - 1)]
            gaps = [pts[i + 1][0] - pts[i][0] for i in range(len(pts) - 1)]
            var = sum(d * d / max(g, 1e-9) for d, g in zip(diffs, gaps)) / max(len(diffs), 1)
            vol = math.sqrt(max(var, 0.0))
            vol = (len(diffs) * vol + VOL_SHRINK_K * median_vol) / (len(diffs) + VOL_SHRINK_K)
        else:
            vol = median_vol
        vol = min(max(vol, VOL_FLOOR_H), VOL_CAP_H)
        # crash hazard and graduation rate
        if venue == "pool":
            hazard = HAZARD_POOL_H
            grad = float("inf")  # already graduated: no cap
        else:
            hazard = HAZARD_CURVE_H
            if complete:
                grad = GRAD_COMPLETE_H
            elif drift > 0.1:
                grad = GRAD_RISING_H
            else:
                grad = GRAD_OTHER_H
        if drift > 0.2:
            hazard *= 0.7
        elif drift < -0.2:
            hazard *= 1.3
        params[mint] = (y_last, drift, vol, hazard, grad)

    # --- phantom newcomers: arrival rate and cap offsets from the history ---
    # A newcomer is a coin that enters a mark's top 5 without having been in
    # the previous mark's top 20. Use the last half of the transitions so the
    # opening ramp-up (when every coin is new) does not dominate.
    transitions = []
    for i in range(1, len(marks)):
        prev = sorted(marks[i - 1]["rows"], key=lambda r: r["rank"])
        cur = sorted(marks[i]["rows"], key=lambda r: r["rank"])
        if not prev or not cur:
            continue
        dt_h = (parse_time(marks[i]["at"]) - parse_time(marks[i - 1]["at"])) / 3600.0
        if dt_h <= 0:
            continue
        prev_top20 = {r["mint"] for r in prev[:20]}
        prev_p1 = logcap_sol(prev[0]["market_cap_lamports"])
        new = [r for r in cur[:5] if r["mint"] not in prev_top20]
        transitions.append((dt_h, len(new), [logcap_sol(r["market_cap_lamports"]) - prev_p1 for r in new]))
    half = transitions[len(transitions) // 2:] if len(transitions) >= 2 else transitions
    hours = sum(t[0] for t in half)
    count = sum(t[1] for t in half)
    # shrink toward the prior (6/hr) so a single ramp-up transition cannot
    # dominate the estimate
    phantom_rate = min(max((count + 3.0) / (hours + 0.5), PHANTOM_RATE_FLOOR), PHANTOM_RATE_CAP)
    offsets = [d for t in half for d in t[2]]
    if len(offsets) < 3:
        offsets = [d for t in transitions for d in t[2]]
    if len(offsets) >= 3:
        raw_delta = sum(offsets) / len(offsets)
        # shrink toward the prior (-0.1) and clamp: newcomers that reach the
        # top 5 arrive near the top; a ramp-up history must not drag this low
        phantom_delta = (len(offsets) * raw_delta + 4.0 * PHANTOM_DELTA_DEFAULT) / (len(offsets) + 4.0)
        var = sum((d - raw_delta) ** 2 for d in offsets) / len(offsets)
        phantom_sd = min(max(math.sqrt(var), 0.30), 0.6)
        phantom_delta = min(max(phantom_delta, -0.25), 0.15)
    else:
        phantom_delta = PHANTOM_DELTA_DEFAULT
        phantom_sd = PHANTOM_SD_DEFAULT
    p1_last = logcap_sol(last_mark[0]["market_cap_lamports"])

    # --- Monte Carlo over the reads ---
    # Each sim draws the known candidates' log-caps at the read plus a
    # Poisson number of phantom newcomers (coins born after the cut, with
    # caps near the current place-1 level). Phantoms take places in the
    # ranking (so known coins' place probabilities shrink honestly) and set
    # the place-cap levels for the sizes; they are never named, so their
    # share stays in each place's unnamed mass, which the scorer spreads
    # over the chart's unnamed coins.
    n_cand = len(candidates)
    n_reads = len(reads)
    sims = int(TARGET_DRAWS // max(n_cand * n_reads, 1))
    sims = min(max(sims, MIN_SIMS), MAX_SIMS)
    rng = random.Random(SEED)
    decay_scale = DRIFT_HALF_LIFE_H / math.log(2)

    place_counts = {m: [0] * places for m in candidates}
    size_hist = [[0] * (len(edges) + 1) for _ in range(places)]
    total_draws = sims * n_reads

    for h in horizons:
        drift_factor = decay_scale * (1.0 - 0.5 ** (h / DRIFT_HALF_LIFE_H))
        sqrt_h = math.sqrt(h)
        lam_phantom = phantom_rate * h
        phantom_mean_h = p1_last + phantom_delta + 0.05 * sqrt_h
        phantom_sd_h = phantom_sd + 0.05 * sqrt_h
        for _ in range(sims):
            scored = []
            for mint in candidates:
                y_last, drift, vol, hazard, grad = params[mint]
                y = (
                    y_last
                    + drift * drift_factor
                    + FADE_PER_H * h
                    + vol * sqrt_h * rng.gauss(0.0, 1.0)
                )
                # crash jumps: Poisson(hazard * h), each -U(0.8, 2.0)
                lam = hazard * h
                if lam > 0:
                    # Knuth's algorithm (lam is small: <= ~5)
                    l = math.exp(-lam)
                    k = 0
                    p = 1.0
                    while True:
                        p *= rng.random()
                        if p <= l:
                            break
                        k += 1
                    for _ in range(k):
                        y -= rng.uniform(JUMP_LO, JUMP_HI)
                # bonding-curve cap until graduation
                if grad != float("inf"):
                    t_grad = rng.expovariate(grad) if grad > 0 else float("inf")
                    if t_grad > h and y > CURVE_CAP:
                        y = CURVE_CAP
                y = soft_cap(y)
                if y < LOG_FLOOR:
                    y = LOG_FLOOR
                scored.append((y, mint))
            # phantom newcomers (unnamed: mint None)
            if lam_phantom > 0:
                l = math.exp(-lam_phantom)
                k = 0
                p = 1.0
                while True:
                    p *= rng.random()
                    if p <= l:
                        break
                    k += 1
                for _ in range(min(k, PHANTOM_MAX)):
                    y = phantom_mean_h + phantom_sd_h * rng.gauss(0.0, 1.0)
                    scored.append((max(soft_cap(y), LOG_FLOOR), None))
            scored.sort(key=lambda ym: (-ym[0], ym[1] or ""))
            for k in range(places):
                y_k, mint_k = scored[k]
                if mint_k is not None:
                    place_counts[mint_k][k] += 1
                size_hist[k][bin_of(y_k, edges)] += 1

    # --- place probabilities (phantoms already shrank them honestly) ---
    probs = {}
    for mint in candidates:
        probs[mint] = [place_counts[mint][k] / total_draws for k in range(places)]

    # name a coin at a place only when its p beats the unnamed share it would
    # otherwise receive (greedy, in descending p order)
    probabilities = []
    for k in range(places):
        ranked = sorted(
            ((probs[m][k], m) for m in candidates), key=lambda pm: (-pm[0], pm[1])
        )
        named_sum = 0.0
        n_named = 0
        for p, mint in ranked:
            if n_named >= MAX_ROWS_PER_PLACE or p < MIN_NAMED_P:
                break
            unnamed_each = (1.0 - named_sum) / max(100 - n_named, 1)
            if p <= unnamed_each:
                break
            probabilities.append({"mint": mint, "place": k + 1, "p": p})
            named_sum += p
            n_named += 1

    # --- top5 by expected rank (a coin outside the top 5 counts as rank 6),
    # then by P(place 1), then by last-mark rank ---
    last_rank = {r["mint"]: r["rank"] for r in last_mark}
    expect = {}
    for mint in candidates:
        total = sum(probs[mint])
        expect[mint] = sum((k + 1) * probs[mint][k] for k in range(places)) + 6.0 * (
            1.0 - total
        )
    top5 = sorted(
        candidates,
        key=lambda m: (expect[m], -probs[m][0], last_rank.get(m, 999)),
    )[:places]

    # --- sizes: simulated place-cap histograms, rounded down, tail left unnamed ---
    sizes = []
    for k in range(places):
        for b in range(len(edges) + 1):
            p = size_hist[k][b] / total_draws
            if p >= MIN_BIN_P:
                sizes.append({"place": k + 1, "bin": b, "p": p})

    # --- round down to 6 decimals; drop zeros; keep every place's sum <= 1 ---
    def finalize(rows, key):
        out = []
        by_place = {}
        for row in rows:
            u = math.floor(row["p"] * MICRO)
            if u <= 0:
                continue
            r = dict(row)
            r["p"] = u / MICRO
            out.append(r)
            by_place.setdefault(row["place"], []).append(r)
        for place_rows in by_place.values():
            total = sum(r["p"] for r in place_rows)
            if total > 1.0:
                place_rows.sort(key=lambda r: -r["p"])
                excess = total - 1.0
                place_rows[0]["p"] = max(place_rows[0]["p"] - excess, 0.0)
                if place_rows[0]["p"] <= 0:
                    out.remove(place_rows[0])
        return out

    probabilities = finalize(probabilities, "probabilities")
    sizes = finalize(sizes, "sizes")
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))

    doc = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
