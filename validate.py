#!/usr/bin/env python3
"""Validation harness for main.py (not part of the entry's run).

Scores the forecaster with the competition's own metric - mean
pinball loss on the logit scale - in three regimes:

  1. holdout: each practice case is cut earlier and the model must
     forecast the case's own last read (the practice histories are
     one unchanging page, so this regime is degenerate: it checks
     the model stays honest and centred when nothing moves);
  2. refresh shock: the documented first Exophase refresh applied
     to the practice page (players 638 -> 3,868, earners fixed, so
     rates dilute) - the regime the real practice grading used,
     where the house baseline scored about 1.51;
  3. growth week: a synthetic 60-hour history of saturating logit
     growth behind step refreshes, forecast 4.5 days ahead - the
     regime the final case lives in.

Comparisons: a flat baseline (last rate, small fixed spread) and the
house baseline formula published in RULES.txt (last rate, spread
0.5 * sqrt(horizon days) on the logit scale).
"""

import copy
import json
import math
import random
import sys
from statistics import NormalDist

sys.path.insert(0, ".")
from main import forecast_case, logit_of_percent, percent_of_logit, parse_instant

Z = NormalDist()
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
KIT = "/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-dl/" \
      "gta6-trophy-rarity-forecast-01-kit/input"


def pinball_mean(rows, answers):
    total = 0.0
    count = 0
    by_id = {row["id"]: row["q"] for row in rows}
    for tid, y in answers.items():
        ly = logit_of_percent(y)
        for tau, q in zip(LEVELS, by_id[tid]):
            d = ly - logit_of_percent(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


def flat_rows(case, sigma=0.15):
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        last = None
        for read in case["history"]:
            if read["rates"][index] is not None:
                last = read["rates"][index]
        centre = logit_of_percent(last)
        rows.append({"id": trophy["id"], "q": [
            round(percent_of_logit(centre + Z.inv_cdf(t) * sigma), 4)
            for t in LEVELS]})
    return rows


def house_rows(case):
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma = 0.5 * math.sqrt(horizon_days)
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        last = None
        for read in case["history"]:
            if read["rates"][index] is not None:
                last = read["rates"][index]
        centre = logit_of_percent(last)
        rows.append({"id": trophy["id"], "q": [
            round(percent_of_logit(centre + Z.inv_cdf(t) * sigma), 4)
            for t in LEVELS]})
    return rows


def last_answers(case):
    answers = {}
    for index, trophy in enumerate(case["trophies"]):
        answers[trophy["id"]] = case["history"][-1]["rates"][index]
    return answers


def holdout(case, keep_fraction=0.6):
    cut = max(2, int(len(case["history"]) * keep_fraction))
    trimmed = copy.deepcopy(case)
    trimmed["history"] = case["history"][:cut]
    trimmed["target"] = case["history"][-1]["at"]
    trimmed["at"] = case["history"][cut - 1]["at"]
    answers = last_answers(case)
    return trimmed, answers


def refresh_answers(case, players_before=638, players_after=3868):
    answers = {}
    for index, trophy in enumerate(case["trophies"]):
        rate = case["history"][-1]["rates"][index]
        if rate >= 100.0:
            answers[trophy["id"]] = 100.0
        else:
            answers[trophy["id"]] = rate * players_before / players_after
    return answers


def synthetic_case(seed=20261119):
    rng = random.Random(seed)
    tiers = ["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * 46
    trophies = []
    truth = {}
    series = []
    for i, tier in enumerate(tiers):
        tid = i + 1
        trophies.append({"id": tid, "name": "T%02d" % tid,
                         "description": "", "tier": tier,
                         "points": 15, "secret": False})
        start = rng.uniform(0.05, 6.0)
        ceiling_pct = min(97.0, start * rng.uniform(2.5, 14.0))
        speed = rng.uniform(0.004, 0.03)          # per hour
        series.append((logit_of_percent(start), logit_of_percent(ceiling_pct),
                       speed))
    history = []
    refresh_every = 5.0                            # hours
    horizon_end = 60.0
    t = 0.0
    page = None
    while t <= horizon_end:
        if page is None or t % refresh_every < 1e-9 or \
                abs(t / refresh_every - round(t / refresh_every)) < 1e-9:
            page = []
            for l0, ceil, k in series:
                value = ceil + (l0 - ceil) * math.exp(-k * t)
                value += rng.gauss(0, 0.05)
                page.append(round(percent_of_logit(value), 2))
        history.append({"slot": "", "at": _iso(t), "players_tracked": 5000,
                        "trophies_earned": 0, "rates": list(page)})
        t += 0.5
    target_t = horizon_end + 108.0
    for i, (l0, ceil, k) in enumerate(series):
        value = ceil + (l0 - ceil) * math.exp(-k * target_t)
        truth[i + 1] = percent_of_logit(value)
    case = {"id": "syn", "at": _iso(horizon_end),
            "horizon_minutes": int((target_t - horizon_end) * 60),
            "target": _iso(target_t), "levels": LEVELS,
            "game": {}, "trophies": trophies, "history": history,
            "notes": ""}
    return case, truth


def _iso(hours):
    base = parse_instant("2026-11-19T00:00:00Z")
    from datetime import datetime, timezone
    moment = datetime.fromtimestamp(base + hours * 3600, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    for cid in ["p00", "p01"]:
        with open(KIT + "/" + cid + ".in") as handle:
            case = json.load(handle)
        model = forecast_case(case)["trophies"]

        trimmed, answers = holdout(case)
        print(f"[{cid}] holdout (cut at {len(trimmed['history'])} of "
              f"{len(case['history'])} reads, target = last read)")
        print(f"    model {pinball_mean(forecast_case(trimmed)['trophies'], answers):.4f}"
              f" | flat {pinball_mean(flat_rows(trimmed), answers):.4f}"
              f" | house {pinball_mean(house_rows(trimmed), answers):.4f}")

        shocked = refresh_answers(case)
        print(f"[{cid}] refresh shock (players 638 -> 3868, earners fixed)")
        print(f"    model {pinball_mean(model, shocked):.4f}"
              f" | flat {pinball_mean(flat_rows(case), shocked):.4f}"
              f" | house {pinball_mean(house_rows(case), shocked):.4f}")

    case, truth = synthetic_case()
    print("[syn] growth week (60 h history, target +108 h)")
    print(f"    model {pinball_mean(forecast_case(case)['trophies'], truth):.4f}"
          f" | flat {pinball_mean(flat_rows(case), truth):.4f}"
          f" | house {pinball_mean(house_rows(case), truth):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
