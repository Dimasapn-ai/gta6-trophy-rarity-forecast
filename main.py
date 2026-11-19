#!/usr/bin/env python3
"""Launch-Week Trophy Forecaster.

Reads one or more make.fun GTA VI trophy cases and, for every trophy,
forecasts five quantiles of its Exophase earn rate (percent) at the
case's target instant.  The model works on the logit scale, where the
pinball loss is scored:

  * each trophy's observed logit rate is fitted with a saturating
    curve anchored on its last observed rate, so the forecast bends
    toward a per-trophy ceiling (or floor) instead of running away;
  * trophies with little usable history borrow the growth behaviour
    of their tier (platinum / gold / silver / bronze) through
    shrinkage of the curve's headroom and speed;
  * the quantile spread combines the fit's residual scatter, a
    random-walk term that grows with the horizon beyond the data,
    and a floor that prices in Exophase refreshing the page between
    the last read and the target.

Standard library only, no network, no randomness: the same case file
always yields the same forecast file, byte for byte.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist, median

RATE_LO = 0.01          # the scorer clamps rates into [0.01, 99.99]
RATE_HI = 99.99
LOGIT_CAP = 12.5        # keep forecast logits finite and sane
Z = NormalDist()

# Spread ingredients (logit units).  The first refresh of launch night
# multiplied the tracked-player count sixfold overnight; a forecast
# that pretends the page cannot jump between the last read and the
# target is the classic way to lose at this game, so the jump floor
# is charged at every horizon.
JUMP_SD = 0.55
WALK_SD_PER_SQRT_DAY = 0.42
RESID_FLOOR_VAR = 0.04 ** 2
SIGMA_MIN = 0.24
SIGMA_MAX = 3.2


def parse_instant(text):
    """Parse an ISO 8601 UTC timestamp into epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def logit_of_percent(rate):
    clipped = min(max(rate, RATE_LO), RATE_HI)
    p = clipped / 100.0
    return math.log(p / (1.0 - p))


def percent_of_logit(value):
    if value >= 0:
        p = 1.0 / (1.0 + math.exp(-value))
    else:
        e = math.exp(value)
        p = e / (1.0 + e)
    return 100.0 * p


def runs_of(points):
    """Collapse a (time, logit) series to one point per constant run.

    The page only changes when Exophase refreshes it, so a run of
    identical reads is one observation that held for a while, not
    many.  Each run contributes its first time; a closing point at
    the final time is added so the fit knows the last value held
    until the end of the record.
    """
    kept = []
    for t, value in points:
        if kept and kept[-1][1] == value:
            continue
        kept.append((t, value))
    if points and kept and kept[-1][0] != points[-1][0]:
        kept.append((points[-1][0], points[-1][1]))
    return kept


def weighted_anchor_fit(segments, last_logit):
    """Fit logit(t) = L + (last - L) * exp(-k * (t - t_last)).

    Given k the model is linear in the ceiling L, so a grid search
    over k with a closed-form weighted least squares for L (recent
    segments weigh more) finds the fit; a golden-section pass then
    polishes k.  Returns (ceiling, k, residual_variance, sse).
    """
    n = len(segments)
    t_last = segments[-1][0]
    span = max(t_last - segments[0][0], 1e-9)

    def solve(k):
        # With u = exp(k * (t_last - t)) >= 1 on the history, the
        # curve logit = L + (last - L) * u passes through the last
        # point and settles to the ceiling L after it; given k it is
        # linear in L, which the weighted least squares below solves.
        def basis(t):
            return math.exp(min(k * (t_last - t), 45.0))

        num = 0.0
        den = 0.0
        for t, value in segments:
            u = basis(t)
            w = 0.5 + 0.5 * (t - segments[0][0]) / span
            a = 1.0 - u
            num += w * a * (value - last_logit * u)
            den += w * a * a
        if den <= 1e-12:
            return last_logit, float("inf")
        ceiling = num / den
        lo = min(v for _, v in segments) - 1.0
        hi = max(v for _, v in segments) + 2.0
        ceiling = min(max(ceiling, lo), hi)
        sse = 0.0
        for t, value in segments:
            u = basis(t)
            w = 0.5 + 0.5 * (t - segments[0][0]) / span
            diff = value - (ceiling + (last_logit - ceiling) * u)
            sse += w * diff * diff
        return ceiling, sse

    flat_sse = 0.0
    for t, value in segments:
        w = 0.5 + 0.5 * (t - segments[0][0]) / span
        flat_sse += w * (value - last_logit) ** 2

    grid = [math.exp(-6.9 + 0.16 * i) for i in range(48)]
    best_k, best_ceiling, best_sse = 0.0, last_logit, flat_sse
    for k in grid:
        ceiling, sse = solve(k)
        if sse < best_sse:
            best_k, best_ceiling, best_sse = k, ceiling, sse

    if best_k > 0.0:
        lo_k = best_k * math.exp(-0.17)
        hi_k = best_k * math.exp(0.17)
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = hi_k - inv_phi * (hi_k - lo_k)
        x2 = lo_k + inv_phi * (hi_k - lo_k)
        _, f1 = solve(x1)
        _, f2 = solve(x2)
        for _ in range(22):
            if f1 > f2:
                lo_k = x1
                x1, f1 = x2, f2
                x2 = lo_k + inv_phi * (hi_k - lo_k)
                _, f2 = solve(x2)
            else:
                hi_k = x2
                x2, f2 = x1, f1
                x1 = hi_k - inv_phi * (hi_k - lo_k)
                _, f1 = solve(x1)
        k_mid = 0.5 * (lo_k + hi_k)
        ceiling, sse = solve(k_mid)
        if sse <= best_sse:
            best_k, best_ceiling, best_sse = k_mid, ceiling, sse

    # Accept the curve only when it clearly beats staying flat;
    # a marginal fit is noise, and noise extrapolates badly.
    if n < 3 or best_sse > 0.8 * flat_sse:
        dof = max(1, n - 1)
        return last_logit, 0.0, flat_sse / dof, flat_sse
    dof = max(1, n - 3)
    return best_ceiling, best_k, best_sse / dof, best_sse


def trophy_series(case, index):
    """(times_hours, rates) for one trophy, nulls dropped."""
    times = []
    rates = []
    base = None
    for read in case["history"]:
        rate = read["rates"][index]
        if rate is None:
            continue
        moment = parse_instant(read["at"])
        if base is None:
            base = moment
        times.append((moment - base) / 3600.0)
        rates.append(rate)
    return times, rates, base


def forecast_case(case):
    trophies = case["trophies"]
    levels = case["levels"]
    target = parse_instant(case["target"])
    zscores = [Z.inv_cdf(level) for level in levels]

    prepared = []
    for index, trophy in enumerate(trophies):
        times, rates, base = trophy_series(case, index)
        if not rates:
            prepared.append(None)
            continue
        points = [(t, logit_of_percent(r)) for t, r in zip(times, rates)]
        segments = runs_of(points)
        last_logit = points[-1][1]
        ceiling, speed, resid_var, _ = weighted_anchor_fit(segments, last_logit)
        last_time = times[-1]
        horizon_h = (target - (base + last_time * 3600.0)) / 3600.0
        span_h = max(times[-1] - times[0], 0.0)
        prepared.append({
            "tier": trophy["tier"],
            "n": len(segments),
            "last_logit": last_logit,
            "head": ceiling - last_logit,
            "speed": speed,
            "resid_var": resid_var,
            "horizon_h": max(horizon_h, 0.0),
            "span_h": span_h,
        })

    # Tier-level pooling: trophies whose own history is thin borrow
    # the headroom and speed typical of their tier.
    def pooled(key, transform, fallback):
        by_tier = {}
        overall = []
        for item in prepared:
            if item is not None and item["n"] >= 4 and item["speed"] > 0:
                value = transform(item)
                by_tier.setdefault(item["tier"], []).append(value)
                overall.append(value)
        result = {}
        for item in prepared:
            if item is None:
                continue
            group = by_tier.get(item["tier"], [])
            if len(group) >= 3:
                result[id(item)] = median(group)
            elif overall:
                result[id(item)] = median(overall)
            else:
                result[id(item)] = fallback
        return result

    med_head = pooled("head", lambda it: it["head"], 0.0)
    med_speed = pooled("speed", lambda it: math.log(it["speed"]), None)

    seen_logits = [it["last_logit"] for it in prepared if it is not None]
    global_last = median(seen_logits) if seen_logits else 0.0

    rows = []
    for trophy, item in zip(trophies, prepared):
        if item is None:
            centre = global_last
            sigma = 1.6
        else:
            head = item["head"]
            speed = item["speed"]
            weight = item["n"] / (item["n"] + 4.0)
            if item["n"] < 6:
                head = weight * head + (1.0 - weight) * med_head[id(item)]
                if speed > 0 and med_speed[id(item)] is not None:
                    log_speed = weight * math.log(speed) \
                        + (1.0 - weight) * med_speed[id(item)]
                    speed = math.exp(log_speed)
            horizon = item["horizon_h"]
            if speed > 0:
                centre = item["last_logit"] \
                    + head * (1.0 - math.exp(-speed * horizon))
            else:
                centre = item["last_logit"]
            centre = min(max(centre, item["last_logit"] - 4.0),
                         item["last_logit"] + 4.0)
            horizon_days = horizon / 24.0
            stretch = 1.0 + min(horizon / max(item["span_h"], 1.0), 3.0)
            var = max(item["resid_var"], RESID_FLOOR_VAR) * stretch
            var += (WALK_SD_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
            var += JUMP_SD ** 2
            sigma = min(max(math.sqrt(var), SIGMA_MIN), SIGMA_MAX)
        centre = min(max(centre, -LOGIT_CAP), LOGIT_CAP)
        quantiles = []
        for z in zscores:
            q = percent_of_logit(centre + z * sigma)
            q = round(min(max(q, 0.0001), 99.9999), 4)
            if quantiles and q < quantiles[-1]:
                q = quantiles[-1]
            quantiles.append(q)
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        try:
            result = forecast_case(case)
        except Exception:
            # Never leave a case unwritten: fall back to each
            # trophy's last observed rate with a wide honest spread.
            result = fallback_case(case)
        name = os.path.splitext(os.path.basename(path))[0] + ".out"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))
            handle.write("\n")
    return 0


def fallback_case(case):
    rows = []
    levels = case["levels"]
    zscores = [Z.inv_cdf(level) for level in levels]
    for index, trophy in enumerate(case["trophies"]):
        last = None
        for read in case["history"]:
            if read["rates"][index] is not None:
                last = read["rates"][index]
        centre = logit_of_percent(last) if last is not None else 0.0
        quantiles = []
        for z in zscores:
            q = round(percent_of_logit(centre + z * 1.2), 4)
            if quantiles and q < quantiles[-1]:
                q = quantiles[-1]
            quantiles.append(q)
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


if __name__ == "__main__":
    sys.exit(main(sys.argv))
