# Launch-Week Trophy Forecaster — notes

`main.py` forecasts, for each of a case's 55 GTA VI PS5 trophies, the
5/25/50/75/95% quantiles of its Exophase earn rate at the case target,
on the logit scale the pinball loss is scored in.

## Model

- The page only changes when Exophase refreshes it, so each trophy's
  reads collapse to one point per constant run.
- Per trophy, a saturating curve anchored on the last observed rate,
  `logit(t) = L + (last - L) * exp(-k * (t - t_last))`, is fitted by a
  grid + golden-section search over `k` with a closed-form weighted
  least squares for the ceiling `L` (recent runs weigh more). The
  curve is accepted only if it beats staying flat by a clear margin;
  the forecast centre is the curve read at the target, so growth
  bends toward the ceiling instead of extrapolating linearly.
- Trophies with thin history (< 6 runs) shrink their headroom and
  speed toward the median of their tier (platinum/gold/silver/
  bronze), falling back to the field median when a tier is thin.
- Spread: residual variance of the fit, stretched as the horizon
  grows beyond the observed span, plus a random-walk term
  `0.42 * sqrt(horizon days)` and a jump floor (SD 0.55) that prices
  in Exophase refreshing the page between the last read and the
  target — the first refresh of launch night multiplied the tracked
  players sixfold and dragged nearly every rate down with it.
- Rates are clamped to the scorer's [0.01, 99.99] for logits;
  quantiles are monotone, rounded to 4 decimals, deterministic.

## Validation (validate.py, pinball loss on logit, lower is better)

| regime | model | flat | house baseline |
| --- | --- | --- | --- |
| practice holdout (degenerate: page never moves) | 0.112 | 0.030 | 0.025 |
| practice + documented first-refresh shock | 1.374 | 1.670 | 1.688 |
| synthetic growth week, 5 seeds | 0.251–0.261 | 0.496–0.568 | 0.352–0.385 |

The holdout regime is the one case where a tight flat forecast must
win: the practice histories are a single unchanging page, so the
answer is the last rate by construction and every point of honest
spread is pure cost. The regimes that resemble the final — a refresh
the history has not seen, and a week of saturating growth — are the
ones the model is built for, and it beats both baselines clearly in
each. Edge cases checked: null rates, a trophy with no observations,
a one-read history, and a 400-read case (0.2 s of the 8 s budget);
two runs of any case are byte-identical.
