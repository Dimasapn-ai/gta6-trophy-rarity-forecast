# Launch-Week Trophy Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Dimasapn-ai/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Dimasapn-ai/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of the 55 GTA VI PS5 trophies, five quantiles (5/25/50/75/95%) of the share of players who will have earned it one week after launch — scored by pinball loss on the logit scale.

The model fits an anchored saturating curve on the logit of each trophy's earn rate (anchored on the last observed rate, with a fitted ceiling and speed), shrinks sparse trophies toward their grade tier's median behaviour, and builds honest, widening quantile bands from fit residuals, horizon growth and a refresh-jump floor. Fully offline, deterministic, Python standard library only. `validate.py` holds the practice-case validation (refresh-shock and synthetic-growth regimes) against a flat last-rate baseline.

---

<sub>Entered on [make.fun](https://make.fun/?ref=tdk3vr62&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
