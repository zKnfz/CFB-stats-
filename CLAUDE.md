# Gridiron T-Rank

A Bart Torvik style ratings site for FBS college football, built by Cayden. The output is one self-contained HTML page (`gridiron-t-rank.html`) with all data embedded. It has four tabs: Ratings, Games, Matchup, Method.

Cayden is learning as he builds this. He prefers hints and explanations over just being handed solutions, unless he says he needs something fast. Never use dashes (em dashes or hyphens as punctuation) in writing for him.

## How to rebuild

```
pip install -r requirements.txt
./rebuild.sh        # about 2 minutes, downloads ~1.2 GB the first time
```

Output: `gridiron-t-rank.html` in the project root. The current season's files are re-downloaded on every run, so rerunning the script picks up new results.

Pipeline, in order (all scripts live in `src/`, run from `src/`):

| Step | Script | Output in `data/` |
|---|---|---|
| Download | `fetch_data.py 2026` | `pbp_YYYY`, `sched_YYYY`, `teaminfo_YYYY` parquet files, plus `up_cfb_schedules_2026.parquet` (full schedule including unplayed games) |
| Prep | `prep.py` (uses `build.load_season`) | `prep.pkl`: per team-game rows for every season |
| Walk-forward | `run_bt.py` (uses `model.walk`) | `bt.pkl`: pregame features for every 2008+ game |
| Early years | `pregame.py` | `pregame.pkl`: adds 2004 to 2007 |
| Fit + export | `export2.py` | `site_data.json`: model weights, backtest stats, per-game rows, priors |
| Games | `export3.py` | adds pregame picks for every past game and upcoming 2026 games |
| Spreads | `lines.py` | keeps `lines_hist.json` (committed): closing and opening spread plus total for every game since 2013, past seasons fetched once, current season refreshed each run. Fills spreads into upcoming games and adds `lineopen` and `tot` to `site_data.json` |
| Market odds | `kalshi.py` | adds live Kalshi odds to upcoming games, logs a snapshot to `kalshi_history.jsonl` |
| Preseason signal | `preseason.py` | `preseason.json` (committed): CFBD talent composite + returning production, only fetches seasons missing from the file |

CFBD's free tier caps monthly calls, so both CFBD scripts cache to committed files; a normal run makes only the 2 current season `/lines` calls.
| Page | `build_page.py` | injects `site_data.json` into `src/template.html` |

A GitHub Actions workflow (`.github/workflows/rebuild.yml`) runs this whole pipeline every 3 hours and on manual trigger, commits the updated page, and deploys it to GitHub Pages. The `CFBD_API_KEY` secret in the repo is actually named `CFBAPI`.

## Data sources

- Play-by-play with EPA: sportsdataverse `espn_cfb_pbp` GitHub releases, 2004 onward (nothing earlier exists).
- Schedules: `espn_cfb_schedules` releases (completed games only) and the cfbfastR-data repo's `schedules/parquet/cfb_schedules_YYYY.parquet` (CFBD based, includes upcoming games).
- Team info, conferences, FBS classification: cfbfastR-data `team_info/parquet/cfb_team_info_YYYY.parquet`.
- Game ids and team ids are ESPN ids across all sources.
- Coverage: 2004 is 66% of FBS games, 2005 to 2007 are 84 to 90%, 2020 is 81%, everything else 96%+.

## The model

Per stat, a weighted ridge regression over team-games: `value = mu + off[team] + def[opponent] + hfa*loc`, where loc is +1 home, -1 away, 0 neutral. The penalty pulls each team toward a prior: last season's final value times 0.9 (new FBS teams start at the 20th percentile).

Three stats feed the rating (chosen by walk-forward testing on 2008 to 2024):

- `PTS`: points scored per game, weight 1 per game, lambda 3
- `SR`: success rate, weighted by plays, lambda 100
- `NT`: EPA per play with turnover plays removed, lambda 100

Rating = NT net * NT beta + SR net * SR beta + PTS net * PTS beta (net = off minus def), refit each pipeline run so the exact numbers drift slightly; printed by `export2.py` as `beta`. That's projected margin vs an average team on a neutral field. Home field is also refit each run (the 4th beta value), around 3 points. Win probability = normal CDF of margin / sd (also refit, around 16).

The preseason prior gets one more nudge before the season's first ridge fit: each team's CFBD talent composite z-score for the upcoming season, times 0.8 (in std devs of that feature's own net rating), added to offense and subtracted from defense. See `PRESEASON_COEF` in `model.py`.

Play filters: runs, passes and sacks only, FBS vs FBS games only, no kneels, no two point tries, no garbage time (margin over 38 in Q2, 28 in Q3, 22 in Q4).

Raw EPA (`E`) is still solved for display columns, but it's not in the rating.

## Results so far

Out of sample (2025 and 2026 so far): average miss 12.40 points, picks 73.3% of winners. Vegas: 11.74 and 75.7%. The earlier EPA-only version missed by 12.70.

Against closing lines the market is hard to beat: on the 2025+ test years the closing spread alone misses by 11.74, and blending in any share of the model makes it slightly worse (10% model: 11.75, 30%: 11.82). Model vs line disagreements of 3+ points covered 60.5% in training years but 49.5% in test years. Fit on recent windows the share is about 0 (2016 to 2026: −0.02, 2021 to 2026: −0.06). So no edge vs closing lines is proven; beating opening lines is the untested next question. `lines_hist.json` has openers for 2023 on (about 2,400 games), earlier seasons only have closers.

Things tested that didn't help: recency weighting (half-lives of 42 to 140 days), including raw EPA alongside turnover-free EPA, adding explosiveness (EPA per successful, turnover-free play) as a fourth rating input. Explosiveness still shows as a display column (`EX` in `FEATS`), its signal is just already inside EPA per play, so it moved test MAE by under 0.01 points.

Checked whether the model's miss size varies by anything, to see if win probability should use a variable standard deviation instead of one flat number. Bucketing by projected margin size: no, std stays flat whether the favorite is up by 2 or up by 35. Bucketing by how many games each team has played so far that season: yes, games where neither team has played yet (pure preseason prior, no in season data) used to miss by more and run biased, about 3 points too generous to the favorite on average. `sd_early` in `site_data.json` params captures the std difference.

That preseason bias is mostly fixed now: added each team's CFBD recruiting talent composite (z-scored per season) as a nudge to the preseason prior, tuned by walk-forward backtest on 2015 to 2024 preseason-only games and confirmed on 2025 to 2026 (never touched while tuning, see `/tmp` scratch test script if it still exists, otherwise rerun the same grid search against `model.walk`). Preseason-only MAE went from 15.44 to 14.07 points and the bias nearly vanished, -5.38 to +0.04. Returning production (CFBD's `/player/returning`) was also tried, alone and blended with talent, and made things worse both ways, so it's fetched (`preseason.json` has it) but unused (`PRESEASON_COEF["returning"]` stays 0).

## The page

`src/template.html` holds all the UI and the browser math. The browser re-solves every ridge regression live (Cholesky) whenever the date range changes, so any range from 2004 to today works, including ranges that cross seasons. The JS solver was verified to match the Python one to two decimals.

The Games tab uses precomputed pregame picks from `site_data.json`. Each past pick only used games from before that week.

The Matchup tab has an Outlier style Line history panel, all computed in the browser from games with a closing line: how favorites of that size at that site have done since 2004, each team's record against the spread (season, last 10, as favorite, as underdog, at this site), head to head with lines, and how the model's side covered when it disagreed with the line by a similar amount (all years and 2025+ test years). It only uses games before the opened game's date, and for past games it uses that game's pregame pick, not hindsight ratings. Opening a game from the Games tab fills in the spread; it can also be typed in.

The Trends tab is a reverse engineered, more honest take on Outlier's feed. One card per upcoming game: the spread and the Kalshi moneyline, each with a fair probability, EV and quarter Kelly; opening line vs now; a Kalshi price sparkline (`kmove`, built by `kalshi.py` from `kalshi_history.jsonl`); and the two strongest auto generated trends (last 10 against the spread, season, as favorite or underdog, at this site, head to head) with a green/red hit strip. Unlike Outlier, every trend gets a signal check: "covered k of last 10" is backtested across every team and game since 2004 (it has not predicted the next game), and other splits show a Wilson 95% range plus a warning that they were picked from many possible splits. Fair probabilities do not use the raw model. They keep only `FIT.w` of the model's disagreement with the closing line, least squares on the last 5 seasons and clamped at 0. That share was about 0.37 on 2008 to 2024 but is about 0 on 2016 onward (markets tightened), so right now fair spread odds equal the market's and spread EV is never positive. Spread to win chance uses a logistic curve fit to real results (`WINK`, about 8.3 points), because a bell curve underrated big favorites (17 point favorites won 90%, the bell curve said 86%). Kalshi EV includes the taker fee (0.07 × price × (1 − price)) and is skipped when the Kalshi price is more than 12 points off the books, which is almost always a stale price on a thin market; any positive moneyline EV means Kalshi is cheaper than the books, not that the model knows better. A streak bucket only earns "Has held up" with 200+ games behind it. All computed in the browser. The tab opens on an Insights view, one Outlier style card per trend (spread, totals or moneyline), each with the bet and price, a hit strip colored by whether that card's bet hit, and a tap to open signal check. "Over hit k of last 10" is backtested the same way as the spread streak (`OU10`). Straight up win streaks are labeled priced in. The Picks view is the per game cards with EV. Hit rate cards come first: tonight's exact number checked against each team's recent games, like Outlier's props ("Notre Dame has won by 11+ in 10 of its last 10", "Tulane's games have stayed under 55.5 in 9 of the last 10"), with an L5 / L10 / L20 / season / head to head row. Shown when L10 is 8+. `HR10` backtests this across every game: teams that had beaten that game's spread in 10 of their previous 10 covered 52.8% (895 games), 9 of 10 covered 53.0%, so these are labeled likely noise; the books already price recent form.

## Known gaps and next ideas

- The current season (2026) is hardcoded in `rebuild.sh`, `fetch_data.py`'s default and `export3.py`. Make it one setting before next season.
- Preseason recruiting talent is in, returning production is fetched but didn't help. Still not tried: transfers, QB-specific returning production instead of whole-team.
- Not built yet: remaining Five Factors columns (field position, finishing drives), player stats (QB, rusher and receiver EPA are possible; defenders aren't), projected final records, a wins-above-playoff-team résumé stat.
