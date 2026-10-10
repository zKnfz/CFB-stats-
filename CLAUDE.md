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
| Market odds | `kalshi.py` | adds live Kalshi odds to upcoming games, logs a snapshot to `kalshi_history.jsonl` |
| Preseason signal | `preseason.py` | `preseason.json`: CFBD talent composite + returning production, needs `CFBD_API_KEY` env var, skips gracefully without it |
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

Things tested that didn't help: recency weighting (half-lives of 42 to 140 days), including raw EPA alongside turnover-free EPA, adding explosiveness (EPA per successful, turnover-free play) as a fourth rating input. Explosiveness still shows as a display column (`EX` in `FEATS`), its signal is just already inside EPA per play, so it moved test MAE by under 0.01 points.

Checked whether the model's miss size varies by anything, to see if win probability should use a variable standard deviation instead of one flat number. Bucketing by projected margin size: no, std stays flat whether the favorite is up by 2 or up by 35. Bucketing by how many games each team has played so far that season: yes, games where neither team has played yet (pure preseason prior, no in season data) used to miss by more and run biased, about 3 points too generous to the favorite on average. `sd_early` in `site_data.json` params captures the std difference.

That preseason bias is mostly fixed now: added each team's CFBD recruiting talent composite (z-scored per season) as a nudge to the preseason prior, tuned by walk-forward backtest on 2015 to 2024 preseason-only games and confirmed on 2025 to 2026 (never touched while tuning, see `/tmp` scratch test script if it still exists, otherwise rerun the same grid search against `model.walk`). Preseason-only MAE went from 15.44 to 14.07 points and the bias nearly vanished, -5.38 to +0.04. Returning production (CFBD's `/player/returning`) was also tried, alone and blended with talent, and made things worse both ways, so it's fetched (`preseason.json` has it) but unused (`PRESEASON_COEF["returning"]` stays 0).

## The page

`src/template.html` holds all the UI and the browser math. The browser re-solves every ridge regression live (Cholesky) whenever the date range changes, so any range from 2004 to today works, including ranges that cross seasons. The JS solver was verified to match the Python one to two decimals.

The Games tab uses precomputed pregame picks from `site_data.json`. Each past pick only used games from before that week.

## Known gaps and next ideas

- The current season (2026) is hardcoded in `rebuild.sh`, `fetch_data.py`'s default and `export3.py`. Make it one setting before next season.
- Preseason recruiting talent is in, returning production is fetched but didn't help. Still not tried: transfers, QB-specific returning production instead of whole-team.
- Not built yet: remaining Five Factors columns (field position, finishing drives), player stats (QB, rusher and receiver EPA are possible; defenders aren't), projected final records, a wins-above-playoff-team résumé stat.
