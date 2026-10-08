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
| Page | `build_page.py` | injects `site_data.json` into `src/template.html` |

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

Rating = 0.773 * PTS net + 34.8 * SR net + 5.23 * NT net (net = off minus def). That's projected margin vs an average team on a neutral field. Home field = 3.19 points. Win probability = normal CDF of margin / 16.34.

Play filters: runs, passes and sacks only, FBS vs FBS games only, no kneels, no two point tries, no garbage time (margin over 38 in Q2, 28 in Q3, 22 in Q4).

Raw EPA (`E`) is still solved for display columns, but it's not in the rating.

## Results so far

Out of sample (2025 and 2026 through Oct 7): average miss 12.62 points, picks 73% of winners. Vegas: 11.74 and 76%. The earlier EPA-only version missed by 12.89.

Things tested that didn't help: recency weighting (half-lives of 42 to 140 days), including raw EPA alongside turnover-free EPA, adding explosiveness (EPA per successful, turnover-free play) as a fourth rating input. Explosiveness still shows as a display column (`EX` in `FEATS`), its signal is just already inside EPA per play, so it moved test MAE by under 0.01 points.

## The page

`src/template.html` holds all the UI and the browser math. The browser re-solves every ridge regression live (Cholesky) whenever the date range changes, so any range from 2004 to today works, including ranges that cross seasons. The JS solver was verified to match the Python one to two decimals.

The Games tab uses precomputed pregame picks from `site_data.json`. Each past pick only used games from before that week.

## Known gaps and next ideas

- Updates are manual: rerun `rebuild.sh` and republish. A daily scheduled rebuild would make the Games tab update like Torvik's.
- The current season (2026) is hardcoded in `rebuild.sh`, `fetch_data.py`'s default and `export3.py`. Make it one setting before next season.
- Biggest remaining accuracy gap vs Vegas: preseason info (returning production, transfers, recruiting). The CollegeFootballData API has this but needs a free API key.
- Not built yet: remaining Five Factors columns (field position, finishing drives), player stats (QB, rusher and receiver EPA are possible; defenders aren't), projected final records, a wins-above-playoff-team résumé stat.
