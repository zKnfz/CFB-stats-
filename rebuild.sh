#!/usr/bin/env bash
# Full rebuild. Run from the project root. The backtest step (run_bt.py) takes several minutes.
set -e
cd "$(dirname "$0")/src"
python3 fetch_data.py 2026
python3 prep.py
python3 run_bt.py
python3 pregame.py
python3 export2.py
python3 export3.py
python3 kalshi.py
python3 preseason.py   # needs CFBD_API_KEY env var set, skips gracefully if not
python3 build_page.py
