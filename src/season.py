"""The current season, in one place. Every pipeline script reads it from here.

Change it once a year in August, after the new season's schedule is published (cfbfastR-data's
schedules file for that year), then run the pipeline and check it before the first games. The SEASON
environment variable overrides it for a one off run.
"""
import os

SEASON = int(os.environ.get("SEASON") or 2026)
