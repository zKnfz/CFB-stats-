"""Where the forward logs live, and who may write them.

tracking.jsonl, kalshi_history.jsonl and kalshi_flags.jsonl are forward tests: real decisions and prices,
recorded at the time, committed and append only. Synthetic or test entries must never reach them.

So the real files can only be written from the CI rebuild (GITHUB_ACTIONS=true). Any other run (a local
./rebuild.sh, a test, a script run by hand) writes to data/local_<name> instead, which .gitignore keeps
out of git. A path override (TRACK_LOG, KFLAG_LOG, KHIST_LOG) still wins, for tests that want their own file.
Every new line also carries where it came from (CI run id and commit), and check_logs.py in CI refuses any
new line that doesn't belong to the current run.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def in_ci():
    return os.environ.get("GITHUB_ACTIONS") == "true"


def log_path(name, override_env):
    """Real committed file in CI; a gitignored local file anywhere else; an explicit override always wins."""
    if os.environ.get(override_env):
        return os.environ[override_env]
    if in_ci():
        return os.path.join(ROOT, name)
    from paths import DATA_DIR
    return os.path.join(DATA_DIR, "local_" + name)


def provenance():
    """Stamped on every new line: which CI run and commit made it ('local' outside CI)."""
    if in_ci():
        return {"run_id": os.environ.get("GITHUB_RUN_ID"), "code_sha": os.environ.get("GITHUB_SHA")}
    return {"run_id": "local", "code_sha": None}
