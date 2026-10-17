"""CI guard for the forward logs (tracking.jsonl, kalshi_history.jsonl, kalshi_flags.jsonl).

1. Append only: the committed version of each file must still be an exact prefix of the new one,
   so no saved line was changed, reordered or removed.
2. Provenance: every new line must be valid JSON stamped with this CI run's id (forward_log.provenance),
   so nothing written by a local run, a test or by hand can reach the real logs.
Exits non zero on any problem, which stops the deploy. Run from the repo root after the pipeline.
"""
import json, os, subprocess, sys

FILES = ["tracking.jsonl", "kalshi_history.jsonl", "kalshi_flags.jsonl"]


def committed(path):
    r = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else b""


def main():
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        sys.exit("check_logs.py is for CI: GITHUB_RUN_ID is not set")
    bad = []
    for f in FILES:
        new = open(f, "rb").read() if os.path.exists(f) else b""
        old = committed(f)
        if not new.startswith(old):
            bad.append(f"{f}: a committed line was changed, reordered or removed")
            continue
        added = [ln for ln in new[len(old):].decode().splitlines() if ln.strip()]
        for i, ln in enumerate(added, 1):
            try:
                e = json.loads(ln)
            except ValueError:
                bad.append(f"{f}: new line {i} is not valid JSON"); continue
            if str(e.get("run_id")) != run_id:
                bad.append(f"{f}: new line {i} has run_id {e.get('run_id')!r}, not this run's {run_id}")
        print(f"{f}: {len(old.splitlines())} saved lines untouched, {len(added)} new from run {run_id}")
    if bad:
        for b in bad:
            print("::error::" + b)
        sys.exit(1)
    print("forward logs: only appended, every new line from this run")


if __name__ == "__main__":
    main()
