#!/usr/bin/env python3
"""
CAI Job spend-sync-code: the first job GitHub Actions starts through the CAI API
(ci/trigger_cai_pipeline.py), so every job after it runs exactly the pushed commit.

  1. git fetch + reset --hard to origin/<GIT_BRANCH> (default main) in the project.
     Data, models, artifacts/ and .env are gitignored, so they are untouched; an
     uncommitted edit to a tracked file is discarded (develop in Git, not in the
     CAI project).
  2. EXPECTED_GIT_SHA (set by the trigger in the run's environment) stops the chain
     if the branch has already moved on; the newer push runs its own chain.

Jobs > New Job: script cai/jobs/sync_code.py, Python 3.11 runtime, 1 vCPU / 2 GB.

Exit codes: 0 project at the expected commit; 1 the branch moved on; a git error
fails the job.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path(os.getcwd())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_repo_root(), text=True).strip()


def main() -> int:
    branch = os.environ.get("GIT_BRANCH", "main")
    git("fetch", "origin", branch)
    git("reset", "--hard", f"origin/{branch}")
    sha = git("rev-parse", "HEAD")
    print(f"project now at {sha} (origin/{branch})")
    expected = os.environ.get("EXPECTED_GIT_SHA", "")
    if expected and not sha.startswith(expected):
        print(f"expected commit {expected} but origin/{branch} is {sha[:12]} (a newer push?): stopping the chain")
        return 1
    return 0


if __name__ == "__main__":
    # CAI Jobs run a script inside a Jupyter kernel wrapper: any SystemExit,
    # even sys.exit(0), is reported as a failure. Only exit on real failure.
    _rc = main()
    if _rc:
        sys.exit(_rc)
