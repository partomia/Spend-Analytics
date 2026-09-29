#!/usr/bin/env python3
"""
Cloudera AI Application launcher for the Spend Insights Console.

Project > Applications > New Application: Script app/run.py, Python 3.11
runtime, 2 vCPU / 4 GB (8 GB if Customer 360 runs the models in-app without
the endpoint). Set SPEND_IMPALA_* and SPEND_ENDPOINT_URL / _ACCESS_KEY /
_API_KEY as project environment variables.

Also works from a Session terminal: python app/run.py
"""

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # A CAI Application with a JupyterLab kernel runs this inside IPython,
    # where __file__ is undefined; CAI sets the cwd to the project root.
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        cwd = Path.cwd()
        return cwd if (cwd / "app" / "streamlit_app.py").exists() else Path("/home/cdsw")


ROOT = _repo_root()
PORT = os.environ.get("CDSW_APP_PORT", "8090")
HOST = os.environ.get("SPEND_APP_HOST", "127.0.0.1")

cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
       "--server.port", PORT, "--server.address", HOST, "--server.headless", "true",
       "--server.enableCORS", "false", "--server.enableXsrfProtection", "false",
       "--browser.gatherUsageStats", "false"]
print(f"[app/run] {' '.join(cmd)}  (cwd={ROOT})", flush=True)
# Run Streamlit as a child and block: exec-ing would replace the Jupyter kernel
# process, which CAI treats as the application dying.
code = subprocess.call(cmd, cwd=ROOT)
if code:
    raise SystemExit(f"streamlit exited with status {code}")
