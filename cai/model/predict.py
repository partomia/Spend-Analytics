"""
CAI Model Deployment spend-scorer: one customer's spend profile, 12-week
forecast and persona what-if, and live merchant-text categorisation.

Model Deployments > New Model: File cai/model/predict.py, Function predict,
Python 3.11 runtime, 4 vCPU / 16 GB (both models run on CPU, PLAN decision 7),
1 replica, authentication on. The model build runs cdsw-build.sh.

At start-up each replica loads MiniLM and Chronos-2 once and the latest
published daily run from gold via Impala (its scaler, persona centroids and
pinned snapshot IDs; needs SPEND_IMPALA_USER / SPEND_IMPALA_PASSWORD in the
model's environment), or the local parquet store if Impala is not configured.
Restart the model to pick up a newer daily run. Request / response format: see
spend/scoring.py.
"""

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from spend.scoring import load_context, score  # noqa: E402

try:
    import cml.models_v1 as models

    cml_model = models.cml_model
except ImportError:  # outside CAI (local tests)
    def cml_model(fn):
        return fn

CTX = load_context(os.environ.get("SPEND_ENDPOINT_CONTEXT", "auto"))


@cml_model
def predict(args):
    return score(args, CTX)
