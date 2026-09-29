#!/usr/bin/env python3
"""
Try the scorer with the planted demo customer (unit 0, a Bengaluru foodie with
a Goa trip in Sep 2026): his profile (Tier-1 traveller, TRAVEL_SURGE), 12-week
forecast, a what-if that takes the trip away (back to Tier-1 digital foodie),
and live categorisation of UPI payee names.

  python cai/model/test_endpoint.py --print-request             # JSON for the model's Test tab, one per action
  python cai/model/test_endpoint.py --local                     # in-process, local parquet store
  python cai/model/test_endpoint.py --local --context impala    # in-process, gold via Impala
  python cai/model/test_endpoint.py --local --stub              # no torch needed
  python cai/model/test_endpoint.py                             # the deployed endpoint (SPEND_ENDPOINT_*)
  python cai/model/test_endpoint.py --action forecast           # one request only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

DEMO_CIF = "C00016807"   # unit 0: Bengaluru affluent foodie, Goa trip in Sep 2026
EXAMPLES = {
    "profile": {"action": "profile", "cif": DEMO_CIF},
    "forecast": {"action": "forecast", "cif": DEMO_CIF, "categories": ["TRAVEL", "FOOD_DELIVERY", "DINING"]},
    "what_if": {"action": "what_if", "cif": DEMO_CIF, "shift": {"TRAVEL": 0.0}},     # without the Goa trip
    "categorise": {"action": "categorise",
                   "merchant_texts": ["RAMESH KIRANA STORE", "UBER TRIP 8841", "APOLLO PHARMACY 2231",
                                      "PAYTM*BESCOM BILL", "ZOMATO ORDER 5521", "INDIANOIL PETROL PUMP"]},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--print-request", action="store_true")
    p.add_argument("--local", action="store_true", help="score in-process instead of calling the endpoint")
    p.add_argument("--stub", action="store_true", help="with --local: n-gram + seasonal-naive stand-ins")
    p.add_argument("--context", default="file", help="with --local: impala | file | auto")
    p.add_argument("--action", choices=list(EXAMPLES), default=None, help="send one request only")
    args, _ = p.parse_known_args()  # a Jupyter-kernel job runtime adds -f <kernel.json>

    reqs = {args.action: EXAMPLES[args.action]} if args.action else EXAMPLES
    if args.print_request:
        for name, req in reqs.items():
            print(f"# {name}\n{json.dumps(req, indent=2)}")
        return
    t0 = time.time()
    if args.local:
        from spend.scoring import load_context, score

        ctx = load_context(args.context, stub=args.stub)
        print(f"context loaded in {time.time() - t0:.1f}s")
        call = lambda r: score(r, ctx)  # noqa: E731
    else:
        from spend.client import call_endpoint, endpoint_configured

        if not endpoint_configured():
            sys.exit("SPEND_ENDPOINT_URL is not set (or use --local)")
        call = call_endpoint
    failed = False
    for name, req in reqs.items():
        t1 = time.time()
        resp = call(req)
        print(f"\n# {name}: answered in {time.time() - t1:.2f}s")
        print(json.dumps(resp, indent=2, default=str))
        failed |= "error" in resp
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
