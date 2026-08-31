"""Run one BearVision 3 behavioural scenario and print its deterministic trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearvision.contracts import load_scenario
from bearvision.config import load_edge_config
from bearvision.simulation import build_behavioral_system


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/edge.yaml")
    args = parser.parse_args()
    result = build_behavioral_system(
        load_scenario(args.scenario),
        edge_config=load_edge_config(args.config),
    ).run()
    for entry in result.trace:
        print(json.dumps({"at_s": entry.at_s, "kind": entry.kind, "payload": entry.payload}, default=str))
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
