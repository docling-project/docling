"""Replay the isolated grouping experiment through the existing evaluator/report."""

import importlib.util
import sys
from pathlib import Path

from scripts import acroform_keying, replay_acroform_keying

folder = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "grouping_experiment", folder / "experimental_algorithm.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
# The evaluator compares Scope values; use the shared identity across modules.
module.Scope = acroform_keying.Scope
replay_acroform_keying.assign = module.assign
if __name__ == "__main__":
    sys.argv[1:] = ["--out", str(folder / "report"), *sys.argv[1:]]
    replay_acroform_keying.main()
