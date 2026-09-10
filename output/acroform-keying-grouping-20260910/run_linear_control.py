"""Control: count supported option relationships without a per-group saturation."""

import re
import sys
import types
from pathlib import Path

from run_experiment import acroform_keying, replay_acroform_keying

folder = Path(__file__).parent
source = (folder / "experimental_algorithm.py").read_text()
source, count = re.subn(
    r"-2\.0\s*\*\s*\(len\(members\) - 1\)\s*/\s*len\(members\)",
    "-float(len(members) - 1)",
    source,
)
assert count == 1
source = source.replace(
    '"intervening_blocks": 0.25 * boundaries', '"intervening_blocks": 1.0 * boundaries'
)
module = types.ModuleType("linear_group_control")
sys.modules[module.__name__] = module
exec(source, module.__dict__)
module.Scope = acroform_keying.Scope
replay_acroform_keying.assign = module.assign
if __name__ == "__main__":
    sys.argv[1:] = ["--out", str(folder / "linear-control-report"), *sys.argv[1:]]
    replay_acroform_keying.main()
