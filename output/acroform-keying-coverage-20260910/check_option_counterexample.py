"""Reproduce why column-wide caption-side consistency was rejected."""
import json
import sys
import types
from pathlib import Path

from scripts.acroform_keying import Scope, Snapshot

folder = Path(__file__).parent
case = json.loads((folder / 'independent-checkbox-counterexample.json').read_text())
page = Snapshot.model_validate(case['snapshot'])
source = Path('scripts/acroform_keying.py').read_text()
experiment = (folder / 'replay_option_sides.py').read_text()
namespace = {}
exec(experiment[experiment.index('addition = '):experiment.index('\nvariants={}')], namespace)
for weight in (0, 1, 3):
    code = source
    if weight:
        code = code.replace('def transition_penalty(', 'def original_transition_penalty(')
        code = code.replace('def assign(', namespace['addition'].replace('STYLE_PENALTY', str(weight)) + 'def assign(')
    module = types.ModuleType(f'counterexample_{weight}')
    sys.modules[module.__name__] = module
    exec(code, module.__dict__)
    module.Scope = Scope
    result = module.assign(page)
    paired = [result.values[i].native.index for c in result.selected for i in result.candidates[c].members]
    assert paired == ([0, 1] if weight == 0 else [1]), paired
    print(f'Extra penalty {weight}: paired native widgets {paired}')
