"""Diagnostic changes in memory only; no optimizer or truth files are modified."""
import json, sys, types
from pathlib import Path
from collections import Counter
from scripts.replay_acroform_keying import FieldReview,evaluate
from scripts.acroform_keying import Scope

root=Path.cwd(); out=root/'output/acroform-keying-coverage-20260910'
source=(out/'baseline_algorithm.py').read_text()
refs=[FieldReview.model_validate_json(line) for line in (root/'tests/data/groundtruth/acroform_keying/field_reviews.jsonl').read_text().splitlines()]
variants={f'coverage_{threshold}':source.replace('contains(label.bbox, values[i].bbox)', f'values[i].bbox.intersection_over_self(label.bbox) >= {threshold}') for threshold in (0.8,0.9,0.95,0.99)}

results={}
for name,code in variants.items():
 mod=types.ModuleType(name);sys.modules[name]=mod;exec(compile(code,str(root/'scripts/acroform_keying.py'),'exec'),mod.__dict__);mod.Scope=Scope
 counts=Counter();changes=[];groups=[]
 for p in sorted((root/'output/acroform-keying-review-20260908/snapshots').glob('*/*.json')):
  s=mod.Snapshot.model_validate_json(p.read_text()); a=mod.assign(s)
  rr,_=evaluate(s,a,[],[r for r in refs if r.fixture==p.parent.name and r.page==s.page])
  slug=f'{p.parent.name}-p{s.page}';baseline=json.loads((root/'output/acroform-keying-prototype'/f'{slug}.json').read_text())['reviews']
  assert [r['widget_index'] for r in baseline] == [r.widget_index for r in rr]
  assert [r['widget_index'] for r in baseline if r['status']=='excluded by table rule'] == [r.widget_index for r in rr if r.status=='excluded by table rule']
  for old,new in zip(baseline,rr):
   counts[new.status]+=1
   if old['status']!=new.status or old['predicted']!=new.predicted:
    changes.append({'page':slug,'widget':new.widget_index,'before':old['status'],'after':new.status,'old_text':old['predicted'],'new_text':new.predicted})
  groups.append({'page':slug,'groups':[{'kind':a.candidates[c].kind,'members':[a.values[i].native.index for i in a.candidates[c].members],'text':a.labels[a.candidates[c].label].text} for c in a.selected if a.candidates[c].kind in {'inline_clause','composite_field','choice_group'}]})
 results[name]={'counts':dict(counts),'fixes':[r for r in changes if r['before'] not in {'correct','correct abstention'} and r['after'] in {'correct','correct abstention'}],'regressions':[r for r in changes if r['before'] in {'correct','correct abstention'} and r['after'] not in {'correct','correct abstention'}],'changes':changes,'groups':groups}
 print(name,dict(counts),'fixes',len(results[name]['fixes']),'regressions',len(results[name]['regressions']),flush=True)
(out/'ablations.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
