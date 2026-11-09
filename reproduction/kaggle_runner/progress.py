"""Summarize rounded live-log metrics; final figures require saved predictions."""
from pathlib import Path
import json,re
rows={}
for line in (Path(__file__).resolve().parent/'live-v4.jsonl').read_text().splitlines():
    event=json.loads(line)
    for text in event.get('data','').splitlines():
        match=re.search(r'DONE MODEL (\w+) (\S+) session= (\S+) cell= (\S+) seconds= (\S+)',text)
        if match:
            name,tag,session,cell,seconds=match.groups()
            rows[name,tag]=dict(variant=name,tag=tag,session_mean=float(session),cell_mean=float(cell),seconds=float(seconds))
summary={name:sum(n==name for n,t in rows) for name in sorted(set(n for n,t in rows))}
print(json.dumps(dict(completed_models=len(rows),expected=70,by_variant=summary),indent=2))
for name in ['baseline','mlp']:
    cv=[r['session_mean'] for (n,t),r in rows.items() if n==name and t.startswith('loso_')]
    if len(cv)==8:print(name,'all8 LOSO mean session BA (rounded logs)',sum(cv)/8)
