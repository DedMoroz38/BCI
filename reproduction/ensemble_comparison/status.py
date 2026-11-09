"""Read-only progress; optional background status-file watcher."""
import argparse,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def snapshot():
    records=[]
    for p in (HERE/'results').glob('*/*.json'):
        if p.stem=='baseline' or p.stem.startswith('variant'):
            records.append(json.loads(p.read_text()))
    done=sum(len([r for r in records if r['tag']==tag])==8 for tag in set(r['tag'] for r in records))
    verification=HERE/'verification.json'
    log=(HERE/'logs/run.log').read_text() if (HERE/'logs/run.log').exists() else ''
    state='complete' if verification.exists() else ('failed' if 'Traceback (most recent call last)' in log else 'running')
    return dict(status=state,evaluations=len(records),expected_evaluations=80,complete_folds=done,
                expected_folds=10,log_tail=log.splitlines()[-6:],
                report=str(HERE/'REPORT.md') if verification.exists() else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        value=snapshot()
        if not args.watch:
            print(json.dumps(value,indent=2,ensure_ascii=False));return
        path=HERE/'RUN_STATUS.json'
        current=json.loads(path.read_text()) if path.exists() else {}
        current.update(value);current['updated_epoch']=time.time()
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(current,indent=2,ensure_ascii=False));temp.replace(path)
        if value['status']!='running':return
        time.sleep(10)

if __name__=='__main__':main()
