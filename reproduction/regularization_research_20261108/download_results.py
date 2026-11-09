"""Read-only, version-pinned Kaggle artifact download with a local hash receipt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='1')
    parser.add_argument('--mode', choices=['analysis','all'], default='analysis')
    args = parser.parse_args()
    target = Path(__file__).resolve().parent / 'downloads' / ('gpu-v' + args.version)
    target.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate()
    token = None; seen = set(); items = []; inventory = []
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = 'den1shh'; request.kernel_slug = 'proteus-regularization-wave2'
        request.version_label = api._kernel_version_label(args.version)
        request.page_size = 200
        if token: request.page_token = token
        for attempt in range(6):
            try:
                with api.build_kaggle_client() as client:
                    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                break
            except requests.RequestException:
                if attempt == 5: raise
                time.sleep(1)
        if response.log: (target / 'kaggle.log').write_text(response.log)
        for item in response.files or []:
            rel = PurePosixPath(item.file_name)
            if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('Unsafe output filename')
            inventory.append(item.file_name)
            if item.file_name.startswith('regularization-results/'):
                selected_curves={'swad0.1','ema0.99','ls0.2','ls0.1','ls05_noise03_drop3','drop3','mixstyle0.3','shift12'}
                parts=rel.parts
                keep=args.mode=='all' or len(parts)==2 or rel.name in ('prediction.npz','prediction.json','logits.npy','failure.json')
                if len(parts)>4 and parts[3] in selected_curves and parts[-2]=='selection' and rel.name in ('curves.csv','complete.json','curves.png'):keep=True
                if len(parts)>4 and parts[3]=='swad0.1' and rel.name in ('model.pt','alignment.npz'):keep=True
                if keep:items.append((item.file_name, item.url))
        token = response.next_page_token
        if not token: break
        if token in seen: raise RuntimeError('Repeated pagination token')
        seen.add(token)
    (target / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    print('LISTED', len(inventory), 'SELECTED', len(items), flush=True)

    def download(item):
        name, url = item; path = target / name; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            for attempt in range(6):
                try:
                    with requests.get(url, stream=True, timeout=45) as response:
                        response.raise_for_status()
                        partial = path.with_suffix(path.suffix + '.partial')
                        with partial.open('wb') as f:
                            for block in response.iter_content(1024 * 1024): f.write(block)
                    partial.replace(path); break
                except requests.RequestException:
                    if attempt == 5: raise
                    time.sleep(1)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()

    hashes = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for n, (name, digest) in enumerate(pool.map(download, items), 1):
            hashes[name] = digest
            if n % 100 == 0: print('DOWNLOADED', n, '/', len(items), flush=True)
    (target / 'download-receipt.json').write_text(json.dumps(dict(version=args.version, mode=args.mode, files=hashes), indent=2))
    print('COMPLETE DOWNLOAD', len(hashes), target, flush=True)


if __name__ == '__main__': main()
