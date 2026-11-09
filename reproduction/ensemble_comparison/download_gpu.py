"""Guarded,closed-response downloads of partial/full Kaggle outputs."""
import argparse,hashlib,json,time
from pathlib import Path,PurePosixPath
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='2');p.add_argument('--slug',default='den1shh/proteus-author-ensemble-vs-lag4');args=p.parse_args()
    label='recovery' if 'recovery' in args.slug else 'gpu'
    target=HERE/'downloads'/(label+'-v'+args.version);target.mkdir(parents=True,exist_ok=True)
    api=KaggleApi();api.authenticate();token=None;seen=set();files={};count=0
    with requests.Session() as http:
        while True:
            request=ApiListKernelSessionOutputRequest();request.user_name='den1shh'
            request.kernel_slug=args.slug.split('/')[-1];request.version_label=api._kernel_version_label(args.version)
            request.page_size=200
            if token:request.page_token=token
            for attempt in range(6):
                try:
                    with api.build_kaggle_client() as client:
                        response=client.kernels.kernels_api_client.list_kernel_session_output(request)
                    break
                except requests.RequestException:
                    if attempt==5:raise
                    time.sleep(1)
            for item in response.files or []:
                rel=PurePosixPath(item.file_name)
                if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe output path')
                if not item.file_name.startswith('ensemble-results/'):continue
                path=target/str(rel);path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():
                    for attempt in range(6):
                        try:
                            with http.get(item.url,stream=True,timeout=60) as r:
                                r.raise_for_status()
                                with path.with_suffix(path.suffix+'.partial').open('wb') as f:
                                    for block in r.iter_content(1024*1024):f.write(block)
                            path.with_suffix(path.suffix+'.partial').replace(path);break
                        except requests.RequestException:
                            if attempt==5:raise
                            time.sleep(1)
                files[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest();count+=1
                if count%30==0:print('DOWNLOADED',count,flush=True)
            token=response.next_page_token
            if not token:break
            if token in seen:raise RuntimeError('Repeated pagination token')
            seen.add(token)
    (target/'download-receipt.json').write_text(json.dumps(dict(version=args.version,files=files),indent=2))
    print('COMPLETE DOWNLOAD',count,'files',target,flush=True)

if __name__=='__main__':main()
