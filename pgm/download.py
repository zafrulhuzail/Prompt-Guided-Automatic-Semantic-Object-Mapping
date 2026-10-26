"""Download frozen COCO pilot images and pinned CLIP assets; retain checksums."""
import concurrent.futures
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def main():
    data = ROOT/'data'
    data.mkdir(exist_ok=True)
    ann = data/'instances_val2017.json'
    if not ann.exists():
        archive=data/'annotations_trainval2017.zip'
        if not archive.exists():
            with urllib.request.urlopen('https://images.cocodataset.org/annotations/annotations_trainval2017.zip',timeout=180) as source, archive.open('wb') as dest:
                import shutil
                shutil.copyfileobj(source,dest)
        with zipfile.ZipFile(archive) as z:
            ann.write_bytes(z.read('annotations/instances_val2017.json'))
    expected='e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
    if hashlib.sha256(ann.read_bytes()).hexdigest()!=expected:
        raise ValueError('COCO annotation checksum mismatch')
    raw = json.loads(ann.read_text())
    images = sorted(raw['images'], key=lambda x:x['id'])[:24]
    folder = data/'val2017'
    folder.mkdir(exist_ok=True)
    def fetch(im):
        url = im['coco_url'].replace('http:', 'https:')
        dest = folder/im['file_name']
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=60) as source:
                dest.write_bytes(source.read())
        return dict(image_id=im['id'], url=url, file=str(dest.relative_to(ROOT)),
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), license_id=im['license'],
                    flickr_url=im.get('flickr_url'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        provenance = list(pool.map(fetch, images))
    (data/'provenance.json').write_text(json.dumps(dict(images=provenance,
        annotation_sha256=hashlib.sha256(ann.read_bytes()).hexdigest(),
        licenses=raw['licenses'], date='2026-10-26'),indent=2))
    print('Downloaded',len(images),'images',flush=True)
    from huggingface_hub import snapshot_download
    model = 'openai/clip-vit-base-patch32'
    revision = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    (data/'model_revision.json').write_text(json.dumps(dict(model=model,revision=revision)))
    path = snapshot_download(model, revision=revision, cache_dir=str(ROOT/'models'),
                             allow_patterns=['*.json','*.txt','merges.txt','vocab.json','pytorch_model.bin','README.md'])
    files = {str(p.relative_to(Path(path))):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in Path(path).rglob('*') if p.is_file()}
    (data/'model_files.json').write_text(json.dumps(dict(revision=revision,files=files),indent=2))
    print('Model cached at',path,flush=True)
if __name__ == '__main__':
    main()
