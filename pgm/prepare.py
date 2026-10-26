"""Convert local official COCO instance annotations into disjoint inputs and labels."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=24)
    a = p.parse_args()
    data = json.loads(Path(a.annotations).read_text())
    categories = {c['id']: c['name'] for c in data['categories']}
    images = sorted(data['images'], key=lambda x:x['id'])[:a.count]
    ids = {im['id']:im for im in images}
    candidates, labels = [], {}
    for ann in sorted(data['annotations'], key=lambda x:x['id']):
        if ann['image_id'] not in ids or ann.get('iscrowd') or min(ann['bbox'][2:]) < 16:
            continue
        im = ids[ann['image_id']]
        identifier = str(ann['id'])
        image = (Path(a.images)/im['file_name']).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        candidates.append(dict(id=identifier, image=str(image), bbox=ann['bbox']))
        labels[identifier] = categories[ann['category_id']]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Prompt author must inspect images without labels; never auto-fill from truth.
    manifest = dict(model='openai/clip-vit-base-patch32', revision='REPLACE_WITH_IMMUTABLE_COMMIT',
        vocabulary=[categories[k] for k in sorted(categories)], candidates=candidates,
        prompts=dict(general='AUTHOR_BEFORE_EVALUATION', detailed='AUTHOR_BEFORE_EVALUATION', misleading='AUTHOR_BEFORE_EVALUATION'))
    (out/'manifest.template.json').write_text(json.dumps(manifest, indent=2))
    (out/'truth.json').write_text(json.dumps(dict(labels=labels), indent=2))
    (out/'images.json').write_text(json.dumps(images, indent=2))
if __name__ == '__main__':
    main()
