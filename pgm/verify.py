"""Verify observations, input provenance and graph consistency without model downloads."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'results/coco24'
    meta=json.loads((folder/'inference_metadata.json').read_text())
    for path,key in [(ROOT/'inputs/coco24/manifest.template.json','manifest_sha256'),(ROOT/'inputs/prompts.json','prompts_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[key],str(path)
    source=json.loads((ROOT/'data/provenance.json').read_text())
    for record in source['images']:
        assert hashlib.sha256((ROOT/record['file']).read_bytes()).hexdigest()==record['sha256']
    candidates=json.loads((ROOT/'inputs/coco24/manifest.template.json').read_text())['candidates']
    truth=json.loads((ROOT/'inputs/coco24/truth.json').read_text())['labels']
    assert {c['id'] for c in candidates}==set(truth)
    references={c['id']:c for c in candidates}
    for method in ('literal_prior','contextual_text'):
        baseline=None
        for condition in ('none','general','detailed','misleading'):
            graphs=json.loads((folder/method/(condition+'.json')).read_text())
            nodes=[n for g in graphs for n in g['nodes']]
            assert len(nodes)==114 and len({n['id'] for n in nodes})==114
            for n in nodes:
                assert n['observation']['bbox']==references[n['id']]['bbox']
                assert n['category'] is None if n['abstained'] else n['category'] in n['visual_scores']
                assert len(n['visual_scores'])==80
                assert 'label' not in n and 'ground_truth' not in n
            current={n['id']:n['visual_scores'] for n in nodes}
            if baseline is None:baseline=current
            assert baseline==current
    print('Verified image hashes, frozen inputs, 8 conditions, 114 region references each and identical visual scores.')
if __name__=='__main__':main()
