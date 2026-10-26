import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from .core import Candidate, classify, graph
from .evaluate import evaluate

CONDITIONS = ('none','general','detailed','misleading')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--truth')
    p.add_argument('--out', required=True)
    p.add_argument('--fixture', action='store_true', help='Synthetic score plumbing demonstration only')
    args = p.parse_args()
    m = json.loads(Path(args.manifest).read_text())
    if not args.fixture and (m.get('revision', '').startswith('REPLACE') or any('AUTHOR_BEFORE' in v for v in m['prompts'].values())):
        raise ValueError('Pin the model commit and author prompts before inference')
    cs = [Candidate(c['id'], c['image'], tuple(c['bbox'])) for c in m['candidates']]
    start = time.perf_counter()
    if args.fixture:
        scores, sizes = m['fixture_scores'], m['image_sizes']
    else:
        from .model import ClipMatcher
        backend = ClipMatcher(m['model'], m['revision'])
        scores, sizes = backend.scores(cs, m['vocabulary'])
    feature_seconds = time.perf_counter()-start
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    baseline = None
    for condition in CONDITIONS:
        start = time.perf_counter()
        description = '' if condition == 'none' else m['prompts'][condition]
        predictions = [classify(row, m['vocabulary'], description) for row in scores]
        g = graph(cs, predictions, sizes, description)
        if baseline is None:
            baseline = predictions
        flips = sum(a['category'] != b['category'] for a, b in zip(baseline, predictions))
        g['condition'] = condition
        g['synthetic_fixture'] = args.fixture
        (out/(condition+'.json')).write_text(json.dumps(g, indent=2)+'\n')
        metrics[condition] = dict(postprocess_seconds=time.perf_counter()-start, category_change_rate=flips/len(cs) if cs else None)
        if args.truth:
            metrics[condition].update(evaluate(g, json.loads(Path(args.truth).read_text())))
    metrics['provenance'] = dict(synthetic_fixture=args.fixture, python=platform.python_version(),
        feature_seconds_shared=feature_seconds, manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        model=None if args.fixture else m['model'], revision=None if args.fixture else m['revision'])
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
if __name__ == '__main__':
    main()
