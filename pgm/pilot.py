"""Frozen COCO24 inference, followed by a separate evaluation stage."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from .core import Candidate, classify, graph
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('none','general','detailed','misleading')
METHODS = ('literal_prior','contextual_text')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def contextual_prediction(base, conditioned, vocabulary):
    # Fixed exploratory method: 25% contextual embedding score, bounded +/-0.05.
    fused = [s + max(-.05,min(.05,.25*(c-s))) if s >= .18 else s
             for s,c in zip(base,conditioned)]
    p = classify(fused,vocabulary)
    winner = vocabulary.index(p['hypothesis'])
    # Visual floor remains based on unconditioned evidence.
    if base[winner] < .18:
        p.update(category=None,abstained=True)
    p.update(visual_cosine=base[winner], visual_scores=dict(zip(vocabulary,base)),
             conditioned_scores=dict(zip(vocabulary,conditioned)),
             decision_scores=dict(zip(vocabulary,fused)), method='contextual_text')
    return p

def infer(output):
    import numpy as np
    from .model import ClipMatcher
    manifest_path = ROOT/'inputs/coco24/manifest.template.json'
    prompts_path = ROOT/'inputs/prompts.json'
    manifest = json.loads(manifest_path.read_text())
    pin = json.loads((ROOT/'data/model_revision.json').read_text())
    prompts = json.loads(prompts_path.read_text())
    cs = [Candidate(c['id'], str(ROOT/'data/val2017'/Path(c['image']).name),tuple(c['bbox'])) for c in manifest['candidates']]
    vocabulary = manifest['vocabulary']
    start = time.perf_counter()
    backend = ClipMatcher(pin['model'], pin['revision'], str(ROOT/'models'))
    load_seconds = time.perf_counter()-start
    start = time.perf_counter()
    visual, sizes = backend.image_features(cs)
    visual_seconds = time.perf_counter()-start
    start = time.perf_counter()
    label_text = ['a photo of a '+v for v in vocabulary]
    base_text = backend.text_features(label_text)
    base_scores = (visual @ base_text.T).tolist()
    base_text_seconds = time.perf_counter()-start
    by_image = {}
    for i,c in enumerate(cs):
        by_image.setdefault(str(int(Path(c.image).stem)),[]).append(i)
    graphs = {method:{condition:[] for condition in CONDITIONS} for method in METHODS}
    timing = {}
    for scene, indices in by_image.items():
        scene_cs = [cs[i] for i in indices]
        for condition in CONDITIONS:
            prompt = '' if condition=='none' else prompts[scene][condition]
            start = time.perf_counter()
            if condition=='none':
                conditioned = [base_scores[i] for i in indices]
            else:
                ctx = backend.text_features([text+'. Scene: '+prompt for text in label_text])
                conditioned = (visual[indices] @ ctx.T).tolist()
            elapsed = time.perf_counter()-start
            timing.setdefault(condition,[]).append(elapsed)
            for method in METHODS:
                predictions=[]
                for k,i in enumerate(indices):
                    base = base_scores[i]
                    if method=='literal_prior':
                        p = classify(base,vocabulary,prompt)
                        p['method']=method
                    elif condition=='none':
                        p = contextual_prediction(base,base,vocabulary)
                    else:
                        p = contextual_prediction(base,conditioned[k],vocabulary)
                    predictions.append(p)
                g=graph(scene_cs,predictions,sizes,prompt)
                g.update(scene_id=scene,condition=condition,method=method,synthetic_fixture=False)
                graphs[method][condition].append(g)
        print('Inferred scene',scene,'regions',len(indices),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    for method,conditions in graphs.items():
        folder=output/method
        folder.mkdir(exist_ok=True)
        for condition, values in conditions.items():
            (folder/(condition+'.json')).write_text(json.dumps(values,indent=2)+'\n')
    np.savez_compressed(output/'features.npz',visual=visual.numpy(),label_text=base_text.numpy())
    metadata=dict(model=pin, load_seconds=load_seconds,visual_feature_seconds=visual_seconds,
        baseline_text_seconds=base_text_seconds,context_text_seconds={k:sum(v) for k,v in timing.items()},
        manifest_sha256=digest(manifest_path),prompts_sha256=digest(prompts_path),
        python=platform.python_version(),platform=platform.platform(),
        processor=platform.processor(),threads=2,seed=0,
        n_scenes=len(by_image),n_candidates=len(cs),vocabulary=vocabulary,
        notes='CPU, locally cached model; contextual text timing includes tokenization and encoding. Graph construction and serialization excluded.')
    (output/'inference_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

def summarize(output):
    import numpy as np
    truth=json.loads((ROOT/'inputs/coco24/truth.json').read_text())
    all_results={}
    for method in METHODS:
        conditions={k:json.loads((output/method/(k+'.json')).read_text()) for k in CONDITIONS}
        baseline={g['scene_id']:g for g in conditions['none']}
        method_results={}
        for condition,graphs in conditions.items():
            flat=dict(nodes=[n for g in graphs for n in g['nodes']],edges=[e for g in graphs for e in g['edges']])
            metrics=evaluate(flat,truth)
            per_scene=[]
            transitions=dict(correct_to_wrong=0,wrong_to_correct=0,accepted_to_abstained=0,abstained_to_accepted=0,category_changes=0)
            classes={}
            for g in graphs:
                base=baseline[g['scene_id']]
                st=dict(labels={n['id']:truth['labels'][n['id']] for n in g['nodes']})
                a=evaluate(g,st)
                b=evaluate(base,st)
                per_scene.append(dict(scene_id=g['scene_id'],**a,baseline_accuracy=b['accuracy'],delta=a['accuracy']-b['accuracy']))
                for n,old in zip(g['nodes'],base['nodes']):
                    actual=truth['labels'][n['id']]
                    correct=n['category']==actual
                    old_correct=old['category']==actual
                    transitions['correct_to_wrong']+=int(old_correct and not n['abstained'] and not correct)
                    transitions['wrong_to_correct']+=int(not old['abstained'] and not old_correct and correct)
                    transitions['accepted_to_abstained']+=int(not old['abstained'] and n['abstained'])
                    transitions['abstained_to_accepted']+=int(old['abstained'] and not n['abstained'])
                    transitions['category_changes']+=int(old['category']!=n['category'])
                    record=classes.setdefault(actual,dict(n=0,correct=0))
                    record['n']+=1;record['correct']+=int(correct)
            delta=np.array([s['delta'] for s in per_scene])
            rng=np.random.default_rng(0)
            means=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
            metrics.update(coverage=metrics['accepted']/metrics['n'],
                scene_mean_accuracy=float(np.mean([s['accuracy'] for s in per_scene])),
                paired_scene_mean_delta=float(delta.mean()),
                paired_scene_bootstrap_95ci=np.quantile(means,[.025,.975]).tolist(),
                class_balanced_accuracy=float(np.mean([r['correct']/r['n'] for r in classes.values()])),
                transitions=transitions,category_change_rate=transitions['category_changes']/metrics['n'],
                per_class=classes,per_scene=per_scene)
            method_results[condition]=metrics
        all_results[method]=method_results
    (output/'metrics.json').write_text(json.dumps(all_results,indent=2)+'\n')
    for method,conditions in all_results.items():
        for condition,m in conditions.items():
            print(method,condition,'accuracy',round(m['accuracy'],4),'risk',round(m['wrong_target_risk'],4),'abstain',round(m['abstention_rate'],4),'scene delta CI',m['paired_scene_bootstrap_95ci'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['infer','evaluate','all'],default='all')
    p.add_argument('--out',type=Path,default=ROOT/'results/coco24')
    a=p.parse_args()
    if a.stage in ('infer','all'):
        if (a.out/'inference_metadata.json').exists():
            raise FileExistsError('Use a new output folder to preserve existing runs')
        infer(a.out)
    if a.stage in ('evaluate','all'):
        summarize(a.out)
if __name__=='__main__':
    main()
