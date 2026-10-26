"""Evaluate only explicitly reviewed candidate pairs; no unlabelled negatives."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def audit(graphs, annotation):
    edges={(min(e['source'],e['target']),max(e['source'],e['target'])) for g in graphs for e in g['edges'] if e['relation']=='near'}
    tp=fp=fn=tn=unknown=0
    records=[]
    for pair in annotation['pairs']:
        label=pair['near']
        predicted=(min(pair['source'],pair['target']),max(pair['source'],pair['target'])) in edges
        records.append(dict(**pair,predicted_near=predicted))
        if label is None:
            unknown+=1
        elif label and predicted:tp+=1
        elif label:fn+=1
        elif predicted:fp+=1
        else:tn+=1
    n=tp+fp+fn+tn
    return dict(n=n,unknown=unknown,tp=tp,fp=fp,fn=fn,tn=tn,
        accuracy=(tp+tn)/n if n else None,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        always_near_accuracy=(tp+fn)/n if n else None,pairs=records,
        scope='Assistant-labelled 2D region adjacency audit; not independent human benchmark or physical relation truth')

def main():
    folder=ROOT/'results/coco24'
    annotation=json.loads((ROOT/'inputs/relation_audit.json').read_text())
    results={}
    for method in ('literal_prior','contextual_text'):
        results[method]={}
        for condition in ('none','general','detailed','misleading'):
            graphs=json.loads((folder/method/(condition+'.json')).read_text())
            results[method][condition]=audit(graphs,annotation)
    (folder/'relation_audit_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results['literal_prior']['none'].items() if k!='pairs'},indent=2))
if __name__=='__main__':
    main()
