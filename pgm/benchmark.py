"""Single warm replay of classification/graph runtime, excluding disk I/O and encoders."""
import json
import time
from pathlib import Path
from .pilot import ROOT, CONDITIONS, METHODS, contextual_prediction
from .core import Candidate, classify, graph

def main():
    folder=ROOT/'results/coco24'
    images=json.loads((ROOT/'inputs/coco24/images.json').read_text())
    sizes={im['file_name']:(im['width'],im['height']) for im in images}
    results={}
    for method in METHODS:
        results[method]={}
        for condition in CONDITIONS:
            graphs=json.loads((folder/method/(condition+'.json')).read_text())
            start=time.perf_counter()
            for g in graphs:
                vocabulary=list(g['nodes'][0]['visual_scores'])
                cs=[Candidate(n['id'],n['observation']['image'],tuple(n['observation']['bbox'])) for n in g['nodes']]
                ps=[]
                for n in g['nodes']:
                    base=list(n['visual_scores'].values())
                    if method=='literal_prior':ps.append(classify(base,vocabulary,g['description']))
                    else:ps.append(contextual_prediction(base,list(n['conditioned_scores'].values()),vocabulary))
                graph(cs,ps,{c.image:sizes[Path(c.image).name] for c in cs},g['description'])
            results[method][condition]=time.perf_counter()-start
    (folder/'postprocess_runtime.json').write_text(json.dumps(dict(seconds=results,
        notes='Single warm pass, disk and encoder excluded. Actual image sizes from selected metadata; replay graphs discarded.'),indent=2)+'\n')
    print(results)
if __name__=='__main__':main()
