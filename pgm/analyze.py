"""Post-pilot descriptive coverage analysis; never selects a deployment threshold."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'results/coco24'
    truth=json.loads((ROOT/'inputs/coco24/truth.json').read_text())['labels']
    result={}
    for method in ('literal_prior','contextual_text'):
        conditions={}
        for condition in ('none','general','detailed','misleading'):
            graphs=json.loads((folder/method/(condition+'.json')).read_text())
            nodes=[n for g in graphs for n in g['nodes']]
            conditions[condition]=nodes
        k=sum(not n['abstained'] for n in conditions['none'])
        result[method]={}
        for condition,nodes in conditions.items():
            eligible=[n for n in nodes if n['visual_cosine']>=.18]
            selected=sorted(eligible,key=lambda n:(-n['score_margin'],n['id']))[:k]
            correct=sum(n['hypothesis']==truth[n['id']] for n in selected)
            curve=[]
            for margin in (0,.01,.02,.04):
                accepted=[n for n in eligible if n['score_margin']>=margin]
                right=sum(n['hypothesis']==truth[n['id']] for n in accepted)
                curve.append(dict(margin=margin,accepted=len(accepted),correct=right,
                    coverage=len(accepted)/len(nodes),accuracy=right/len(nodes),
                    wrong_target_risk=1-right/len(accepted) if accepted else None))
            result[method][condition]=dict(forced_hypothesis_accuracy=sum(n['hypothesis']==truth[n['id']] for n in nodes)/len(nodes),
                matched_coverage=dict(k=k,correct=correct,wrong=len(selected)-correct,
                    wrong_target_risk=1-correct/len(selected),coverage=len(selected)/len(nodes)),
                descriptive_margin_curve=curve)
    (folder/'coverage_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for method,conditions in result.items():
        for condition,v in conditions.items():
            print(method,condition,'top1',round(v['forced_hypothesis_accuracy'],4),'matched',v['matched_coverage'])
if __name__=='__main__':main()
