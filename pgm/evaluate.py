"""Evaluation boundary: labels enter only here, after inference."""
def evaluate(graph, truth):
    expected = truth['labels']
    nodes = graph['nodes']
    if set(expected) != {n['id'] for n in nodes}:
        raise ValueError('Evaluation IDs must match exactly')
    total = len(nodes)
    accepted = [n for n in nodes if not n['abstained']]
    correct = sum(n['category'] == expected[n['id']] for n in accepted)
    wrong = len(accepted)-correct
    result = dict(n=total, accepted=len(accepted), correct=correct, wrong=wrong,
                  accuracy=correct/total if total else None,
                  abstention_rate=1-len(accepted)/total if total else None,
                  wrong_target_risk=wrong/len(accepted) if accepted else None,
                  wrong_per_candidate=wrong/total if total else None,
                  relation_precision=None, relation_recall=None)
    # Requires independently annotated, exhaustive near edges. Absent != empty.
    if 'relations' in truth:
        actual = {(min(e['source'],e['target']), max(e['source'],e['target']),e['relation']) for e in graph['edges']}
        target = {(min(e['source'],e['target']), max(e['source'],e['target']),e['relation']) for e in truth['relations']}
        result.update(relation_precision=len(actual & target)/len(actual) if actual else None,
                      relation_recall=len(actual & target)/len(target) if target else None)
    return result
