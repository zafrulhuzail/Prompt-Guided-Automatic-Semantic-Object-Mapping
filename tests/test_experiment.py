import unittest
from pgm.core import Candidate, graph, classify
from pgm.pilot import contextual_prediction
from pgm.relation_audit import audit
from pgm.evaluate import evaluate

class ExperimentTests(unittest.TestCase):
    def test_contextual_none_equals_original(self):
        scores=[.35,.22,.1]
        old=classify(scores,['a','b','c'])
        new=contextual_prediction(scores,scores,['a','b','c'])
        for key in old:self.assertEqual(old[key],new[key])
    def test_context_cannot_create_visual_evidence(self):
        p=contextual_prediction([.1,.12],[.9,.99],['a','b'])
        self.assertTrue(p['abstained'])
        self.assertEqual(p['visual_scores'],{'a':.1,'b':.12})
    def test_audit_ignores_unreviewed_pairs(self):
        g=[{'edges':[{'source':'a','target':'b','relation':'near'}, {'source':'c','target':'d','relation':'near'}]}]
        labels={'pairs':[{'source':'b','target':'a','near':True},{'source':'c','target':'e','near':None}]}
        result=audit(g,labels)
        self.assertEqual(result['tp'],1)
        self.assertEqual(result['fp'],0)
        self.assertEqual(result['unknown'],1)
    def test_evaluation_requires_exact_ids(self):
        with self.assertRaises(ValueError):
            evaluate({'nodes':[]},{'labels':{'extra':'cat'}})
    def test_no_cross_image_relation(self):
        cs=[Candidate('a','x',(0,0,2,2)),Candidate('b','y',(0,0,2,2))]
        p=classify([.4,.2],['cat','dog'])
        self.assertEqual(graph(cs,[p,p],{'x':(10,10),'y':(10,10)},'')['edges'],[])
    def test_contextual_adjustment_bounded(self):
        p=contextual_prediction([.4,.2],[1,-1],['a','b'])
        self.assertAlmostEqual(p['decision_scores']['a'],.45)
        self.assertAlmostEqual(p['decision_scores']['b'],.15)
if __name__=='__main__':unittest.main()
