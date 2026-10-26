import unittest
from pgm.core import Candidate, classify, graph, interpret
from pgm.evaluate import evaluate

class Tests(unittest.TestCase):
    def test_no_prompt_objects(self):
        p = classify([.1,.12], ['button','machine'], 'machine')
        self.assertTrue(p['abstained'])
    def test_context_bounded(self):
        self.assertEqual(classify([.4,.2], ['cat','dog'], 'dog')['category'], 'cat')
    def test_ambiguity(self):
        self.assertTrue(classify([.3,.3], ['cat','dog'])['abstained'])
    def test_literal_interpretation(self):
        self.assertEqual(interpret('Not a hotdog', ['dog']), [])
    def test_regions_and_no_part_of(self):
        cs = [Candidate('a','x',(0,0,10,10)),Candidate('b','x',(1,1,2,2))]
        pred = [classify([.4,.1],['cat','dog'])]*2
        g = graph(cs,pred,{'x':(100,100)},'machine')
        self.assertEqual(len(g['nodes']),2)
        self.assertTrue(all(e['relation']=='near' for e in g['edges']))
        self.assertIsNone(evaluate(g, {'labels':{'a':'cat','b':'dog'}})['relation_precision'])
    def test_invalid(self):
        with self.assertRaises(ValueError):
            Candidate('a','x',(0,0,-1,1))
    def test_empty_acceptance(self):
        g = graph([Candidate('a','x',(0,0,1,1))],[classify([.1,.1],['a','b'])],{'x':(2,2)},'')
        self.assertIsNone(evaluate(g,{'labels':{'a':'a'}})['wrong_target_risk'])
if __name__ == '__main__':
    unittest.main()
