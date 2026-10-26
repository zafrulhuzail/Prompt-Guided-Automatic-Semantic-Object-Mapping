"""Inference only: never accepts evaluation annotations."""
import math
import re
from dataclasses import dataclass
from itertools import combinations

@dataclass(frozen=True)
class Candidate:
    id: str
    image: str
    bbox: tuple  # x, y, width, height in pixels

    def __post_init__(self):
        if len(self.bbox) != 4 or not all(math.isfinite(v) for v in self.bbox):
            raise ValueError('Invalid box')
        if min(self.bbox) < 0 or min(self.bbox[2:]) <= 0:
            raise ValueError('Invalid box extent')


def interpret(description, vocabulary):
    """Conservative explicit category mentions; no LLM or hidden ontology."""
    words = set(re.findall(r'[a-z]+', description.lower()))
    return [label for label in vocabulary if set(re.findall(r'[a-z]+', label.lower())) <= words]


def classify(scores, vocabulary, description='', weight=0.02, floor=0.18, margin=0.02):
    if len(scores) != len(vocabulary) or len(scores) < 2:
        raise ValueError('Need one score per label and at least two labels')
    if not all(math.isfinite(x) for x in scores):
        raise ValueError('Non-finite score')
    if not 0 <= weight <= 0.05:
        raise ValueError('Context weight must be bounded')
    mentions = interpret(description, vocabulary)
    # Context cannot rescue a category below the visual evidence floor.
    fused = [s + (weight if label in mentions and s >= floor else 0)
             for s, label in zip(scores, vocabulary)]
    order = sorted(range(len(scores)), key=lambda i: fused[i], reverse=True)
    winner, runner = order[:2]
    gap = fused[winner] - fused[runner]
    abstain = scores[winner] < floor or gap < margin
    return dict(category=None if abstain else vocabulary[winner],
                hypothesis=vocabulary[winner], abstained=abstain,
                visual_cosine=scores[winner], score_margin=gap,
                confidence_type='uncalibrated cosine and margin; not probability',
                visual_scores=dict(zip(vocabulary, scores)), context_mentions=mentions)


def relations(candidates, image_sizes):
    """2D center-distance near hypotheses; no physical part_of from box containment."""
    edges = []
    for a, b in combinations(candidates, 2):
        if a.image != b.image:
            continue
        w, h = image_sizes[a.image]
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        distance = math.hypot(ax + aw/2 - bx - bw/2, ay + ah/2 - by - bh/2)/math.hypot(w, h)
        if distance <= .2:
            edges.append(dict(source=a.id, target=b.id, relation='near',
                              status='hypothesis', evidence='2d_center_distance',
                              normalized_distance=distance, confidence=None))
    return edges


def graph(candidates, predictions, image_sizes, description):
    if len(candidates) != len(predictions) or len({c.id for c in candidates}) != len(candidates):
        raise ValueError('Candidate alignment or duplicate IDs')
    return dict(schema_version=1, scope='supplied-region semantic grounding; 2D',
                description=description,
                nodes=[dict(id=c.id, observation=dict(image=c.image, bbox=list(c.bbox)), **p)
                       for c, p in zip(candidates, predictions)],
                edges=relations(candidates, image_sizes))
