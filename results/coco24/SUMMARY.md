# Executed COCO24 pilot results

24 images, 114 supplied regions, fixed CLIP ViT-B/32. Accuracy counts abstentions as unsuccessful predictions. Wrong-target risk is wrong labels / accepted labels.

| Method | Context | Correct / 114 | Accepted | Accuracy | Wrong-target risk | Abstention |
|---|---|---:|---:|---:|---:|---:|
| literal_prior | none | 27 | 35 | 23.68% | 22.86% | 69.30% |
| literal_prior | general | 27 | 35 | 23.68% | 22.86% | 69.30% |
| literal_prior | detailed | 38 | 54 | 33.33% | 29.63% | 52.63% |
| literal_prior | misleading | 27 | 35 | 23.68% | 22.86% | 69.30% |
| contextual_text | none | 27 | 35 | 23.68% | 22.86% | 69.30% |
| contextual_text | general | 28 | 35 | 24.56% | 20.00% | 69.30% |
| contextual_text | detailed | 26 | 32 | 22.81% | 18.75% | 71.93% |
| contextual_text | misleading | 27 | 34 | 23.68% | 20.59% | 70.18% |
