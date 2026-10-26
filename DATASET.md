# Dataset provenance and setup

Selected pilot: deterministic small subset of **COCO 2017 validation**, real Flickr photographs with independent human instance categories, boxes and segmentation annotations. The dataset itself is large; the pilot selects the first 24 image IDs sorted numerically, before examining predictions. Fixed eligibility: noncrowd boxes at least 16 pixels in width and height; no balancing or filtering on model success. For calibration and later confirmatory work use disjoint image IDs and scenes.

Primary sources:
- Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014: https://arxiv.org/abs/1405.0312
- Official downloads: https://cocodataset.org/#download
- Official terms: https://cocodataset.org/#termsofuse
- Annotations archive: https://images.cocodataset.org/annotations/annotations_trainval2017.zip
- Images: https://images.cocodataset.org/zips/val2017.zip

Download and unpack these under data/. For a smaller image download, use the selected image metadata's coco_url (HTTPS) and file_name rather than downloading the full image ZIP. `pgm.prepare` requires the images to be present. It outputs selected original image metadata including source URLs and license IDs. Record download date and SHA256 hashes of source archives, images, completed manifests and model files alongside results. Do not commit data unless its image-specific license permits redistribution.

COCO annotations are CC BY 4.0; images have individual Flickr licenses, linked from the original annotation file's `licenses` table. Retain attribution and resolve each image's license ID against that table before redistributing. The executed pilot retrieved the official annotation archive and 24 original images on 2026-10-26. A mirror copy of the instance JSON was also downloaded and verified byte-identical to the official archive member (SHA256 e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f). Official bytes were retained. Image URLs, Flickr URLs, license IDs, hashes and the complete license lookup are retained in data/provenance.json and results/coco24/provenance/.

The converter strips category labels from inference candidates and writes them into a separate truth.json. Boxes are **oracle proposals from human annotations**, an explicit localization advantage. This isolates classification; do not report detection recall or automatic discovery. Masks are available in the upstream dataset but this baseline uses rectangular crops, which may contain background or neighboring objects. The fixed vocabulary includes all 80 COCO categories, not only classes present in the selected scenes.

COCO has no independent near/part_of ground truth. Relation metrics must remain null for this dataset. Do not treat box-derived relations as evaluation truth. For a second stage consider Visual Genome, whose independent object and relationship annotations support scene-graph evaluation, but require synonym normalization, image-disjoint splits, missing-label handling and a separately specified predicate ontology: https://arxiv.org/abs/1602.07332 . Part-whole and spatial relations need separate auditing; containment alone does not establish part_of.

## Executed subset

24 images, 114 noncrowd eligible boxes, 36 represented evaluation classes from the fixed 80-class vocabulary. Selected IDs: 139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584, 1675, 1761, 1818, 1993, 2006, 2149. No captions or category labels were used to author descriptions. Object labels entered only after visual predictions were persisted. Region proposals are human annotations, not outputs of an automatic detector.

A separate assistant-authored visual audit labels 22 deterministic region pairs for apparent adjacency/overlap (20 resolved, two uncertain). These are not public human semantic-relation annotations. COCO relation precision/recall against independent benchmark truth remains null; the audit has its own clearly scoped output and cannot support claims about physical relation accuracy.
