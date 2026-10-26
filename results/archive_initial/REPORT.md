# Research report — 2026-10-26

## Current conclusion

Whether short scene descriptions provide a measurable benefit remains **undetermined**. A modular Python baseline and executed synthetic checks exist. No real visual experiment completed: torch and dataset retrieval failed in the sandbox (proxy connection/operation-not-permitted); the subsequent additional-network attempt was interrupted. No pretrained model was loaded. No fabricated visual accuracy, relation accuracy or runtime is reported. `results/experiment_status.json` records this boundary.

## Prior work and research opportunity

This is a targeted primary-source review, not a systematic or exhaustive novelty search. Broad novelty for language-guided semantic mapping or context-assisted recognition is not supported:

- Torralba et al., Context-Based Vision System for Place and Object Recognition (ICCV 2003), already uses scene/place information as recognition priors. Relevant distinction: this prototype supplies external text rather than estimating all context from visual appearance. https://dspace.mit.edu/entities/publication/9a3cb422-588d-4b0b-adfc-fcc75df29612
- Radford et al., Learning Transferable Visual Models From Natural Language Supervision (ICML 2021), establishes transferable image/text matching and zero-shot classification. CLIP crop recognition supplies a reproducible baseline; it does not itself prove benefits from a scene narrative. https://arxiv.org/abs/2103.00020
- Li et al., Grounded Language-Image Pre-Training (CVPR 2022), unifies object detection and phrase grounding. Language-conditioned region recognition is established. https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html
- Liu et al., Grounding DINO (ECCV 2024), supports category names and referring expressions for open-set detection. It is a future proposal backend, but changing proposals with prompts would confound the current classification comparison. https://arxiv.org/abs/2303.05499
- Peng et al., OpenScene (CVPR 2023), co-embeds 3D features with text/image features for open-vocabulary scene understanding. This prototype does not reproduce 3D mapping. https://arxiv.org/abs/2211.15654
- Gu et al., ConceptGraphs (ICRA 2024), combines 2D foundation models with multiview association into open-vocabulary 3D graphs for planning. Graph-based language-accessible maps are established. https://concept-graphs.github.io/
- Chang et al., Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs (CoRL 2023), grounds free-form queries using relational context. This is particularly close prior art and rules out a broad claim of novel context-aware graph grounding. https://arxiv.org/abs/2309.15940
- Krishna et al., Visual Genome (IJCV 2017), provides independent objects, attributes and relations suitable for later scene-graph evaluation. https://arxiv.org/abs/1602.07332

A defensible potential thesis contribution is a controlled empirical study of **external scene-description reliability versus selective recognition risk**, with fixed observations, proposals and recognition backbone. This is an inference from the reviewed work, not a verified literature gap. The precise overlap with recent prompt reliability and contextual recognition studies still needs a systematic search and supervisor review before any novelty claim.

## Executed work

Seven standard-library unit tests passed: visual evidence floor, bounded prompt influence, ambiguity abstention, token boundaries, region references and conservative relations, invalid geometry, and undefined selective risk. Four fixture graphs and metrics were generated from explicitly invented cosine scores. The fixture demonstrates serialization and abstention paths only; it is not an experiment on semantic recognition. The optional CLIP backend is implemented but unvalidated in this environment.

## Limitations and next steps

Literal mention bonuses are a narrow context baseline: `industrial workshop` alone implies no machine prior in this implementation. There is no natural-language relation parser, learned contextual inference, physical part_of detector, calibrated probability, unknown-object model or persistent multiview map. COCO classes do not cover industrial controls adequately. Oracle boxes, crop contamination, nonrandom public-image sampling, possible CLIP pretraining overlap, human description supervision and image-domain mismatch restrict external validity. Real visual generalization to unfamiliar environments cannot be inferred from a public benchmark alone.

Next: obtain the specified data/weights; freeze per-scene descriptions independently of labels; run the four-condition pilot; analyze paired scene-level uncertainty and harms; add a calibrated context embedding method only if justified. Add independent relation labels or a Visual Genome subset before evaluating relations. Extend to real workshop images, part annotations, automatic proposals and 3D association only after evidence supports those investments. Preserve all unsuccessful conditions and previous negative findings.

## Dependencies and licenses

Python standard library core; optional Pillow (HPND), NumPy (BSD-3-Clause), torch and torchvision (BSD-style licenses), Transformers (Apache-2.0). CLIP source and released model are covered by the OpenAI CLIP repository's MIT license; retain its license and consult the model card's intended-use restrictions and biases: https://github.com/openai/CLIP and https://huggingface.co/openai/clip-vit-base-patch32 . Model ID alone is not a reproducibility pin: the visual run requires an immutable commit. Dependency versions are proposed pins, not a successfully installed lock; save the resolved environment after installation. Dataset licenses are separate; see DATASET.md.
