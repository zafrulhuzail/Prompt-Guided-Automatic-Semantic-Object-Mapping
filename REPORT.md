# Research report — completed exploratory pilot, 2026-10-26

## Conclusion

**Detailed object-naming descriptions can change selective recognition usefully, but this pilot does not establish a reliable benefit from general scene descriptions.** With fixed oracle regions and CLIP visual evidence, the literal-prior method accepted 11 more correct labels and eight more incorrect labels under detailed context. The full-description embedding method changed very little. The simple relation heuristic failed its small visual audit. These results do not justify Unity integration yet.

The real-image experiments have now executed successfully. Earlier download failures and the initial incomplete report are retained in `results/archive_initial/`; no previous negative findings or fixture results were removed. All code, data setup, raw predictions and analyses remain isolated here.

## Executed design

Twenty-four COCO 2017 validation images selected by ascending image ID, 114 noncrowd boxes at least 16x16 pixels, fixed vocabulary of 80 classes (36 represented). Proposals are oracle human boxes, with labels removed from inference inputs. This is supplied-region semantic grounding, not automatic detection, multiview instance association or 3D semantic mapping. RGB crops may include background/neighbor objects.

CLIP ViT-B/32 runs locally on CPU at immutable revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Each crop embedding is computed once and reused. Text template is `a photo of a {category}`. Four conditions: none, general scene type, detailed image description and intentionally misleading scene description. Descriptions were authored by the assistant from images before category labels or model outputs were consulted and frozen with a hash. They are not independent human descriptions; detailed descriptions supply extra semantic information. No reference captions were used.

Two fixed methods:

1. **literal_prior**: original implementation, adds 0.02 to explicitly mentioned category scores, only above the visual floor. General descriptions frequently have no recognized object mentions. This baseline cannot infer workshop → machine and does not understand negation or synonyms.
2. **contextual_text**, exploratory extension: same CLIP model embeds each category template followed by the scene description. Add 0.25 times the conditioned-minus-base cosine, bounded to +/-0.05, to eligible visual scores. No test-label tuning.

Both abstain below baseline visual cosine 0.18 or decision-score margin 0.02. Scores are uncalibrated, not probabilities. Context never creates candidate nodes or expands the vocabulary, but can still produce wrong labels for observed regions. The inference CLI writes all raw graphs before the evaluation stage reads truth.json.

## Classification and abstention results

Accuracy below is accepted correct / all 114 candidates, counting abstention as unsuccessful. Wrong-target risk means wrong category / accepted category, a semantic-assignment proxy rather than robot action or instance-retrieval risk.

| Method | Context | Correct | Accepted | Accuracy | Wrong-target risk | Abstention |
|---|---|---:|---:|---:|---:|---:|
| Literal prior | None | 27 | 35 | 23.68% | 22.86% | 69.30% |
| Literal prior | General | 27 | 35 | 23.68% | 22.86% | 69.30% |
| Literal prior | Detailed | 38 | 54 | 33.33% | 29.63% | 52.63% |
| Literal prior | Misleading | 27 | 35 | 23.68% | 22.86% | 69.30% |
| Contextual text | None | 27 | 35 | 23.68% | 22.86% | 69.30% |
| Contextual text | General | 28 | 35 | 24.56% | 20.00% | 69.30% |
| Contextual text | Detailed | 26 | 32 | 22.81% | 18.75% | 71.93% |
| Contextual text | Misleading | 27 | 34 | 23.68% | 20.59% | 70.18% |

The detailed literal-prior improvement is entirely from 19 abstained-to-accepted transitions (11 correct, eight wrong). There are no accepted wrong-to-correct transitions. Thus the gain at the default threshold is mainly a coverage change, not a demonstrated repair of accepted misclassifications. Unweighted scene-mean paired gain is +9.79 percentage points; a 10,000-draw scene-cluster bootstrap (seed 0) gives a descriptive 95% interval of [+2.20, +19.17] points. This is exploratory, not a confirmatory significance claim. Candidate-weighted gain is +9.65 points.

General contextual-text gain is only one correct candidate (+0.88 candidate-weighted points); scene-mean gain is +0.42 points with interval [0, +1.25]. Detailed contextual-text scene-mean change is -0.60 points with interval [-1.79, 0]. The primary general-vs-none comparison does not establish a clear benefit.

At matched acceptance of 35 regions, ranked by margin without using labels, detailed literal context has 30 correct / five wrong versus baseline 27 correct / eight wrong. This is a post-pilot descriptive ranking analysis, not a tuned deployment threshold or independent validation. Margin curves and per-class/per-scene results are included as raw JSON. Baseline forced-hypothesis accuracy before abstention is 50.00%; detailed literal context gives 51.75%. High abstention is a substantial baseline weakness; thresholds need held-out calibration rather than test-set optimization.

Misleading context leaves literal-prior accepted accuracy unchanged, but forced-hypothesis accuracy falls from 50.00% to 46.49%; contextual text falls to 47.37%. Unchanged accepted accuracy therefore does **not** prove prompt robustness. The abstention rule hides some changes in low-margin hypotheses. These deliberately mismatched kitchen/snow prompts are limited stress tests; plausible confusers, negation, multiple authors and unknown categories remain untested.

## Relations

`near` uses 2D center distance <=0.2 image diagonal. It is prompt-independent and emitted as a hypothesis; physical proximity and `part_of` are not inferred. Public independent relation annotations are absent from COCO, so the main semantic relation metrics remain null.

A separate assistant visual audit of the first two eligible regions in each eligible image covers 22 pairs: 20 resolved and two uncertain. Against subjective apparent adjacency/overlap judgments, the heuristic has TP=6, FP=2, FN=10, TN=2: precision 75.0%, recall 37.5%, accuracy 40.0%. Always predicting near achieves 80.0% accuracy on this imbalanced audit. All prompt conditions have identical edges. This negative result shows that the center-distance heuristic poorly matches the audit's adjacency concept. The audit is not independent human benchmark truth, and its sample is too small to claim semantic-relation validity. Use independently annotated relations before evaluating a learned relation component.

## Runtime and reproducibility

Executed environment: Python 3.12.14, torch 2.5.1+cpu, torchvision 0.20.1+cpu, Transformers 4.46.3, NumPy 2.1.3, Pillow 11.0.0; AMD EPYC 9V74 guest, three visible vCPUs, two torch threads. Complete resolved dependencies are in requirements.lock.txt.

With cached weights, model initialization took 2.23 s, all 114 crop features 10.39 s, baseline label embeddings 0.37 s. Contextual text encoding across 24 scenes took 17.61 s (general), 24.44 s (detailed), 23.06 s (misleading). These costs are shared across candidates within scenes and exclude downloads. Literal context requires no additional encoder calls. A separate single warm classification-and-graph replay took roughly 8–11 ms per condition across all scenes, excluding I/O and model computation. Timings are measurements from one CPU run, not latency distributions.

Raw graphs preserve candidate IDs, image/box references, all 80 base scores, hypotheses, abstentions and evidence margins; contextual graphs also preserve conditioned and decision scores. Feature arrays, exact prompts, paired bootstrap statistics, provenance, checksums, model revision and runtime are retained. Verification confirms eight conditions each have identical 114 region references and identical base visual evidence. Thirteen tests pass; the real-image backend ran successfully offline after asset download. Synthetic fixture results remain separate.

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

- Li et al., **DesCo: Learning Object Recognition with Rich Language Descriptions** (NeurIPS 2023), directly studies description-conditioned recognition and reports that adding rich language alone does not ensure contextual use. This is closer recognition prior art than a generic CLIP citation. https://papers.neurips.cc/paper_files/paper/2023/hash/761c3284ee4859bff3c7e5d9299a45ee-Abstract-Conference.html
- An et al., **PerceptionCLIP: Visual Classification by Inferring and Conditioning on Contexts** (ICLR 2024), explicitly conditions CLIP classification on contextual attributes. A claim that adding scene context to CLIP is new would be untenable. The exploratory template here is not a reproduction of their two-step method. https://arxiv.org/abs/2308.01313
- Zhang et al., **Putting Visual Object Recognition in Context** (CVPR 2020), studies systematic manipulations of scene context and recognition, supporting controlled evaluation rather than assuming contextual benefits. https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Putting_Visual_Object_Recognition_in_Context_CVPR_2020_paper.html
- Zhang et al., **Do Images Speak Louder than Words? Investigating the Effect of Textual Misinformation in VLMs** (EACL 2026), tests conflicting text on VQA models. Prompt reliability itself is also an existing research topic; this prototype's fixed-region classification setting differs, but broad novelty is unsupported. https://aclanthology.org/2026.eacl-long.323/

These verified publications show a real empirical question, not a verified novel idea. A viable thesis may examine calibrated risk/coverage tradeoffs under externally supplied scene descriptions in held-out industrial environments, with a reproducible controlled protocol. Establishing that contribution requires a more exhaustive literature review and a comparison against description-conditioned/contextual CLIP methods. The present pilot is a feasibility and failure-analysis study, not a claim of a new state-of-the-art mapping system.

## Limitations and next steps

The general-context result is weak, and the largest apparent gain uses explicit visible-object names, which are additional supervision rather than free context. Descriptions are assistant-authored; replication with independent human authors is needed. There are only 24 nonrandom public scenes, with oracle boxes, possible CLIP training overlap, 36 represented classes, uncalibrated thresholds and no workshop-specific parts. Crop contamination, missing unknown-class rejection, plural/synonym/negation failures and the simplistic context-fusion rule constrain conclusions. Confidence is not calibrated; no operational safety bound follows from cosine gating. No cross-scene physical identity or temporal fusion is implemented. No independent public semantic-relation evaluation or part_of recognition was completed; the separate subjective audit makes this gap explicit rather than inventing relation truth.

Next steps, **proposed rather than executed**:

1. Reserve independent calibration scenes to select abstention thresholds and context strength. Repeat with multiple description authors and matched coverage; predeclare acceptable wrong-assignment risk.
2. Test at least 100 held-out scenes (final size based on pilot variance and desired interval precision), including real workshop images with independent object and part annotations. Use scene-level paired uncertainty and preserve all failures.
3. Compare stronger established context-conditioned methods, then test misleading plausible context, negation, absent categories, paraphrases, unknown objects and gradual prompt corruption. Diagnose whether gains come from class mentions or scene-type information.
4. Replace the center-distance rule and evaluate relations against an independently annotated dataset or blinded human annotation. Add 3D geometry for physical near and explicit visual part evidence for part_of.
5. Only after replicated benefit at acceptable risk, add automatic proposals and multiview mapping. Unity remains outside this prototype.

No deploy/integrate recommendation is supported yet. The measured detailed-description ranking improvement merits further Python experiments, while the general-context and relation results remain negative or inconclusive.

## Dependencies and licenses

Python standard-library core; optional Pillow (HPND), NumPy (BSD-3-Clause), torch and torchvision (BSD-style), Transformers (Apache-2.0), Hugging Face Hub (Apache-2.0). The OpenAI CLIP repository distributes code and model weights under MIT; retain its license and model-card bias/intended-use notes: https://github.com/openai/CLIP and https://huggingface.co/openai/clip-vit-base-patch32 . All model and processor asset hashes and the immutable commit are retained. COCO annotation and image rights are separate; see DATASET.md and the retained per-image provenance. Model files and dataset images are local caches, excluded from the compact source/results archive.
