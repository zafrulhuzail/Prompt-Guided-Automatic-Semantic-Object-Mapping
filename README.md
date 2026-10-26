# Prompt-guided semantic grounding research prototype

**Completed exploratory pilot:** 24 real COCO validation images, 114 supplied object regions, 80 candidate classes, two local CLIP-based context methods and all four prompt conditions. See [REPORT.md](REPORT.md) and [results/coco24/SUMMARY.md](results/coco24/SUMMARY.md).

Detailed descriptions improved the literal-prior method's accepted-correct rate from 23.68% to 33.33%, but wrong-target risk increased from 22.86% to 29.63%. General scene descriptions showed no clear benefit. This is evidence to refine the research, not evidence to integrate into Unity.

This is **supplied-region semantic grounding**, not end-to-end automatic object discovery or mapping. Boxes come from human instance annotations. Nodes identify observed regions, not persistent physical identities. There is no multiview association, 3D reconstruction or Unity integration.

## Run the lightweight demonstration

From this directory, Python standard library only:

```bash
python -m unittest discover -s tests -v
python -m pgm.run --fixture --manifest demo.json --truth demo_truth.json --out results/fixture_replay
python -m pgm.relation_audit
```

The fixture uses invented scores and no real image. Its results are engineering checks only. The COCO24 results are real visual experiments and are stored separately.

## Reproduce real-image experiments

The current workspace already has the data and cached model. A one-image real demonstration has also been executed and saved in `results/real_demo/`. To replay it from this directory:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pgm.run --manifest inputs/demo_real.json --truth inputs/demo_real_truth.json --out results/real_demo_replay
```

For the full pilot, use a new output directory to preserve prior results:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pgm.pilot --stage all --out results/reproduction
python -m pgm.verify
```

On a fresh machine, install dependencies and retrieve assets first:

```bash
python -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python -m pgm.download
HF_HUB_OFFLINE=1 .venv/bin/python -m pgm.pilot --stage all --out results/reproduction
```

The frozen `inputs/` files contain the actual pilot proposals, prompts, selected image metadata, evaluation labels, and authorship record. The pilot loader resolves the image filename beneath this folder's data/val2017, so a relocated folder works. The downloader fetches the official annotation archive only if needed (~241 MB), then the selected 24 images (~3 MB) and pinned CLIP weights (~600 MB). No online inference service is used. CPU timings and environment are recorded; see requirements.lock.txt for the executed environment. Model dependencies are optional for reading results and running unit tests.

For new COCO subsets use `python -m pgm.prepare --help`. The generic single-scene inference CLI is `python -m pgm.run --help`; it requires an authored prompt manifest and pinned model revision. The pilot CLI handles separate prompts for each scene, reads no truth during inference, and evaluates only after predictions have been written. Evaluation can be replayed without loading the model:

```bash
.venv/bin/python -m pgm.pilot --stage evaluate
.venv/bin/python -m pgm.analyze
python -m pgm.relation_audit
python -m pgm.benchmark
```

## Files

- `pgm/core.py`: candidates, literal prompt interpretation, bounded fusion, abstention and graph construction.
- `pgm/model.py`: local CLIP image/text encoders.
- `pgm/pilot.py`: frozen per-scene inference and separate scene-cluster evaluation.
- `pgm/evaluate.py`, `analyze.py`, `relation_audit.py`: classification risk, matched coverage and reviewed-pair relation audit.
- `inputs/`: proposals and descriptions; `truth.json` is used only by evaluation.
- `results/coco24/`: eight raw graph collections, metrics, score-bearing predictions, feature cache, runtime, checksums and provenance.
- `DATASET.md`, `PROTOCOL.md`, `REPORT.md`: provenance, protocol, measured results, limitations and literature.

Graph node predictions always reference an existing observed region. Scores and margins are uncalibrated evidence, not probabilities. `near` is a 2D heuristic; `part_of` is unsupported rather than invented. Initial incomplete status and report are preserved in `results/archive_initial/`; synthetic fixtures remain untouched.
