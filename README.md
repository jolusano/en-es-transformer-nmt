# Transformer-Based Neural Machine Translation — English ↔ Spanish

**AIG230 · Natural Language Processing · Seneca Polytechnic**
Final Project, Option 1 — **Group 7**: Jose Luis Sanchez Noriega, Bikash Subedi
Instructor: Ellie Azizi

A complete neural machine translation pipeline built around a transformer
encoder–decoder written from primitives in PyTorch. **One set of weights
translates in both directions** — English→Spanish and Spanish→English — using a
joint subword vocabulary, three-way tied embeddings, and a reserved direction
tag prepended to the source sentence.

```
<2es> I am tired .     →   Estoy cansado .
<2en> Estoy cansado .  →   I am tired .
```

---

## Table of contents

1. [What this project contains](#1-what-this-project-contains)
2. [Quick start](#2-quick-start)
3. [Installation](#3-installation)
4. [Building the dataset](#4-building-the-dataset)
5. [Training](#5-training)
6. [Evaluation and error analysis](#6-evaluation-and-error-analysis)
7. [Demonstration video](#7-demonstration-video)
8. [Running the inference application](#8-running-the-inference-application)
9. [Reproducing the report](#9-reproducing-the-report)
10. [Repository layout](#10-repository-layout)
11. [Design decisions](#11-design-decisions)
12. [Testing](#12-testing)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What this project contains

| Deliverable | Where |
|---|---|
| Source code | [`src/nmt/`](src/nmt), [`app/`](app) |
| Trained model weights | [`best_bleu_release.pt` on Google Drive](https://drive.google.com/file/d/10PVZjE4ctldX3hyd0BYLbJir61fqdDik/view?usp=sharing) (144 MB) |
| Project report | [`main.pdf`](reports/final_report/main.pdf) — source: [`main.tex`](reports/final_report/main.tex) |
| Demonstration video | [Google Drive](https://drive.google.com/file/d/19z4fEsQfhgtJMMy6xaw2-D33NoixSP75/view?usp=sharing) |
| Notebooks | [`notebooks/`](notebooks) |

**Four systems are trained and compared:**

| Run | Tokenisation | Embeddings | Purpose |
|---|---|---|---|
| `bpe_scratch` | joint BPE 16k | learned from scratch | **primary model** |
| `word_random` | word-level 32k | random, 300-d | control for the experiment below |
| `word_muse` | word-level 32k | MUSE cross-lingual, 300-d | does a pre-trained init help? |
| `lstm_baseline` | joint BPE 16k | learned from scratch | architecture comparison |

`word_random` and `word_muse` are byte-for-byte identical except for the
contents of the embedding matrix at step 0 — that is what makes the comparison
interpretable.

---

## 2. Quick start

Already have a checkpoint? Two commands:

```bash
pip install -r requirements.txt
python app/app.py
```

From nothing, end to end:

```bash
python -m nmt.data.build                                     # ~4 min, downloads 172 MB
python -m nmt.training.train --config configs/bpe_scratch.yaml
python -m nmt.evaluation.evaluate --checkpoint artifacts/checkpoints/bpe_scratch/best_bleu.pt
python app/app.py
```

Or use the Makefile:

```bash
make setup data train evaluate figures report app
```

---

## 3. Installation

### Requirements

* Python 3.10 or newer
* ~2 GB free disk for the corpus, ~1.3 GB more if you use the MUSE embeddings
* A GPU is optional for inference and strongly recommended for training

### Virtual environment (recommended by the project brief)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Making the package importable

Every command below assumes `src/` is on the import path. Either install the
project in editable mode:

```bash
pip install -e .
```

…or prefix commands with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m nmt.data.build
```

### Verifying the install

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
pytest -q
```

The code selects its device automatically — CUDA, then Apple MPS, then CPU — so
the same commands work on a MacBook and in Colab without edits.

---

## 4. Building the dataset

```bash
python -m nmt.data.build
```

This downloads the official Tatoeba exports (~172 MB), reconstructs the
English–Spanish bitext from the translation-link graph, cleans and filters it,
splits it **by connected component** to avoid leakage, fits both tokenisers on
the training split only, and writes corpus statistics.

| Output | Contents |
|---|---|
| `data/interim/pairs_raw.tsv` | the joined bitext, with Tatoeba sentence ids |
| `data/processed/{train,validation,test}.tsv` | the splits |
| `artifacts/tokenizers/joint_bpe.model` | joint SentencePiece BPE, 16k |
| `artifacts/tokenizers/word_vocab.json` | word-level vocabulary, 32k |
| `artifacts/results/dataset_stats.json` | everything the report's data section quotes |

Useful flags:

```bash
python -m nmt.data.build --skip-download        # raw files already present
python -m nmt.data.build --limit 20000          # fast smoke test
python -m nmt.data.build --subword-vocab 8000   # different vocabulary size
```

### Pre-trained embeddings (only for `word_muse`)

```bash
mkdir -p data/raw/embeddings
curl -L -o data/raw/embeddings/wiki.multi.en.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
curl -L -o data/raw/embeddings/wiki.multi.es.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec
python -m nmt.data.embeddings
```

Writes `artifacts/tokenizers/muse_embeddings.npy` and a coverage report.

---

## 5. Training

### Locally

```bash
python -m nmt.training.train --config configs/bpe_scratch.yaml
```

Check the pipeline first — this takes about a minute on a laptop CPU:

```bash
python -m nmt.training.train --config configs/smoke.yaml
```

Overrides without editing the YAML:

```bash
python -m nmt.training.train --config configs/bpe_scratch.yaml \
    --epochs 10 --device cpu --limit-train-pairs 50000
```

### On Google Colab (recommended for the full runs)

A 16 GB MacBook can train these models, but slowly. Open
[`notebooks/04_training_colab.ipynb`](notebooks/04_training_colab.ipynb) in
Colab, select a GPU runtime, and run the cells. It clones the repository,
installs dependencies, mounts Drive for checkpoints, and runs the same
`nmt.training.train` entry point used locally.

Training is **resumable** — checkpoints carry optimiser and scheduler state, so
a disconnected session can be picked up where it stopped:

```bash
python -m nmt.training.train --config configs/bpe_scratch.yaml \
    --resume artifacts/checkpoints/bpe_scratch/last.pt
```

### Outputs

| File | Contents |
|---|---|
| `artifacts/checkpoints/<run>/best_bleu.pt` | best validation BLEU — **use this for inference** |
| `artifacts/checkpoints/<run>/best_loss.pt` | best validation loss |
| `artifacts/checkpoints/<run>/last.pt` | most recent epoch, for resuming |
| `artifacts/results/<run>/training_summary.json` | full run manifest and per-epoch history |
| `artifacts/results/<run>/train_log.jsonl` | per-step log, written as training proceeds |

Checkpoints are **not committed** (see `.gitignore`) — a full one is ~430 MB,
because Adam keeps two moment tensors per parameter so the optimiser state is
twice the size of the weights again.

For distribution, strip everything inference does not read:

```bash
python scripts/export_checkpoint.py --all
```

That writes `best_bleu_release.pt` (~144 MB, 67% smaller) beside each
checkpoint, containing only the weights, the model config, the tokeniser path
and the evaluation metrics. `Translator.from_checkpoint` and the Gradio app load
it exactly as they load the full file; the weights are bit-identical. Keep the
full checkpoints only if you may want to resume training.

### Downloading the trained model

The primary model is published here:

**[`best_bleu_release.pt` ▸](https://drive.google.com/file/d/10PVZjE4ctldX3hyd0BYLbJir61fqdDik/view?usp=sharing)** — 144 MB

Place it at `artifacts/checkpoints/bpe_scratch/best_bleu.pt` and the app and the
evaluation script will find it. The tokenisers it needs are already in this
repository under `artifacts/tokenizers/`, so nothing else is required:

```bash
mkdir -p artifacts/checkpoints/bpe_scratch
# ...move the downloaded file there, then:
python app/app.py
```

Test-set scores for each system:

| System | Test BLEU (EN→ES / ES→EN) |
|---|---|
| `bpe_scratch` — **the one the app uses** | 49.99 / 56.83 |
| `word_muse` | 45.88 / 53.05 |
| `word_random` | 45.42 / 52.62 |
| `lstm_baseline` | 46.15 / 49.73 |

### Training all four systems

```bash
for cfg in bpe_scratch word_random word_muse lstm_baseline; do
  python -m nmt.training.train --config configs/$cfg.yaml
done
```

---

## 6. Evaluation and error analysis

```bash
python -m nmt.evaluation.evaluate \
    --checkpoint artifacts/checkpoints/bpe_scratch/best_bleu.pt
```

Produces, per direction: sacreBLEU and chrF2 (plus a from-scratch BLEU
implementation as a cross-check), a greedy-vs-beam comparison, BLEU stratified
by source length, a failure-mode census, the twenty worst translations, and
every hypothesis so scores can be recomputed without a GPU.

| Output | Contents |
|---|---|
| `artifacts/results/<run>/evaluation_test.json` | all metrics and all hypotheses |
| `artifacts/results/<run>/error_analysis_test.md` | worst examples, human-readable |

Options:

```bash
--split validation      # evaluate on validation instead of test
--strategy greedy       # skip beam search
--beam-size 8           # wider beam
--length-penalty 0.8    # favour longer output
--limit 500             # quick check
```

---

## 7. Demonstration video

A short walkthrough of the running application — both translation directions,
the round trip, the subword tokenisation, the cross-attention alignment, and
the difference between greedy and beam decoding:

**[Watch the demo ▸](https://drive.google.com/file/d/19z4fEsQfhgtJMMy6xaw2-D33NoixSP75/view?usp=sharing)**

---

## 8. Running the inference application

```bash
python app/app.py
```

Opens at <http://127.0.0.1:7860>. It discovers a checkpoint automatically;
point it at a specific one with `--checkpoint`, expose it publicly with
`--share`, or change the port with `--port`.

The interface provides a direction switch, decoding controls (greedy vs beam,
beam width, length penalty), a cross-attention alignment view, and a
tokenisation view showing how the sentence was split.

The model is loaded **once at start-up** and is never retrained by the app.

---

## 9. Reproducing the report

```bash
python -m nmt.viz.make_figures        # all figures → reports/figures/
python reports/build_report_data.py   # all tables  → reports/final_report/generated/
cd reports/final_report && latexmk -pdf main.tex
```

Or `make report`.

Every number, table and figure in the report is generated from the JSON
artefacts under `artifacts/results/`. Nothing is transcribed by hand, so the
document cannot drift out of step with the code and checkpoints. The report
compiles even before any model is trained — unavailable numbers render as `--`
and missing figures render as placeholders.

---

## 10. Repository layout

```
en-es-transformer-nmt/
├── README.md
├── requirements.txt
├── Makefile
├── configs/                      one YAML per experiment
│   ├── bpe_scratch.yaml          primary model
│   ├── word_random.yaml          control for the embedding experiment
│   ├── word_muse.yaml            pre-trained cross-lingual embeddings
│   ├── lstm_baseline.yaml        recurrent baseline
│   └── smoke.yaml                one-minute pipeline check
├── src/nmt/
│   ├── constants.py              special tokens and direction tags
│   ├── config.py                 typed configuration objects
│   ├── data/
│   │   ├── download.py           fetch the Tatoeba exports
│   │   ├── pairing.py            reconstruct the bitext from the link graph
│   │   ├── cleaning.py           normalisation and filtering
│   │   ├── splitting.py          leakage-free component-aware splitting
│   │   ├── tokenizer.py          joint BPE + word-level, one interface
│   │   ├── embeddings.py         MUSE vectors → initialisation matrix
│   │   ├── dataset.py            datasets, token bucketing, masking
│   │   └── build.py              runs the whole preprocessing pipeline
│   ├── model/
│   │   ├── attention.py          scaled dot-product + multi-head, from scratch
│   │   ├── positional.py         sinusoidal and learned encodings
│   │   ├── masking.py            padding, causal, cross-attention masks
│   │   ├── layers.py             encoder/decoder blocks, pre-LN and post-LN
│   │   ├── transformer.py        the full bidirectional model
│   │   └── baseline_lstm.py      Bi-LSTM + Bahdanau attention baseline
│   ├── training/
│   │   ├── loss.py               label-smoothed cross entropy
│   │   ├── scheduler.py          inverse-sqrt warmup, cosine, constant
│   │   ├── trainer.py            the hand-written training loop
│   │   └── train.py              CLI entry point
│   ├── inference/
│   │   ├── search.py             greedy and beam search decoding
│   │   └── translator.py         checkpoint → callable translator
│   ├── evaluation/
│   │   ├── bleu.py               BLEU from the definition + sacreBLEU
│   │   ├── error_analysis.py     automated failure-mode classification
│   │   └── evaluate.py           CLI entry point
│   ├── viz/                      all figures (distill.pub-inspired styling)
│   └── utils/                    seeding, logging, devices, I/O
├── app/app.py                    Gradio inference application
├── notebooks/                    exploration → training → evaluation
├── scripts/export_checkpoint.py  strip optimiser state for distribution
├── reports/
│   ├── build_report_data.py      JSON artefacts → LaTeX tables and macros
│   ├── figures/                  generated figures (PDF + PNG + SVG)
│   └── final_report/main.tex     the report
├── tests/                        unit tests
├── data/                         corpus (gitignored)
└── artifacts/                    checkpoints, tokenisers, results
```

---

## 11. Design decisions

Reasoning is set out in full in the report; the short version:

**One model, both directions.** A direction tag on the source lets one set of
weights serve both ways. Each cleaned pair yields two training examples, so
every parameter sees twice the supervision — which matters on a corpus two
orders of magnitude smaller than the WMT datasets transformers were designed
for.

**Joint 16k BPE vocabulary, three-way tied.** Sharing one vocabulary is what
makes the bidirectional model possible, and tying the encoder embedding, the
decoder embedding and the output projection removes ~17M parameters (about a
third of the model). Cognates like *nation*/*nación* share subword units.

**Component-aware splitting.** Tatoeba is a translation *graph*: one English
sentence links to several Spanish ones. Splitting pairs at random puts
paraphrases of the same sentence on both sides of the train/test boundary, and
the resulting BLEU measures memorisation. Splitting connected components
removes the leak — verified on every build.

**4+4 layers, not 6+6.** Depth is the first thing to cut when data rather than
compute is the binding constraint; it also halves step time and memory, which
is what makes an epoch fit in a free Colab session.

**Pre-layer-normalisation.** The residual path stays un-normalised from input
to output, so gradients reach early layers undamped and training is stable at
higher learning rates with a shorter warmup. A run that diverges at step 300 is
an hour of Colab wasted.

**Label smoothing at 0.1.** Several translations of a sentence are genuinely
correct, so a loss demanding probability 1.0 on one reference is asking for
something false.

**Token-count batching.** The length distribution is right-skewed, so a fixed
sentence count produces batches whose padding waste swings widely. Capping by
tokens keeps memory per step flat.

**No key/value cache in the decoder.** A deliberate trade: the cache would
roughly double the decoder's size and obscure the architecture the report
explains, and at a median of eight subword tokens the cost is negligible.

---

## 12. Testing

```bash
pytest -q                 # all tests
pytest -q -k attention    # one area
```

The suite covers the pieces where a silent bug would be invisible in the loss
curve: mask shapes and causality, attention output shapes and softmax
normalisation, tokeniser round-trips and reserved-id pinning, the BLEU
implementation against sacreBLEU, the label-smoothed loss against
`torch.nn.functional.cross_entropy` at zero smoothing, leakage-free splitting,
and a regression test for the tensor-contiguity bug described below.

---

## 13. Troubleshooting

**`ModuleNotFoundError: No module named 'nmt'`**
`src/` is not on the path. Run `pip install -e .` or prefix with `PYTHONPATH=src`.

**Training loss looks impossibly low, or token accuracy is ~95% at step 1**
This was a real bug and there is a regression test for it. `decoder_input` and
`labels` are slices of the same padded tensor; on Apple's MPS backend, moving a
non-contiguous view to the device ignored the storage offset, so the decoder
received the labels as its input and "predicted" the answer. Both slices are
now made contiguous in `collate_batch`. If you see this symptom after modifying
the collate function, that is the cause.

**Out of memory during training**
Lower `data.max_tokens_per_batch` (8192 → 4096) and raise
`optim.gradient_accumulation_steps` to keep the effective batch size the same.

**`FileNotFoundError` for `wiki.multi.*.vec`**
Only `word_muse` needs those; download them as shown in
[section 4](#4-building-the-dataset), or train the other three configs.

**SentencePiece fails with `sentences_.empty()`**
An earlier pipeline stage produced an empty corpus. Delete
`data/interim/pairs_raw.tsv` and re-run `python -m nmt.data.build`.

**The report compiles but numbers show as `--`**
`reports/build_report_data.py` has not been run, or the corresponding run has
no results yet. Placeholders are intentional so the document always compiles.

---

## Data and licensing

Sentence data comes from the [Tatoeba Project](https://tatoeba.org), released
under CC-BY 2.0 FR. Pre-trained vectors are MUSE
([Conneau et al., 2018](https://github.com/facebookresearch/MUSE)). Both are
downloaded at build time and are not redistributed in this repository.

## Acknowledgements

Vaswani et al. (2017) for the architecture; Johnson et al. (2017) for the
direction-tagging approach that makes one model serve both directions; Post
(2018) for sacreBLEU. Course materials and guidance from Ellie Azizi, AIG230,
Seneca Polytechnic.
