# ===========================================================================
# AIG230 Final Project — Group 7
#
#   make setup     install dependencies
#   make data      download and preprocess the corpus
#   make train     train the primary model
#   make all-runs  train all four systems
#   make evaluate  score the primary model on the test set
#   make figures   regenerate every figure
#   make report    rebuild the PDF report
#   make app       launch the inference application
#   make test      run the unit tests
#   make check     lint + tests
# ===========================================================================

PYTHON      ?= python
export PYTHONPATH := src

CONFIG      ?= configs/bpe_scratch.yaml
RUN         ?= bpe_scratch
CHECKPOINT  ?= artifacts/checkpoints/$(RUN)/best_bleu.pt
CONFIGS     := bpe_scratch word_random word_muse lstm_baseline

.DEFAULT_GOAL := help
.PHONY: help setup data embeddings train all-runs evaluate evaluate-all \
        figures report app test lint check smoke clean clean-nb clean-report

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- environment -----------------------------------------------------------

setup: ## Install Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# --- data ------------------------------------------------------------------

data: ## Download Tatoeba, build the corpus, fit the tokenisers
	$(PYTHON) -m nmt.data.build

embeddings: ## Build the MUSE initialisation matrix (needs the .vec files)
	@mkdir -p data/raw/embeddings
	@test -f data/raw/embeddings/wiki.multi.en.vec || \
		curl -L --progress-bar -o data/raw/embeddings/wiki.multi.en.vec \
		https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
	@test -f data/raw/embeddings/wiki.multi.es.vec || \
		curl -L --progress-bar -o data/raw/embeddings/wiki.multi.es.vec \
		https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec
	$(PYTHON) -m nmt.data.embeddings

# --- training --------------------------------------------------------------

smoke: ## One-minute end-to-end pipeline check
	$(PYTHON) -m nmt.training.train --config configs/smoke.yaml

train: ## Train the primary model (override with CONFIG=...)
	$(PYTHON) -m nmt.training.train --config $(CONFIG)

all-runs: ## Train all four systems in sequence
	@for cfg in $(CONFIGS); do \
		echo "=== $$cfg ==="; \
		$(PYTHON) -m nmt.training.train --config configs/$$cfg.yaml || exit 1; \
	done

# --- evaluation ------------------------------------------------------------

evaluate: ## Evaluate one checkpoint (override with RUN=...)
	$(PYTHON) -m nmt.evaluation.evaluate --checkpoint $(CHECKPOINT)

evaluate-all: ## Evaluate every trained checkpoint
	@for cfg in $(CONFIGS); do \
		if [ -f artifacts/checkpoints/$$cfg/best_bleu.pt ]; then \
			echo "=== $$cfg ==="; \
			$(PYTHON) -m nmt.evaluation.evaluate \
				--checkpoint artifacts/checkpoints/$$cfg/best_bleu.pt || exit 1; \
		fi; \
	done

# --- outputs ---------------------------------------------------------------

figures: ## Regenerate every figure
	$(PYTHON) -m nmt.viz.make_figures

report: figures ## Rebuild the PDF report
	$(PYTHON) reports/build_report_data.py
	cd reports/final_report && latexmk -pdf -interaction=nonstopmode main.tex
	@echo "-> reports/final_report/main.pdf"

app: ## Launch the Gradio inference application
	$(PYTHON) app/app.py

# --- quality ---------------------------------------------------------------

test: ## Run the unit tests
	$(PYTHON) -m pytest

lint: ## Check PEP 8 compliance
	$(PYTHON) -m ruff check src app tests reports

check: lint test ## Lint and test

# --- housekeeping ----------------------------------------------------------

clean-nb: ## Strip notebook outputs before committing
	$(PYTHON) -m nbstripout notebooks/*.ipynb

clean-report: ## Remove LaTeX build artefacts
	cd reports/final_report && latexmk -C

clean: clean-report ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
