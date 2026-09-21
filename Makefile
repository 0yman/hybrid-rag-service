.PHONY: help install corpus index serve test lint eval eval-keyword eval-all docker clean

help:
	@echo "install      Install runtime + dev dependencies"
	@echo "corpus       Download the demo corpus from Wikipedia"
	@echo "index        Build the FAISS + BM25 index from data/corpus"
	@echo "serve        Run the API on http://localhost:8000"
	@echo "test         Run the test suite (offline, no API key)"
	@echo "lint         Run ruff"
	@echo "eval-all     Run both evaluation suites and write eval/results*.md"
	@echo "docker       Build and run the container"

install:
	pip install -r requirements-dev.txt

corpus:
	python scripts/fetch_corpus.py

index:
	python scripts/ingest.py

serve:
	uvicorn rag.api:app --app-dir src --reload --port 8000

test:
	python -m pytest

lint:
	ruff check src eval scripts tests

eval:
	python eval/run_eval.py --ablate --k-sweep 1,3,5,10 --generate

eval-keyword:
	python eval/run_eval.py --ablate --k-sweep 1,3,5 -k 3 \
		--golden eval/golden_keyword.jsonl \
		--out eval/results_keyword.md --json-out eval/results_keyword.json

eval-all: eval eval-keyword

docker:
	docker compose up --build

clean:
	rm -rf data/index .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
