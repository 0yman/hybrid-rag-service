.PHONY: help install run test lint eval eval-keyword eval-reference eval-all corpus docker clean

help:
	@echo "install         Install the app"
	@echo "run             Start the app and open it in your browser"
	@echo "test            Run the test suite (offline, no API key)"
	@echo "lint            Run ruff"
	@echo "eval-all        Evaluate on the benchmark with the default runtime (fastembed)"
	@echo "eval-reference  Same, with sentence-transformers (needs requirements-extras.txt)"
	@echo "corpus          Re-download the benchmark corpus from Wikipedia"
	@echo "docker          Build and run the container"

install:
	pip install -r requirements.txt

run:
	python app.py

test:
	python -m pytest

lint:
	ruff check src eval scripts tests app.py

eval:
	python eval/run_eval.py --ablate --k-sweep 1,3,5,10 --generate

eval-keyword:
	python eval/run_eval.py --ablate --k-sweep 1,3,5 -k 3 \
		--golden eval/golden_keyword.jsonl \
		--out eval/results_keyword.md --json-out eval/results_keyword.json

eval-all: eval eval-keyword

eval-reference:
	python eval/run_eval.py --engine sentence-transformers --ablate --k-sweep 1,3,5,10 --generate \
		--out eval/results_sentence_transformers.md --json-out eval/results_sentence_transformers.json
	python eval/run_eval.py --engine sentence-transformers --ablate --k-sweep 1,3,5 -k 3 \
		--golden eval/golden_keyword.jsonl \
		--out eval/results_keyword_sentence_transformers.md \
		--json-out eval/results_keyword_sentence_transformers.json

corpus:
	python scripts/fetch_corpus.py

docker:
	docker compose up --build

clean:
	rm -rf data/index data/uploads data/benchmark_index .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
