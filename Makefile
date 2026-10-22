PYTHON ?= python

.PHONY: install data train evaluate test lint typecheck run package-demo docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements-runtime.txt -e ".[dev]"

data:
	$(PYTHON) -m pip install -r requirements-runtime.txt -e ".[data]"
	$(PYTHON) scripts/fetch_data.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy

run:
	$(PYTHON) -m intentops.serve

package-demo:
	$(PYTHON) scripts/package_demo.py

docker-build:
	docker build -t intentops:local .

docker-run:
	docker run --rm -p 8000:8000 intentops:local
