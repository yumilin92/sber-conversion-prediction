DAY ?= 2021-12-15

install:
	pip install -r requirements-dev.txt

test:
	python -m pytest -q

lint:
	ruff check src tests

store:
	python -m src.build_store --day $(DAY)

score:
	python -m src.batch_score --day $(DAY) --out scores/$(DAY).parquet

monitor:
	python -m src.monitor report --day $(DAY)

api:
	uvicorn src.api:app --reload

docker:
	docker build -t sber-conversion . && docker run --rm -p 8000:8000 sber-conversion

.PHONY: install test lint store score monitor api docker
