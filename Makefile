CONFIG ?= configs/sam_vit_b.yaml

install:
	pip install -e ".[api,tracking,dev]"

prepare:
	waterseg-prepare --config $(CONFIG)

train:
	waterseg-train --config $(CONFIG)

evaluate:
	waterseg-evaluate --config $(CONFIG)

test:
	pytest -q

lint:
	ruff check src tests

api:
	uvicorn waterseg.api:app --host 0.0.0.0 --port 8000

build:
	docker build -t aereo-water-sam:latest .

sweep:
	waterseg-sweep --config $(CONFIG)
