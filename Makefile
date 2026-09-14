PYTHON ?= python
CONFIG ?= configs/confirmatory.yaml

.PHONY: test pilot main artifacts verify all traffic-check

test:
	$(PYTHON) -m unittest discover -s tests -v

pilot:
	$(PYTHON) scripts/run_experiments.py --config $(CONFIG) --stage pilot

main:
	$(PYTHON) scripts/run_experiments.py --config $(CONFIG) --stage main

artifacts:
	$(PYTHON) scripts/make_confirmatory_artifacts.py --config $(CONFIG)

verify:
	$(PYTHON) scripts/verify_confirmatory.py --config $(CONFIG)

all: test pilot main artifacts verify

traffic-check:
	$(PYTHON) scripts/check_traffic_data.py --config configs/publication_traffic.yaml
