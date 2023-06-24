# from https://github.com/alexander-beedie/polars/blob/a083a26ca092b3821bf4044aed74d91ee127bad1/py-polars/Makefile
.DEFAULT_GOAL := help

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHONPATH=
VENV = .venv
MAKE = make

VENV_BIN=$(VENV)/bin

.venv:  ## Set up virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh all project requirements
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements-dev.txt
	$(VENV_BIN)/pip install -r requirements.txt

.PHONY: build
build: .venv  ## Compile and install ranx
	source $(VENV_BIN)/activate

.PHONY: test
test: .venv build  ## Run unittest
	source $(VENV_BIN)/activate
	$(VENV_BIN)/python -m pytest --ignore=tests/unit/ranx/trec

.PHONY: coverage
coverage: .venv build  ## Run tests and report coverage
	source $(VENV_BIN)/activate
	$(VENV_BIN)/pytest --cov -n auto --dist worksteal -m "not benchmark"

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .venv/
	@rm -rf target/
	@rm -rf .pytest_cache/

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort