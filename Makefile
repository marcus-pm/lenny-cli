.PHONY: setup install clean submodule help

VENV := .venv
PIP := $(VENV)/bin/pip

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/bin/activate ## Create venv, install package, prepare .env
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. Activate the venv:  source $(VENV)/bin/activate"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "  2. Add your API key:   edit .env"; \
	else \
		echo "  2. .env already exists"; \
	fi
	@echo "  3. Run the app:        lenny"

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install .

install: ## Reinstall package into existing venv
	$(PIP) install .

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

submodule: ## Initialize the transcripts submodule
	git submodule update --init
