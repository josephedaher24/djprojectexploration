SHELL := /bin/bash

.DEFAULT_GOAL := help

DATA_DIR := data
SNIPPETS_DIR := $(DATA_DIR)/snippets
EXPORTS_DIR := $(DATA_DIR)/exports

CSV ?= $(EXPORTS_DIR)/dataset_tracks.csv
CACHE ?= dataset_tracks

.PHONY: help sync snippet-cache snippets-list snippets-size clean-snippets clean-snippet-cache clean-snippets-one clean-interactive-html clean-generated

help:
	@echo "Available targets:"
	@echo "  make sync                 - Run uv sync"
	@echo "  make snippet-cache CSV=...- Build snippet cache from a playlist CSV"
	@echo "  make snippets-list        - List snippet cache subdirectories"
	@echo "  make snippets-size        - Show snippet cache disk usage"
	@echo "  make clean-snippets       - Remove all snippet caches under $(SNIPPETS_DIR)/"
	@echo "  make clean-snippets-one CACHE=<name> - Remove one snippet cache directory"
	@echo "  make clean-interactive-html - Remove generated *interactive_pacmap.html exports"
	@echo "  make clean-generated      - Clean snippets + interactive HTML exports"

sync:
	uv sync

snippet-cache:
	uv run djprojectexploration-snippet-cache "$(CSV)"

snippets-list:
	@mkdir -p "$(SNIPPETS_DIR)"
	@echo "Snippet cache directories in $(SNIPPETS_DIR):"
	@find "$(SNIPPETS_DIR)" -mindepth 1 -maxdepth 1 -type d -print | sort || true

snippets-size:
	@mkdir -p "$(SNIPPETS_DIR)"
	@echo "Disk usage for $(SNIPPETS_DIR):"
	@du -sh "$(SNIPPETS_DIR)"/* 2>/dev/null || echo "(empty)"

clean-snippets:
	@mkdir -p "$(SNIPPETS_DIR)"
	@echo "Removing all snippet caches under $(SNIPPETS_DIR)/..."
	@find "$(SNIPPETS_DIR)" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	@echo "Done."

# Alias
clean-snippet-cache: clean-snippets

clean-snippets-one:
	@mkdir -p "$(SNIPPETS_DIR)"
	@echo "Removing snippet cache: $(SNIPPETS_DIR)/$(CACHE)"
	@rm -rf "$(SNIPPETS_DIR)/$(CACHE)"
	@echo "Done."

clean-interactive-html:
	@mkdir -p "$(EXPORTS_DIR)"
	@echo "Removing generated interactive HTML files in $(EXPORTS_DIR)/..."
	@find "$(EXPORTS_DIR)" -maxdepth 1 -type f -name '*interactive_pacmap.html' -print -delete
	@echo "Done."

clean-generated: clean-snippets clean-interactive-html
