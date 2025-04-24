# Makefile for Awesome-LLMOps Landscape
# See: https://github.com/cncf/landscape2 for reference

# Configuration
DATA_FILE := website/data.yml
SETTINGS_FILE := website/settings.yml
GUIDE_FILE := website/guide.yml
LOGOS_PATH := website/logos
OUTPUT_DIR := build
CACHE_DIR := .cache
LANDSCAPE2_VERSION := latest
CONTAINER_NAME := awesome-llmops-landscape

# Detect OS for installation
UNAME_S := $(shell uname -s)
LANDSCAPE2_BIN := $(shell command -v landscape2 2> /dev/null)

# The image only supports this amd64 platform
DOCKER_PLATFORM := --platform linux/amd64

# Default target
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  install    Install landscape2"
	@echo "  validate   Validate data and settings files"
	@echo "  build      Build the landscape"
	@echo "  serve      Serve the landscape website locally"
	@echo "  docker-build  Build the landscape using Docker"
	@echo "  docker-serve  Serve the landscape using Docker"
	@echo "  docker-stop   Stop the Docker container serving the landscape"
	@echo "  run        Run complete workflow (install, validate, build, serve)"
	@echo "  clean      Clean build artifacts and Docker containers"

all: install validate build

# Install landscape2 based on detected OS
.PHONY: install
install:
	@echo "Installing landscape2..."
ifeq ($(LANDSCAPE2_BIN),)
ifeq ($(UNAME_S),Darwin)
	@echo "Installing via Homebrew on macOS..."
	brew install cncf/landscape2/landscape2
else ifeq ($(UNAME_S),Linux)
	@echo "Installing via curl script on Linux..."
	curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cncf/landscape2/releases/download/$(LANDSCAPE2_VERSION)/landscape2-installer.sh | sh
else ifeq ($(shell echo "$(UNAME_S)" | grep -c "MINGW\|MSYS\|CYGWIN"),1)
	@echo "Installing via PowerShell on Windows..."
	powershell -Command "irm https://github.com/cncf/landscape2/releases/download/$(LANDSCAPE2_VERSION)/landscape2-installer.ps1 | iex"
else
	@echo "Unsupported OS for direct installation. Please install manually:"
	@echo "See: https://github.com/cncf/landscape2#installation"
	@exit 1
endif
else
	@echo "landscape2 is already installed."
endif

# Validate data and settings files
.PHONY: validate
validate:
	@echo "Validating data and settings files..."
	landscape2 validate data --data-file $(DATA_FILE)
	landscape2 validate settings --settings-file $(SETTINGS_FILE)
	landscape2 validate guide --guide-file $(GUIDE_FILE)
	@echo "Validation completed successfully ✓"

# Build the landscape
.PHONY: build
build:
	@echo "Building landscape website..."
	@mkdir -p $(OUTPUT_DIR) $(CACHE_DIR)
	landscape2 build \
		--data-file $(DATA_FILE) \
		--settings-file $(SETTINGS_FILE) \
		--guide-file $(GUIDE_FILE) \
		--logos-path $(LOGOS_PATH) \
		--output-dir $(OUTPUT_DIR) \
		--cache-dir $(CACHE_DIR)
	@echo "Build completed ✓"

# Serve the landscape locally
.PHONY: serve
serve:
	@echo "Serving landscape website on http://127.0.0.1:8000 ..."
	landscape2 serve --landscape-dir $(OUTPUT_DIR)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts and Docker containers..."
	rm -rf $(OUTPUT_DIR)
	@echo "Stopping any running landscape Docker containers..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true

# Full workflow: install, validate, build, and serve
.PHONY: run
run: install validate build serve

# Stop Docker container if running
.PHONY: docker-stop
docker-stop:
	@echo "Stopping any running landscape Docker containers..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true

# Docker-based alternatives (useful for CI/CD)
.PHONY: docker-build
docker-build:
	@echo "Building landscape using Docker ..."
	@mkdir -p $(OUTPUT_DIR) $(CACHE_DIR)
	docker run --rm $(DOCKER_PLATFORM) -v $(PWD):/landscape public.ecr.aws/g6m3a0y9/landscape2:latest \
		landscape2 build \
		--data-file /landscape/$(DATA_FILE) \
		--settings-file /landscape/$(SETTINGS_FILE) \
		--guide-file /landscape/$(GUIDE_FILE) \
		--logos-path /landscape/$(LOGOS_PATH) \
		--output-dir /landscape/$(OUTPUT_DIR) \
		--cache-dir /landscape/$(CACHE_DIR)
	@echo "Docker build completed ✓"

.PHONY: docker-serve
docker-serve: docker-stop
	@echo "Serving landscape using Docker on http://localhost:8000 ..."
	docker run --rm $(DOCKER_PLATFORM) -p 8000:8000 --name $(CONTAINER_NAME) -v $(PWD):/landscape public.ecr.aws/g6m3a0y9/landscape2:latest \
		landscape2 serve \
		--addr 0.0.0.0:8000 \
		--landscape-dir /landscape/$(OUTPUT_DIR)
	@echo "Docker container stopped" 