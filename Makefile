SHELL := /bin/bash

DOCKER_BUILDKIT := 1
TEST_IMAGE_NAME ?= 'cpu-tests:0'
SERVER_IMAGE_NAME ?= 'text-gen-server:0'
GIT_COMMIT_HASH := $(shell git rev-parse --short HEAD)

.PHONY: all
all: help

.PHONY: help
help: ## Display this help
	@awk 'BEGIN{FS=":.*##"; printf("\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n")} /^[-a-zA-Z_0-9\\.]+:.*?##/ {t=$$1; if(!(t in p)){p[t]; printf("  \033[36m%-20s\033[0m %s\n", t, $$2)}}' $(MAKEFILE_LIST)

.PHONY: build
build: ## Build server release image
	docker build --progress=plain --target=server-release --build-arg GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) -t $(SERVER_IMAGE_NAME) .
	docker images

.PHONY: install-server
install-server:
	cd server && make install

.PHONY: install-custom-kernels
install-custom-kernels:
	if [ "$$BUILD_EXTENSIONS" = "True" ]; then \
		cd server/custom_kernels && \
		python setup.py install; \
	else \
		echo "Custom kernels are disabled, you need to set the BUILD_EXTENSIONS environment variable to 'True' in order to build them. (Please read the docs, kernels might not work on all hardware)"; \
	fi

.PHONY: install-router
install-router:
	cd router && cargo install --path .

.PHONY: install-launcher
install-launcher:
	cd launcher && env GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) cargo install --path .

.PHONY: install-launcher-linux
install-launcher-linux:
	cd launcher && env GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-unknown-linux-gnu-gcc cargo install --path . --target=x86_64-unknown-linux-gnu

.PHONY: install
install: install-server install-router install-launcher install-custom-kernels ## Install server, router, launcher, and custom kernels

.PHONY: server-dev
server-dev:
	cd server && make run-dev

.PHONY: router-dev
router-dev:
	cd router && cargo run

.PHONY: run-bloom-560m
run-bloom-560m:
	text-generation-launcher --model-name bigscience/bloom-560m --num-shard 2

.PHONY: run-bloom-560m-quantize
run-bloom-560m-quantize:
	text-generation-launcher --model-name bigscience/bloom-560m --num-shard 2 --dtype-str int8

.PHONY: download-bloom
download-bloom:
	text-generation-server download-weights bigscience/bloom

.PHONY: run-bloom
run-bloom:
	text-generation-launcher --model-name bigscience/bloom --num-shard 8

.PHONY: run-bloom-quantize
run-bloom-quantize:
	text-generation-launcher --model-name bigscience/bloom --num-shard 8 --dtype-str int8

.PHONY: build-test-image
build-test-image: ## Build the test image
	docker build --progress=plain --target=cpu-tests -t $(TEST_IMAGE_NAME) .

.PHONY: check-test-image
check-test-image:
	@docker image inspect $(TEST_IMAGE_NAME) >/dev/null 2>&1 || $(MAKE) build-test-image

.PHONY: integration-tests
integration-tests: check-test-image ## Run integration tests
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HF_HUB_CACHE=/transformers_cache \
		-w /usr/src/integration_tests \
		$(TEST_IMAGE_NAME) make test

.PHONY: python-tests
python-tests: check-test-image ## Run Python tests
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HF_HUB_CACHE=/transformers_cache \
		$(TEST_IMAGE_NAME) pytest -sv --ignore=server/tests/test_utils.py server/tests

.PHONY: clean
clean:
	rm -rf target
