SHELL := /bin/bash

DOCKER_BUILDKIT := 1
TEST_IMAGE_NAME ?= 'cpu-tests:0'
SERVER_IMAGE_NAME ?= 'text-gen-server:0'

build: ## Build server release image.
	docker build --progress=plain --target=server-release -t  $(SERVER_IMAGE_NAME) .
	docker images

all: help

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN{FS=":.*##"; printf("\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n")} /^[-a-zA-Z_0-9\\.]+:.*?##/ {t=$$1; if(!(t in p)){p[t]; printf("  \033[36m%-20s\033[0m %s\n", t, $$2)}}' $(MAKEFILE_LIST)

install-server:
	cd server && make install

install-custom-kernels:
	if [ "$$BUILD_EXTENSIONS" = "True" ]; then cd server/custom_kernels && python setup.py install; else echo "Custom kernels are disabled, you need to set the BUILD_EXTENSIONS environment variable to 'True' in order to build them. (Please read the docs, kernels might not work on all hardware)"; fi

install-router:
	cd router && cargo install --path .

install-launcher:
	cd launcher && cargo install --path .

install: install-server install-router install-launcher install-custom-kernels ## Install server, router, launcher, and custom kernels.

server-dev:
	cd server && make run-dev

router-dev:
	cd router && cargo run

run-bloom-560m:
	text-generation-launcher --model-name bigscience/bloom-560m --num-shard 2

run-bloom-560m-quantize:
	text-generation-launcher --model-name bigscience/bloom-560m --num-shard 2 --dtype-str int8

download-bloom:
	text-generation-server download-weights bigscience/bloom

run-bloom:
	text-generation-launcher --model-name bigscience/bloom --num-shard 8

run-bloom-quantize:
	text-generation-launcher --model-name bigscience/bloom --num-shard 8 --dtype-str int8

build-test-image: ## Build the test image.
	docker build --progress=plain --target=cpu-tests -t $(TEST_IMAGE_NAME) .

integration-tests: build-test-image  ## Run integration tests.
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HUGGINGFACE_HUB_CACHE=/transformers_cache \
		-e TRANSFORMERS_CACHE=/transformers_cache \
		-w /usr/src/integration_tests \
		$(TEST_IMAGE_NAME) make test

python-tests: build-test-image  ## Run Python tests.
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HUGGINGFACE_HUB_CACHE=/transformers_cache \
		-e TRANSFORMERS_CACHE=/transformers_cache \
		$(TEST_IMAGE_NAME) pytest -sv --ignore=server/tests/test_utils.py server/tests


.PHONY: build build-test-image integration-tests python-tests