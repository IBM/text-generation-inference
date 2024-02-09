SHELL := /bin/bash

DOCKER_BUILDKIT := 1
TEST_IMAGE_NAME ?= 'cpu-tests:0'
SERVER_IMAGE_NAME ?= 'text-gen-server:0'
GIT_COMMIT_HASH := $(shell git rev-parse --short HEAD)

build:
	docker build --progress=plain --target=server-release --build-arg GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) -t $(SERVER_IMAGE_NAME) .
	docker images

all: help

install-server:
	cd server && make install

install-custom-kernels:
	if [ "$$BUILD_EXTENSIONS" = "True" ]; then cd server/custom_kernels && python setup.py install; else echo "Custom kernels are disabled, you need to set the BUILD_EXTENSIONS environment variable to 'True' in order to build them. (Please read the docs, kernels might not work on all hardware)"; fi

install-router:
	cd router && cargo install --path .

install-launcher:
	cd launcher && env GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) cargo install --path .

.PHONY: install-launcher-linux
install-launcher-linux:
	cd launcher && env GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-unknown-linux-gnu-gcc cargo install --path . --target=x86_64-unknown-linux-gnu

install: install-server install-router install-launcher install-custom-kernels

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

build-test-image:
	docker build --progress=plain --target=cpu-tests -t $(TEST_IMAGE_NAME) .

check-test-image:
	@docker image inspect $(TEST_IMAGE_NAME) >/dev/null 2>&1 || $(MAKE) build-test-image

integration-tests: check-test-image
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HUGGINGFACE_HUB_CACHE=/transformers_cache \
		-e TRANSFORMERS_CACHE=/transformers_cache \
		-w /usr/src/integration_tests \
		$(TEST_IMAGE_NAME) make test

python-tests: check-test-image
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HUGGINGFACE_HUB_CACHE=/transformers_cache \
		-e TRANSFORMERS_CACHE=/transformers_cache \
		$(TEST_IMAGE_NAME) pytest -sv --ignore=server/tests/test_utils.py server/tests

clean:
	rm -rf target

.PHONY: build build-test-image integration-tests python-tests