SHELL := /bin/bash

build:
	DOCKER_BUILDKIT=1 docker build --progress=plain --target=server-release -t text-gen-server:0 .
	docker images

all: help

install-server:
	cd server && make install

install-custom-kernels:
	if [ "$$BUILD_EXTENSIONS" = "True" ]; then cd server/custom_kernels && python setup.py install; else echo "Custom kernels are disabled, you need to set the BUILD_EXTENSIONS environment variable to 'True' in order to build them. (Please read the docs, kernels might not work on all hardware)"; fi

install-router:
	cd router && cargo install --path .

install-launcher:
	cd launcher && cargo install --path .

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
	DOCKER_BUILDKIT=1 docker build --progress=plain --target=cpu-tests -t cpu-tests:0 .

integration-tests: build-test-image
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HUGGINGFACE_HUB_CACHE=/transformers_cache \
		-e TRANSFORMERS_CACHE=/transformers_cache -w /usr/src/integration_tests cpu-tests:0 make test

python-tests: build-test-image
	mkdir -p /tmp/transformers_cache
	docker run --rm -v /tmp/transformers_cache:/transformers_cache \
		-e HUGGINGFACE_HUB_CACHE=/transformers_cache \
		-e TRANSFORMERS_CACHE=/transformers_cache cpu-tests:0 pytest -sv --ignore=server/tests/test_utils.py server/tests

clean:
	rm -rf target

.PHONY: build build-test-image integration-tests python-tests