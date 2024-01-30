include Makefile-flash-att

gen-server:
	# Compile protos
	pip install grpcio-tools==1.60.0 mypy-protobuf==3.5.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

TORCH_URL := https://download.pytorch.org/whl/cu118
TORCH_VERSION := 2.0.0+cu118

install-torch:
	# Install specific version of torch
	pip install ninja==1.11.1.1 torch==$(TORCH_VERSION) --extra-index-url $(TORCH_URL) --no-cache-dir

install-deepspeed:
	# Install specific version of deepspeed
	pip install deepspeed==0.7.6 --no-cache-dir

install: gen-server install-torch
	pip install pip --upgrade
	pip install -e . --no-cache-dir

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m
