gen-client:
	# Compile protos
	pip install grpcio-tools==1.60.0 mypy-protobuf==3.5.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir tgis_eval/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=tgis_eval/pb \
		--grpc_python_out=tgis_eval/pb --mypy_out=tgis_eval/pb ../proto/generation.proto
	find tgis_eval/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch tgis_eval/pb/__init__.py

install: gen-client
	pip install pip --upgrade
	pip install -e . --no-cache-dir

image: gen-client
  podman build -t quay.io/wxpe/lm-eval-tgis:0.0.2 .
