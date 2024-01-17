gen-client:
	# Compile protos
	pip install grpcio-tools==1.60.0 mypy-protobuf==3.5.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir text_generation_tests/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=text_generation_tests/pb \
		--grpc_python_out=text_generation_tests/pb --mypy_out=text_generation_tests/pb ../proto/generation.proto
	find text_generation_tests/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_tests/pb/__init__.py

install: gen-client
	pip install pip --upgrade
	pip install -e . --no-cache-dir

test:
	pytest -sv text_generation_tests/test_server.py