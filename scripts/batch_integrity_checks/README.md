## Batching Integrity Verification

This script can be used to help verify correctness of models running in TGIS with dynamic/continuous batching.

False negatives are still possible since there is some amount of inconsistency expected due to the fixed precision floating point operations, for example in float16 and especially bfloat16.


First compile protobuf stubs for the external API:
```
python -m grpc_tools.protoc -I../../proto --python_out=. --grpc_python_out=. generation.proto
```

Then run the script, it currently assumes TGIS is running locally/port-forwarded on port 8033.
```
python batching_integrity_checks.py
```
