## Text Generation Inference Server

This repo is an early fork of https://github.com/huggingface/text-generation-inference.

It was developed internally in IBM and diverged somewhat from the original repo, but we tried to keep it aligned as much as possible - pulling in relevant upstream changes and contributing features/improvements back.

A number of features here are similar/equivalent but are implemented differently. This is generally because we had implemented them first internally, and then either they were implemented independently in the upstream repo before we had had a chance to contribute the feature back (such as response streaming), or we had opened a PR against the upstream repo but the maintainers decided to reimplement in another way.

Some upstream changes were intentionally not pulled in because they weren't required for our current usage, for example OPT/Galactica model support. And we have stopped pulling in any upstream work after TGI version 1.0, following which the Apache 2.0 OSS license doesn't apply.

---

### Some of the features in this repo not in HF TGI as of v1.0
- gRPC front-end interface instead of REST, different arrangement of API parameters
- Support for batch inputs in the API
- Independent tokenization API
- More comprehensive CI tests (excluding GPU / flash attention impls)
- Configurable inference engine abstraction
- UBI based container image
- More sophisticated dynamic batch sizing (upstream [PR](https://github.com/huggingface/text-generation-inference/pull/210))
- Detokenization and stopping evaluation done on rust side (upstream [PR](https://github.com/huggingface/text-generation-inference/pull/138))
  - Includes parity of streaming and non-streaming output
- More granular "extra token details" options
- "top n" candidate token output option
- Return token ranks in addition to logprobs
- Length penalty, min new tokens parameters
- Option to omit stop sequence from response text, include matched stop sequence in response
- Optimum integration (for Onnx Runtime and BetterTransformer)
- Support for tuned prompts, as trained via the PEFT library (not for sharded impls yet)
- Vectorized decoding for non-flash model deployments (in addition to flash)
- Support for PyTorch 2 compile
- Exllama V2 kernel for GPTQ quantized models

---

### Run the integration tests

```shell
make build-test-image integration-tests
```

### Build the final container image

```shell
make build
```

### Deploy model in Kubernetes/OpenShift

```shell
cd deployment
./deploy_model.sh <model-subdir-name>
```

---

### Model configuration

When deploying TGIS, the `MODEL_NAME` environment variable can contain either the full name of a model on the Hugging Face hub (such as `google/flan-ul2`) or an absolute path to a (mounted) model directory inside the container. In the former case, the `TRANSFORMERS_CACHE` and `HUGGINGFACE_HUB_CACHE` environment variables should be set to the path of a mounted directory containing a local HF hub model cache, see [this](deployment/base/patches/pvcs/pvc.yaml) kustomize patch as an example.

### Downloading model weights

TGIS will not download model data at runtime. To populate the local HF hub cache with models so that it can be used per above, the image can be run with the following command:
```shell
text-generation-server download-weights model_name
```
where `model_name` is the name of the model on the HF hub. Ensure that it's run with the same mounted directory and `TRANSFORMERS_CACHE` and `HUGGINGFACE_HUB_CACHE` environment variables, and that it has write access to this mounted filesystem. 

This will attempt to download weights in `.safetensors` format, and if those aren't in the HF hub will download pytorch `.bin` weights and then convert them to `.safetensors`.

If needed, specific file extensions can be downloaded by using the `--extension` option, for example:
```shell
text-generation-server download-weights --extension ".json,.bin,.md,.model,.py" model_name
```

### Converting weights to `safetensors` format

`.saftensors` weights are now required for many models, in particular:
- When using the optimized flash attention mode (`FLASH_ATTENTION=true`) - this is currently supported for Llama, Falcon, Starcoder and GPT-NeoX based models, on newer GPUs
- When using tensor parallel (see below)
- Also recommended for BLOOM and T5 type models generally

They can be downloaded directly from the huggingface hub for some models. As explained above, the download command by default will download and convert them from PyTorch weights if safetensors weights aren't available.

To convert from pre-existing PyTorch `.bin` weights:
```shell
text-generation-server convert-to-safetensors model_name
```

### Running sharded models (Tensor Parallel)

The following model types can currently be run in sharded mode where the weights are split across multiple GPUs:
- BLOOM
- T5
- GPT-NeoX
- RefinedWeb (Falcon) (*)
- LLaMA (*)
- GPT-BigCode (Starcoder) (*)

(*) These require GPUs that support Flash Attention such as A100, A10

1. Ensure that the model weights are in `safetensors format (see above)
2. Ensure that the `CUDA_VISIBLE_DEVICES` environment variable is set appropriately (e.g. "0,1" to use the first two GPUs). The number of GPUs to use will be inferred from this or else can be set explicitly with the `NUM_GPUS` environment variable.
3. Set the environment variable `DEPLOYMENT_FRAMEWORK=hf_custom_tp`

### TLS configuration

TLS can be enabled in the TGIS containers via the following env vars:

- `TLS_CERT_PATH` - path to cert
- `TLS_KEY_PATH` - path to private key
- `TLS_CLIENT_CA_CERT_PATH` - path to ca cert to use for client authentication (optional, client auth not enabled if omitted)

These paths can reference mounted secrets containing the certs.

### Metrics

Prometheus metrics are exposed on the same port as the health probe endpoint (default 3000), at `/metrics`.

They are all prefixed with `tgi_`. Descriptions will be added to the table below soon.

| Metric                                     | Kind        | Labels                                                                       | Description                                                                                |
|--------------------------------------------|-------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `tgi_request_count`                        | `counter`   | kind = "single" or "batch" or "stream"                                       | Count of generate requests (batch of n counts as 1)                                        |
| `tgi_request_input_count`                  | `counter`   |                                                                              | Count of generate request inputs (batch of n counts as n)                                  |
| `tgi_request_failure`                      | `counter`   | err                                                                          | Count of failed requests, segmented by error type                                          |
| `tgi_request_success`                      | `counter`   | stop_reason, kind = "single" or "batch" or "stream"                          | Count of successful requests                                                               |
| `tgi_request_max_new_tokens`               | `histogram` |                                                                              | Value of `max_new_tokens` request parameter                                                |
| `tgi_request_input_length`                 | `histogram` |                                                                              | Request input length in tokens                                                             |
| `tgi_request_raw_input_length`             | `histogram` |                                                                              | Raw request input length in tokens (including "too long" validation failures)              |
| `tgi_request_mean_time_per_token_duration` | `histogram` |                                                                              | Mean time per token, per request (in seconds)                                              |
| `tgi_request_validation_duration`          | `histogram` |                                                                              | Request validation time (in seconds)                                                       |
| `tgi_request_queue_duration`               | `histogram` |                                                                              | Request time spent in queue (in seconds)                                                   |
| `tgi_request_generated_tokens`             | `histogram` |                                                                              | Number of tokens generated for request                                                     |
| `tgi_request_total_tokens`                 | `histogram` |                                                                              | Total sequence length of request (input tokens + generated tokens)                         |
| `tgi_request_duration`                     | `histogram` |                                                                              | End-to-end generate request duration (in seconds)                                          |
| `tgi_request_inference_duration`           | `histogram` |                                                                              | Duration of inferencing portion of request (in seconds)                                    |
| `tgi_batch_concatenation_count`            | `counter`   |                                                                              | How many times the continuous batcher combined a new batch into the running batch          |
| `tgi_batch_inference_count`                | `counter`   | method = "prefill" or "next_token"                                           | Count of model forward-pass iterations                                                     |
| `tgi_batch_inference_success`              | `counter`   | method = "prefill" or "next_token"                                           | Count of successful model forward-pass iterations                                          |
| `tgi_batch_inference_failure`              | `counter`   | method = "prefill" or "next_token", reason = "oom", "connection", or "error" | Count of failed model forward-pass iterations                                              |
| `tgi_batch_inference_batch_size`           | `histogram` | method = "prefill" or "next_token"                                           | Batch size for each forward-pass iteration                                                 |
| `tgi_batch_inference_duration`             | `histogram` | method = "prefill" or "next_token", makeup                                   | Time taken for each forward-pass iteration (in seconds)                                    |
| `tgi_batch_inference_forward_duration`     | `histogram` | method = "prefill" or "next_token", makeup                                   | Time taken for each model `forward()` method invocation (in seconds)                       |
| `tgi_batch_inference_tokproc_duration`     | `histogram` | method = "prefill" or "next_token", makeup                                   | Rust-side token-processing time per model forward-pass iteration (in secs)                 |
| `tgi_batch_next_tokens`                    | `histogram` |                                                                              | Total number of tokens included in prefill batch (including padding)                       |
| `tgi_batch_current_size`                   | `gauge`     |                                                                              | Current batch size                                                                         |
| `tgi_batch_input_tokens`                   | `gauge`     |                                                                              | Total number of input tokens in current batch, including padding tokens                    |
| `tgi_batch_max_remaining_tokens`           | `gauge`     |                                                                              | Maximum number of to-be-generated tokens of requests in current batch                      |
| `tgi_queue_size`                           | `gauge`     |                                                                              | Current number of queued requests                                                          |
| `tgi_queue_jump`                           | `counter`   |                                                                              | Count of queue-jumps when batch filling                                                    |
| `tgi_granular_batch_addition`              | `counter`   |                                                                              | Count of batch additions due to granular analysis that would not otherwise fit             |
| `tgi_prefill_weight_limit_exceeded`        | `counter`   |                                                                              | Count of times the max prefill weight is reached during new batch construction             |
| `tgi_prefill_padding_limit_exceeded`       | `counter`   |                                                                              | Count of times the max prefill padding proportion is reached during new batch construction |
| `tgi_prompt_load_failure`                  | `counter`   |                                                                              | Count of failed tuned soft-prompt loads                                                    |
| `tgi_prompt_load_duration`                 | `histogram` |                                                                              | Time taken to JIT-load tuned soft-prompt in seconds (includes count of such loads)         |
| `tgi_tokenize_request_count`               | `counter`   |                                                                              | Count of tokenize requests (batch of n counts as 1)                                        |
| `tgi_tokenize_request_input_count`         | `counter`   |                                                                              | Count of tokenize request inputs (batch of n counts as n)                                  |
| `tgi_tokenize_request_tokens`              | `histogram` |                                                                              | Count of tokenized tokens per tokenize request                                             |
| `tgi_tokenize_request_duration`            | `histogram` |                                                                              | Tokenize request duration (in seconds)                                                     |
