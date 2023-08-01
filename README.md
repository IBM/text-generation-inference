## Text Generation Inference Server

This repo is an early fork of https://github.com/huggingface/text-generation-inference.

It was developed internally in IBM and diverged somewhat from the original repo, but we have tried to keep aligned as much as possible - pulling in relevant upstream changes and contributing features/improvements back. This is an ongoing process and there's a backlog in both directions which we hope to work through quickly.

It's not clear yet whether/when we will be able to reconcile the repos completely because goals/priorities of the projects may differ in some cases. Decisions related to design or implementation approaches are also sometimes different.

Some features here are similar/equivalent to those in the upstream repo but are implemented differently. This is generally because we had implemented them first internally, and they were subsequently implemented independently in the upstream repo before we had had a chance to contribute back that feature. One example of this is response streaming. In some other cases we had opened PRs against the upstream repo but the maintainers decided to reimplement in another way.

Some upstream changes were intentionally not pulled in because they weren't required for our current usage, for example LLaMA model support. Others we have just not caught up with and ported back yet.

---

### Some of the differences in this repo
- gRPC front-end interface instead of REST, different arrangement of API parameters
- Support for batch inputs in the API
- Tokenization API
- More comprehensive CI tests (excluding flash attention impls)
- Configurable inference engine abstraction
- UBI based container image


### Some of the changes we're contributing back or plan to contribute back soon
- Dynamic batch sizing (upstream [PR](https://github.com/huggingface/text-generation-inference/pull/210))
- Detokenization and stopping evaluation done on rust side (upstream [PR](https://github.com/huggingface/text-generation-inference/pull/138))
  - Includes parity of streaming and non-streaming output
- More granular "extra token details" options
- "top n" candidate token output option
- Return token ranks in addition to logprobs
- Length penalty, min new tokens parameters
- Optimum integration (for Onnx Runtime and BetterTransformer)
- Support for tuned prompts, as trained via the PEFT library (not for flash or sharded impls yet)
- Support for PyTorch 2 compile

---

### Run the integration tests

```shell
make integration-tests
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
text-generation-server download-weights --extension ".json,.bin,.md,.model,.py" model_name
```
where `model_name` is the name of the model on the HF hub. Ensure that it's run with the same mounted directory and `TRANSFORMERS_CACHE` and `HUGGINGFACE_HUB_CACHE` environment variables, and that it has write access to this mounted filesystem. 

### Running sharded models (Tensor Parallel)

The following model types can currently be run in sharded mode where the weights are split across multiple GPUs:
- BLOOM
- T5
- RefinedWeb (Falcon) (*)
- LLaMA (*)

(*) These require GPUs that support Flash Attention such as A100, A10

Model weights must be in `safetensors` format. These are available on the HF hub for some models and can be downloaded like:
```shell
text-generation-server download-weights --extension ".json,.safetensors,.md,.model,.py" model_name
```
or otherwise can be converted from PyTorch `.bin` weights:
```shell
text-generation-server convert-to-safetensors model_name
```

Then ensure that the `CUDA_VISIBLE_DEVICES` environment variable is set appropriately (e.g. "0,1" to use the first two GPUs). The number of GPUs to use will be inferred from this or else can be set explicitly with the `NUM_GPUS` environment variable.

### TLS configuration

TLS can be enabled in the TGIS containers via the following env vars:

- `TLS_CERT_PATH` - path to cert
- `TLS_KEY_PATH` - path to private key
- `TLS_CLIENT_CA_CERT_PATH` - path to ca cert to use for client authentication (optional, client auth not enabled if omitted)

These paths can reference mounted secrets containing the certs.

### Metrics

Prometheus metrics are exposed on the same port as the health probe endpoint (default 3000), at `/metrics`.

They are all prefixed with `tgi_`. A full list with descriptions will be added here soon.