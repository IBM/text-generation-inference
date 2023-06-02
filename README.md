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