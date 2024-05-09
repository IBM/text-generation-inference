# TGIS eval framework

This directory contains an adapter to run the [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
framework on a TGIS server. We subclass the Model class to collect the benchmark requests and send
them to the TGIS server over gRPC.

## Installing

To install lm-eval with tgis support in your environment run `make install`.


## Running:

To run the benchmark, call it as python module on the command line:
```
python3 -m tgis_eval \
  --model_args server=<host, defaults to localhost>,port=<defaults to 8033> \
  --model=tgis_eval \ 
  --batch_size=16 \ # <-- change the batch size to fit your gpu
  --tasks <task_id>
```

For example, to run the 5 benchmarks that make up the huggingface leaderboard
on a TGIS instance running on hostname `flan-t5-inference-server`:

```
python3 -m tgis_eval \
  --model_args server=flan-t5-inference-server,port=8033 \
  --model=tgis_eval \
  --batch_size=16 \
  --tasks ai2_arc,hellaswag,mmlu,truthfulqa,winogrande,gsm8k 
```

## Building the container

To build the container, run `make image`.


## Running as a job on Kubernetes

You can run tgis-eval as a Kubernetes Job. Locate the `job.yaml` file in this directory
and edit it to adjust it to your needs. Make sure that the hostname is correct and that
the benchmarks to run are the ones you need. Then submit the job with

```
kubectl apply -f job.yaml
```

If you're going to run several rounds of tests, it is recommended to allocate a persistent
volume and mount it in the job pod. This will avoid downloading the same datasets over and
over.
