import os
from enum import Enum
from typing import Optional

import typer

from pathlib import Path

app = typer.Typer()


class Quantization(str, Enum):
    bitsandbytes = "bitsandbytes"
    gptq = "gptq"


@app.command()
def serve(
    model_name: str,
    deployment_framework: str,
    dtype: Optional[str] = None,
    quantize: Optional[Quantization] = None,
    # Max seq length, new tokens, batch size and weight
    # only used for PT2 compile warmup
    max_sequence_length: int = 2048,
    max_new_tokens: int = 1024,
    max_batch_size: int = 12,
    batch_safety_margin: int = typer.Option(20, help="Integer from 0-100, a percentage of free GPU memory to hold back as a safety margin to avoid OOM"),
    revision: Optional[str] = None,
    sharded: bool = False,
    cuda_process_memory_fraction: float = 1.0,
    uds_path: Path = "/tmp/text-generation",
):
    from text_generation_server import server
    from text_generation_server.utils.termination import write_termination_log

    if sharded:
        try:
            assert (
                os.getenv("RANK", None) is not None
            ), "RANK must be set when sharded is True"
            assert (
                os.getenv("WORLD_SIZE", None) is not None
            ), "WORLD_SIZE must be set when sharded is True"
            assert (
                os.getenv("MASTER_ADDR", None) is not None
            ), "MASTER_ADDR must be set when sharded is True"
            assert (
                os.getenv("MASTER_PORT", None) is not None
            ), "MASTER_PORT must be set when sharded is True"
        except AssertionError as e:
            write_termination_log(str(e))
            raise e

    try:
        server.serve(
            model_name,
            revision,
            deployment_framework,
            dtype,
            # Downgrade enum into str for easier management later on
            None if quantize is None else quantize.value,
            max_sequence_length,
            max_new_tokens,
            max_batch_size,
            batch_safety_margin,
            sharded,
            cuda_process_memory_fraction,
            uds_path
        )
    except Exception as e:
        # Any exceptions in the blocking server thread here should mean that
        # the server terminated due to an error
        write_termination_log(str(e))
        raise e


@app.command()
def download_weights(
    model_name: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    extension: str = ".safetensors",
    auto_convert: bool = True,
):
    from text_generation_server import utils

    meta_exts = [".json", ".py", ".model", ".md"]

    extensions = extension.split(",")

    if len(extensions) == 1 and extensions[0] not in meta_exts:
        extensions.extend(meta_exts)

    files = utils.download_weights(model_name, extensions, revision=revision, auth_token=token)

    if auto_convert and ".safetensors" in extensions:
        if not utils.local_weight_files(utils.get_model_path(model_name, revision), ".safetensors"):
            if ".bin" not in extensions:
                print(".safetensors weights not found, downloading pytorch weights to convert...")
                utils.download_weights(model_name, ".bin", revision=revision, auth_token=token)

            print(".safetensors weights not found, converting from pytorch weights...")
            convert_to_safetensors(model_name, revision)
        elif not any(f.endswith(".safetensors") for f in files):
            print(".safetensors weights not found on hub, but were found locally. Remove them first to re-convert")
    if auto_convert:
        convert_to_fast_tokenizer(model_name, revision)

@app.command()
def convert_to_onnx(
    model_name: str,
    target_model_name: str,
    revision: Optional[str] = None,
    merge_graphs: Optional[bool] = False,
    optimize: Optional[bool] = False,
    provider: Optional[str] = None,
):
    # Onnx currently isn't included in image used for CI tests
    from text_generation_server.utils import onnx

    onnx.convert_model(
        model_name, target_model_name,
        revision=revision,
        merge_graphs=merge_graphs,
        optimize=optimize,
        provider=provider
    )


@app.command()
def convert_to_safetensors(
    model_name: str,
    revision: Optional[str] = None,
):
    from text_generation_server import utils

    # Get local pytorch file paths
    model_path = utils.get_model_path(model_name, revision)
    local_pt_files = utils.local_weight_files(model_path, ".bin")
    local_pt_index_files = utils.local_index_files(model_path, ".bin")
    if len(local_pt_index_files) > 1:
        print(f"Found more than one .bin.index.json file: {local_pt_index_files}")
        return

    if not local_pt_files:
        print("No pytorch .bin files found to convert")
        return

    local_pt_files = [Path(f) for f in local_pt_files]
    local_pt_index_file = local_pt_index_files[0] if local_pt_index_files else None

    # Safetensors final filenames
    local_st_files = [
        p.parent / f"{p.stem.removeprefix('pytorch_')}.safetensors"
        for p in local_pt_files
    ]

    if any(os.path.exists(p) for p in local_st_files):
        print("Existing .safetensors weights found, remove them first to reconvert")
        return

    try:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=utils.TRUST_REMOTE_CODE
        )
        architecture = config.architectures[0]

        class_ = getattr(transformers, architecture)

        # Name for this variable depends on transformers version
        discard_names = getattr(class_, "_tied_weights_keys", [])
        discard_names.extend(getattr(class_, "_keys_to_ignore_on_load_missing", []))

    except Exception:
        discard_names = []

    if local_pt_index_file:
        local_pt_index_file = Path(local_pt_index_file)
        st_prefix = local_pt_index_file.stem.removeprefix('pytorch_').rstrip('.bin.index')
        local_st_index_file = local_pt_index_file.parent / f"{st_prefix}.safetensors.index.json"

        if os.path.exists(local_st_index_file):
            print("Existing model.safetensors.index.json file found, remove it first to reconvert")
            return

        utils.convert_index_file(local_pt_index_file, local_st_index_file, local_pt_files, local_st_files)

    # Convert pytorch weights to safetensors
    utils.convert_files(local_pt_files, local_st_files, discard_names)


@app.command()
def quantize(
    model_name: str,
    output_dir: str,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    upload_to_model_id: Optional[str] = None,
    percdamp: float = 0.01,
    act_order: bool = False,
):

    if not os.path.exists(model_name):
        download_weights(model_name=model_name, revision=revision)

    from text_generation_server.utils.gptq.quantize import quantize

    quantize(
        model_name=model_name,
        bits=4,
        groupsize=128,
        output_dir=output_dir,
        trust_remote_code=trust_remote_code,
        upload_to_model_id=upload_to_model_id,
        percdamp=percdamp,
        act_order=act_order,
    )


@app.command()
def convert_to_fast_tokenizer(
    model_name: str,
    revision: Optional[str] = None,
    output_path: Optional[str] = None,
):
    from text_generation_server import utils

    # Check for existing "tokenizer.json"
    model_path = utils.get_model_path(model_name, revision)

    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        print(f"Model {model_name} already has a fast tokenizer")
        return

    if output_path is not None:
        if not os.path.isdir(output_path):
            print(f"Output path {output_path} must exist and be a directory")
            return
    else:
        output_path = model_path

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.save_pretrained(output_path)

    print(f"Saved tokenizer to {output_path}")


if __name__ == "__main__":
    app()
