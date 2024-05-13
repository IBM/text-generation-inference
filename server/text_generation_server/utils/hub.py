import concurrent
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import LocalEntryNotFoundError
from tqdm import tqdm

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE") == "true"


def weight_hub_files(model_name, extension=".safetensors", revision=None, auth_token=None):
    """Get the safetensors filenames on the hub"""
    exts = [extension] if type(extension) == str else extension
    api = HfApi()
    info = api.model_info(model_name, revision=revision, token=auth_token)
    filenames = [
        s.rfilename for s in info.siblings if any(
            s.rfilename.endswith(ext)
            and len(s.rfilename.split("/")) == 1
            and "arguments" not in s.rfilename
            and "args" not in s.rfilename
            and "training" not in s.rfilename
            for ext in exts
        )
    ]
    return filenames


def weight_files(model_name, extension=".safetensors", revision=None):
    """Get the local safetensors filenames"""
    filenames = weight_hub_files(model_name, extension)
    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(model_name, filename=filename, revision=revision)
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_name} not found in "
                f"{os.getenv('HUGGINGFACE_HUB_CACHE', 'the local cache')}. "
                f"Please run `text-generation-server download-weights {model_name}` first."
            )
        files.append(cache_file)

    return files


def download_weights(model_name, extension=".safetensors", revision=None, auth_token=None):
    """Download the safetensors files from the hub"""
    filenames = weight_hub_files(model_name, extension, revision=revision, auth_token=auth_token)

    download_function = partial(
        hf_hub_download,
        repo_id=model_name,
        local_files_only=False,
        revision=revision,
        token=auth_token,
    )

    print(f"Downloading {len(filenames)} files for model {model_name}")
    executor = ThreadPoolExecutor(max_workers=5)
    futures = [
        executor.submit(download_function, filename=filename) for filename in filenames
    ]
    files = [
        future.result()
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))
    ]

    return files


def get_model_path(model_name: str, revision: Optional[str] = None):
    """ Get path to model dir in local huggingface hub (model) cache"""
    config_file = "config.json"
    err = None
    try:
        config_path = try_to_load_from_cache(
            model_name, config_file,
            revision=revision,
        )
        if config_path is not None:
            return config_path.removesuffix(f"/{config_file}")
    except ValueError as e:
        err = e

    if os.path.isfile(f"{model_name}/{config_file}"):
        return model_name  # Just treat the model name as an explicit model path

    if err is not None:
        raise err

    raise ValueError(f"Weights not found in local cache for model {model_name}")


def local_weight_files(model_path: str, extension=".safetensors"):
    """Get the local safetensors filenames"""
    ext = "" if extension is None else extension
    return glob.glob(f"{model_path}/*{ext}")


def local_index_files(model_path: str, extension=".safetensors"):
    """Get the local .index.json filename"""
    ext = "" if extension is None else extension
    return glob.glob(f"{model_path}/*{ext}.index.json")
