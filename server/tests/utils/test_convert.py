from pathlib import Path

from text_generation_server.utils.hub import (
    download_weights,
    weight_files,
)

from text_generation_server.utils.convert import convert_files


def test_convert_files():
    model_id = "bigscience/bloom-560m"
    local_pt_files = download_weights(model_id, extension=".bin")
    local_pt_files = [Path(p) for p in local_pt_files]
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors" for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])

    found_st_files = weight_files(model_id)

    assert all([str(p) in found_st_files for p in local_st_files])
