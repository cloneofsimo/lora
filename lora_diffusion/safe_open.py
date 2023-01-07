"""
Pure python version of Safetensors safe_open
From https://gist.github.com/Narsil/3edeec2669a5e94e4707aa0f901d2282
"""

import json
import mmap
import os

import torch


class SafetensorsWrapper:
    def __init__(self, metadata, tensors):
        self._metadata = metadata
        self._tensors = tensors

    def metadata(self):
        return self._metadata

    def keys(self):
        return self._tensors.keys()

    def get_tensor(self, k):
        return self._tensors[k]


DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


def create_tensor(storage, info, offset):
    dtype = DTYPES[info["dtype"]]
    shape = info["shape"]
    start, stop = info["data_offsets"]
    return (
        torch.asarray(storage[start + offset : stop + offset], dtype=torch.uint8)
        .view(dtype=dtype)
        .reshape(shape)
    )


def safe_open(filename, framework="pt", device="cpu"):
    if framework != "pt":
        raise ValueError("`framework` must be 'pt'")

    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    size = os.stat(filename).st_size
    storage = torch.ByteStorage.from_file(filename, shared=False, size=size).untyped()
    offset = n + 8

    return SafetensorsWrapper(
        metadata=metadata.get("__metadata__", {}),
        tensors={
            name: create_tensor(storage, info, offset).to(device)
            for name, info in metadata.items()
            if name != "__metadata__"
        },
    )
