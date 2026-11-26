#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gguf==0.19.0",
# ]
# ///

"""Create a LoRA adapter copy with activation and task metadata for fixture recording."""

import argparse
from pathlib import Path

import gguf


def build_adapter(src: Path, dst: Path) -> None:
    """Copy a GGUF LoRA adapter and add activation and task metadata.

    Args:
        src: Path to the source LoRA GGUF file.
        dst: Path to the output LoRA GGUF file.
    """
    reader = gguf.GGUFReader(src, "r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field else "llama"
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = alignment_field.contents()

    overrides: dict[str, tuple[gguf.GGUFValueType, object, gguf.GGUFValueType | None]] = {
        "adapter.alora.invocation_tokens": (
            gguf.GGUFValueType.ARRAY,
            [42, 43],
            gguf.GGUFValueType.UINT32,
        ),
        "adapter.lora.task_name": (gguf.GGUFValueType.STRING, "fixture task", None),
        "adapter.lora.prompt_prefix": (gguf.GGUFValueType.STRING, "fixture prefix", None),
    }

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in overrides:
            vtype, val, sub = overrides.pop(field.name)
            writer.add_key_value(field.name, val, vtype, sub_type=sub)
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if val is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

    for key, (vtype, val, sub) in overrides.items():
        writer.add_key_value(key, val, vtype, sub_type=sub)

    for tensor in reader.tensors:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()


def main() -> None:
    """Parse CLI arguments and write the metadata-modified GGUF adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Source LoRA adapter GGUF path")
    parser.add_argument("dst", type=Path, help="Output LoRA adapter GGUF path")
    args = parser.parse_args()
    build_adapter(args.src, args.dst)


if __name__ == "__main__":
    main()
