#!/usr/bin/env python3
"""Validate the Python and CUDA environment used by PersonalCoder."""

from importlib.metadata import PackageNotFoundError, version
import platform
import sys

import torch


PACKAGES = ("transformers", "bitsandbytes", "peft", "datasets", "trl")


def package_version(package: str) -> str:
    """Return an installed package version without importing the package."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA runtime: {torch.version.cuda or 'not available'}")

    if cuda_available:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        total_gib = properties.total_memory / 1024**3
        print(f"GPU: {properties.name}")
        print(f"GPU total memory: {total_gib:.2f} GiB")
    else:
        print("GPU: not available")
        print("GPU total memory: not available")

    for package in PACKAGES:
        print(f"{package}: {package_version(package)}")

    if not cuda_available:
        print(
            "ERROR: CUDA is not available. PersonalCoder QLoRA training requires "
            "a CUDA-capable GPU and a CUDA-enabled PyTorch installation.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
