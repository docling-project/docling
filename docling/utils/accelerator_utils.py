import logging
from typing import List, Optional

import torch

from docling.datamodel.accelerator_options import AcceleratorDevice

_log = logging.getLogger(__name__)


def decide_device(
    accelerator_device: str, supported_devices: Optional[List[AcceleratorDevice]] = None
) -> str:
    r"""
    Resolve the device based on the acceleration options and the available devices in the system.

    Rules:
    1. AUTO: Check for the best available device on the system.
    2. User-defined: Check if the device actually exists, otherwise fall-back to CPU
    """
    device = "cpu"

    has_cuda = torch.backends.cuda.is_built() and torch.cuda.is_available()
    has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()

    if supported_devices is not None:
        if has_cuda and AcceleratorDevice.CUDA not in supported_devices:
            _log.info(
                f"Removing CUDA from available devices because it is not in {supported_devices=}"
            )
            has_cuda = False
        if has_mps and AcceleratorDevice.MPS not in supported_devices:
            _log.info(
                f"Removing MPS from available devices because it is not in {supported_devices=}"
            )
            has_mps = False

    if accelerator_device == AcceleratorDevice.AUTO.value:  # Handle 'auto'
        if has_cuda:
            device = "cuda:0"
        elif has_mps:
            device = "mps"

    elif accelerator_device.startswith("cuda"):
        if has_cuda:
            # if cuda device index specified extract device id
            parts = accelerator_device.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                # select cuda device's id
                cuda_index = int(parts[1])
                if cuda_index < torch.cuda.device_count():
                    device = f"cuda:{cuda_index}"
                else:
                    _log.warning(
                        "CUDA device 'cuda:%d' is not available. Fall back to 'CPU'.",
                        cuda_index,
                    )
            elif len(parts) == 1:  # just "cuda"
                device = "cuda:0"
            else:
                _log.warning(
                    "Invalid CUDA device format '%s'. Fall back to 'CPU'",
                    accelerator_device,
                )
        else:
            _log.warning("CUDA is not available in the system. Fall back to 'CPU'")

    elif accelerator_device == AcceleratorDevice.MPS.value:
        if has_mps:
            device = "mps"
        else:
            _log.warning("MPS is not available in the system. Fall back to 'CPU'")

    elif accelerator_device == AcceleratorDevice.CPU.value:
        device = "cpu"

    else:
        _log.warning(
            "Unknown device option '%s'. Fall back to 'CPU'", accelerator_device
        )

    _log.info("Accelerator device: '%s'", device)
    return device
