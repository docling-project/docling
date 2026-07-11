import logging
from contextlib import suppress
from typing import List, Optional

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.exceptions import AcceleratorDeviceNotAvailableError

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
    has_torch = False
    with suppress(ImportError):
        import torch

        has_torch = True

    device = "cpu"

    if has_torch:
        has_cuda = torch.backends.cuda.is_built() and torch.cuda.is_available()
        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    else:
        has_cuda = has_mps = has_xpu = False

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
        if has_xpu and AcceleratorDevice.XPU not in supported_devices:
            _log.info(
                f"Removing XPU from available devices because it is not in {supported_devices=}"
            )
            has_xpu = False

    if accelerator_device == AcceleratorDevice.AUTO.value:  # Handle 'auto'
        if has_cuda:
            device = "cuda:0"
        elif has_mps:
            device = "mps"
        elif has_xpu:
            device = "xpu"

    elif accelerator_device.startswith("cuda"):
        if (
            supported_devices is not None
            and AcceleratorDevice.CUDA not in supported_devices
        ):
            raise AcceleratorDeviceNotAvailableError(
                f"CUDA is not supported by this model. Supported devices: {[d.value for d in supported_devices]}"
            )

        if has_cuda:
            # if cuda device index specified extract device id
            parts = accelerator_device.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                # select cuda device's id
                cuda_index = int(parts[1])
                if cuda_index < torch.cuda.device_count():
                    device = f"cuda:{cuda_index}"
                else:
                    raise AcceleratorDeviceNotAvailableError(
                        f"CUDA device 'cuda:{cuda_index}' is not available. "
                        f"Available CUDA devices: 0-{torch.cuda.device_count() - 1}"
                    )
            elif len(parts) == 1:  # just "cuda"
                device = "cuda:0"
            else:
                raise AcceleratorDeviceNotAvailableError(
                    f"Invalid CUDA device format '{accelerator_device}'. "
                    f"Use 'cuda' or 'cuda:N' where N is a valid device index."
                )
        else:
            raise AcceleratorDeviceNotAvailableError(
                "CUDA is not available in the system. "
                "Please ensure PyTorch with CUDA support is installed, or use --device auto/cpu."
            )

    elif accelerator_device == AcceleratorDevice.MPS.value:
        if (
            supported_devices is not None
            and AcceleratorDevice.MPS not in supported_devices
        ):
            raise AcceleratorDeviceNotAvailableError(
                f"MPS is not supported by this model. Supported devices: {[d.value for d in supported_devices]}"
            )

        if has_mps:
            device = "mps"
        else:
            raise AcceleratorDeviceNotAvailableError(
                "MPS is not available in the system. "
                "Please ensure you are running on Apple Silicon with MPS support, or use --device auto/cpu."
            )

    elif accelerator_device == AcceleratorDevice.XPU.value:
        if (
            supported_devices is not None
            and AcceleratorDevice.XPU not in supported_devices
        ):
            raise AcceleratorDeviceNotAvailableError(
                f"XPU is not supported by this model. Supported devices: {[d.value for d in supported_devices]}"
            )

        if has_xpu:
            device = "xpu"
        else:
            raise AcceleratorDeviceNotAvailableError(
                "XPU is not available in the system. "
                "Please ensure PyTorch with Intel XPU support is installed, or use --device auto/cpu."
            )

    elif accelerator_device == AcceleratorDevice.CPU.value:
        device = "cpu"

    else:
        raise AcceleratorDeviceNotAvailableError(
            f"Unknown device option '{accelerator_device}'. "
            f"Valid options are: auto, cpu, cuda, mps, xpu, or cuda:N"
        )

    _log.info("Accelerator device: '%s'", device)
    return device
