"""
gpu_utils.py

CUDA and GPU environment configuration for TensorFlow and DeepCell.
Manages GPU memory growth, device visibility, and VRAM limits to avoid
out-of-memory (OOM) errors during large bioimage batch inference.
"""

import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

def setup_cuda_environment(
    gpu_id: int = 0,
    allow_growth: bool = True,
    memory_limit_mb: Optional[int] = None,
    mixed_precision: bool = False
) -> Dict[str, Any]:
    """
    Initializes and configures the CUDA GPU environment for TensorFlow.

    Args:
        gpu_id: Index of the physical GPU to designate as visible device.
        allow_growth: If True, enables dynamic GPU memory allocation instead of
                      reserving all available VRAM up-front.
        memory_limit_mb: Optional hard cap on GPU VRAM allocation in megabytes.
        mixed_precision: If True, activates fp16 mixed-precision policy.

    Returns:
        dict containing GPU device status and configuration summary.
    """
    status = {
        "cuda_available": False,
        "device_name": "CPU",
        "gpu_count": 0,
        "selected_gpu": None,
        "memory_limit_mb": memory_limit_mb,
        "memory_growth": allow_growth
    }

    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow is not installed in the current environment. Defaulting to CPU mode.")
        return status

    # Query physical GPUs available to TensorFlow
    physical_gpus: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices("GPU")
    status["gpu_count"] = len(physical_gpus)

    if not physical_gpus:
        logger.warning(
            "No CUDA-compatible GPU detected by TensorFlow. Running pipeline in CPU mode. "
            "Inference may be significantly slower."
        )
        return status

    status["cuda_available"] = True

    # Validate requested GPU index
    if gpu_id >= len(physical_gpus) or gpu_id < 0:
        logger.warning(
            f"Requested GPU ID {gpu_id} is out of bounds (found {len(physical_gpus)} GPUs). "
            f"Defaulting to GPU 0."
        )
        gpu_id = 0

    target_gpu = physical_gpus[gpu_id]
    status["selected_gpu"] = target_gpu.name

    try:
        # Restrict TensorFlow to only the targeted GPU
        tf.config.set_visible_devices(target_gpu, "GPU")

        if memory_limit_mb is not None and memory_limit_mb > 0:
            # Set explicit virtual memory limit
            tf.config.set_logical_device_configuration(
                target_gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            logger.info(f"Set VRAM limit on {target_gpu.name} to {memory_limit_mb} MB.")
        elif allow_growth:
            # Enable dynamic memory growth
            tf.config.experimental.set_memory_growth(target_gpu, True)
            logger.info(f"Enabled dynamic memory growth on {target_gpu.name}.")

        # Retrieve device details
        device_details = tf.config.experimental.get_device_details(target_gpu)
        status["device_name"] = device_details.get("device_name", target_gpu.name)
        compute_cap = device_details.get("compute_capability", "Unknown")
        logger.info(f"Successfully configured GPU: {status['device_name']} (Compute Capability: {compute_cap})")

    except RuntimeError as e:
        # Memory growth / device configuration must be set before initializing tensors
        logger.error(f"Error configuring GPU device: {e}")

    # Configure mixed precision if enabled
    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as keras_mp
            keras_mp.set_global_policy("mixed_float16")
            logger.info("Enabled TensorFlow fp16 mixed-precision policy.")
        except Exception as e:
            logger.warning(f"Could not configure mixed precision: {e}")

    return status


def print_hardware_summary(config: Optional[dict] = None) -> None:
    """Prints a formatted summary of system hardware and CUDA accelerator status."""
    hw_config = config.get("hardware", {}) if config else {}
    gpu_id = hw_config.get("gpu_id", 0)
    allow_growth = hw_config.get("allow_growth", True)
    mem_limit = hw_config.get("memory_limit_mb", None)
    mixed_p = hw_config.get("mixed_precision", False)

    status = setup_cuda_environment(
        gpu_id=gpu_id,
        allow_growth=allow_growth,
        memory_limit_mb=mem_limit,
        mixed_precision=mixed_p
    )

    print("\n" + "=" * 60)
    print(" HARDWARE & CUDA ACCELERATION SUMMARY")
    print("=" * 60)
    print(f" CUDA Acceleration Available : {status['cuda_available']}")
    print(f" Physical GPUs Detected      : {status['gpu_count']}")
    print(f" Active Compute Device       : {status['device_name']}")
    if status["cuda_available"]:
        print(f" Dynamic Memory Growth       : {status['memory_growth']}")
        print(f" Memory Limit (MB)           : {status['memory_limit_mb'] or 'Dynamic (Uncapped)'}")
    print("=" * 60 + "\n")

