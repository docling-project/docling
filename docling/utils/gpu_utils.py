"""Add commentMore actions
Utilities for managing GPU resources.
"""

import logging
import importlib.util

_log = logging.getLogger(__name__)

def clear_gpu_memory():
    """
    Clear GPU memory cache for supported frameworks.
    Currently supports PyTorch and TensorFlow.
    """
    # Try to clear PyTorch CUDA cache
    try:
        if importlib.util.find_spec("torch") is not None:
            import torch
            if torch.cuda.is_available():
                _log.info("Clearing PyTorch CUDA memory cache")
                torch.cuda.empty_cache()
                
            # Handle Apple MPS (Metal Performance Shaders)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _log.info("MPS device detected - note that MPS may not support explicit memory clearing")
                # MPS doesn't have an explicit memory clearing mechanism like CUDA
                # But we can try to force garbage collection
                import gc
                gc.collect()
    except Exception as e:
        _log.warning(f"Failed to clear PyTorch GPU memory: {e}")
    
    # Try to clear TensorFlow GPU memory
    try:
        if importlib.util.find_spec("tensorflow") is not None:
            import tensorflow as tf
            if len(tf.config.list_physical_devices('GPU')) > 0:
                _log.info("Clearing TensorFlow GPU memory")
                for device in tf.config.list_physical_devices('GPU'):
                    try:
                        tf.config.experimental.reset_memory_stats(device)
                    except Exception as e:
                        _log.warning(f"Failed to reset memory stats for device {device}: {e}")
    except Exception as e:
        _log.warning(f"Failed to clear TensorFlow GPU memory: {e}")
    
    # Force Python garbage collection
    try:
        import gc
        gc.collect()
    except Exception as e:
        _log.warning(f"Failed to run garbage collection: {e}")
