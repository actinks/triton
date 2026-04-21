import os

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.backends.nvidia.driver import CudaDriver, _cuda_driver_is_active


class OptimusDriver(DriverBase):
    """
    Relatively independent runtime wrapper.

    Initial stage reuses CUDA runtime functions through composition while
    exposing an Optimus-specific driver identity and target reporting.
    """

    def __init__(self):
        self._delegate = CudaDriver()
        self.utils = self._delegate.utils
        self.launcher_cls = self._delegate.launcher_cls

    @staticmethod
    def is_active():
        # Avoid auto-select conflict when CUDA driver is also active.
        return os.environ.get("TRITON_DEFAULT_BACKEND") == "optimus" and _cuda_driver_is_active()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return self._delegate.map_python_to_cpp_type(ty)

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        return GPUTarget("optimus", capability, 32)

    def get_active_torch_device(self):
        return self._delegate.get_active_torch_device()

    def get_benchmarker(self):
        return self._delegate.get_benchmarker()

    def allocate_default_profile_scratch(self, size: int, alignment: int, stream):
        return self._delegate.allocate_default_profile_scratch(size, alignment, stream)

    def get_current_stream(self, device):
        return self._delegate.get_current_stream(device)

    def get_current_device(self):
        return self._delegate.get_current_device()

    def set_current_device(self, device):
        return self._delegate.set_current_device(device)

    def get_device_capability(self, device):
        return self._delegate.get_device_capability(device)

    # Optional CUDA-specific helpers forwarded for compatibility.
    def get_device_interface(self):
        return self._delegate.get_device_interface()

    def get_empty_cache_for_benchmark(self):
        return self._delegate.get_empty_cache_for_benchmark()

    def clear_cache(self, cache):
        return self._delegate.clear_cache(cache)
