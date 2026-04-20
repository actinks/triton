import os

from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import CudaDriver, _cuda_driver_is_active


class OptimusDriver(CudaDriver):
    """
    Optimus runtime shim.

    Initial implementation reuses CUDA runtime plumbing and reports an
    `optimus` target so the compiler can select the Optimus backend while still
    producing NV code.
    """

    @staticmethod
    def is_active():
        # Avoid becoming active by default alongside CUDA (which would make the
        # active-driver auto-detection ambiguous). Activate only when explicitly
        # selected via TRITON_DEFAULT_BACKEND=optimus.
        return os.environ.get("TRITON_DEFAULT_BACKEND") == "optimus" and _cuda_driver_is_active()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        return GPUTarget("optimus", capability, 32)
