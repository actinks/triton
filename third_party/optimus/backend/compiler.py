from dataclasses import dataclass
from typing import Any, Optional

from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend, CUDAOptions


@dataclass(frozen=True)
class OptimusOptions(CUDAOptions):
    # Initial implementation reuses NVIDIA lowering and PTX generation. In a
    # follow-up this can drive Optimus-specific TTGIR/dialect selection.
    optimus_core_policy: Optional[str] = "auto"
    backend_name: str = "optimus"


class OptimusBackend(CUDABackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "optimus"

    def parse_options(self, opts) -> Any:
        base = super().parse_options(opts)
        args = dict(base.__dict__)
        if "optimus_core_policy" in opts and opts["optimus_core_policy"] is not None:
            args["optimus_core_policy"] = opts["optimus_core_policy"]
        args["backend_name"] = "optimus"
        return OptimusOptions(**args)
