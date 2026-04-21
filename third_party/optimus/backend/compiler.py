from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.backends.nvidia.compiler import CUDABackend, CUDAOptions


@dataclass(frozen=True)
class OptimusOptions(CUDAOptions):
    # Keep a backend-local option namespace so future Optimus-specific lowering
    # can evolve without touching NVIDIA backend option types.
    optimus_core_policy: Optional[str] = "auto"
    backend_name: str = "optimus"


class OptimusBackend(BaseBackend):
    """
    Relatively independent backend wrapper.

    Initial stage reuses NVIDIA codegen implementation through composition, but
    keeps an Optimus-owned backend class surface for future full customization.
    """

    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "optimus"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        # Delegate to CUDA backend implementation for initial bring-up while
        # avoiding class inheritance coupling.
        cuda_target = GPUTarget("cuda", target.arch, target.warp_size)
        self._delegate = CUDABackend(cuda_target)
        self.binary_ext = self._delegate.binary_ext

    def hash(self) -> str:
        return self._delegate.hash()

    def parse_options(self, opts) -> Any:
        parsed = self._delegate.parse_options(opts)
        args = dict(parsed.__dict__)
        if "optimus_core_policy" in opts and opts["optimus_core_policy"] is not None:
            args["optimus_core_policy"] = opts["optimus_core_policy"]
        args["backend_name"] = "optimus"
        return OptimusOptions(**args)

    def add_stages(self, stages: dict, options: object, language: Language) -> None:
        self._delegate.add_stages(stages, options, language)

    def load_dialects(self, context):
        self._delegate.load_dialects(context)

    def get_module_map(self) -> Dict[str, ModuleType]:
        return self._delegate.get_module_map()

    def get_codegen_implementation(self, options):
        return self._delegate.get_codegen_implementation(options)

    def pack_metadata(self, metadata):
        return self._delegate.pack_metadata(metadata)
