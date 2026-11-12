# Test helper: make FLA deterministic across window sizes and processes. FLA autotunes its Triton kernels by timing, once
# per process and key (l2norm's key includes the number of 2048-token blocks), and some of the candidate configs give
# different (equally valid) bf16 roundings. Pinning every autotuner to one config removes that source of difference, so
# bit-level comparisons only see what the code under test changes. Call before the first FLA kernel runs.
import sys

from triton.runtime.autotuner import Autotuner


def pin_fla_autotune() -> int:
    import fla.ops.gated_delta_rule  # noqa: F401  (imports every kernel the GDN prefill uses)
    pinned = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith("fla"):
            continue
        for obj in vars(module).values():
            for _ in range(4):    # @triton.heuristics wraps the Autotuner, which wraps the JITFunction
                if isinstance(obj, Autotuner):
                    if len(obj.configs) > 1:
                        obj.configs = obj.configs[:1]
                        pinned += 1
                    break
                obj = getattr(obj, "fn", None)
                if obj is None:
                    break
    return pinned
