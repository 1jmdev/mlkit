"""Launching of a compiled Triton specialization without per-call argument binding.

Triton derives a specialization key from every argument on every call, which
costs more than a small kernel takes to run. A layer that always launches the
same kernel with the same constants can keep the compiled kernel and launch it
directly.

A kernel launched this way must be declared with ``do_not_specialize`` and
``do_not_specialize_on_alignment`` for all runtime arguments, because a reused
specialization must not depend on argument values or pointer alignment. The
caller is responsible for reusing a compiled kernel only with the tensor
dtypes and constant arguments it was compiled for.
"""

from typing import Any

from triton.runtime import driver

REQUIRED_ATTRIBUTES = ("run", "function", "packed_metadata")


def compile_and_launch(kernel: Any, grid: tuple[int, int], arguments: tuple[Any, ...]) -> Any:
    """Launch through the public Triton path and return the reusable compiled kernel.

    Returns ``None`` when this Triton release does not expose what a direct launch needs.
    """
    compiled = kernel[grid](*arguments)
    if all(hasattr(compiled, name) for name in REQUIRED_ATTRIBUTES):
        return compiled
    return None


def launch(compiled: Any, grid: tuple[int, int], arguments: tuple[Any, ...]) -> None:
    """Launch a compiled kernel on the current CUDA stream.

    ``arguments`` are the positional arguments of the kernel function, including
    its constant arguments. A ``TypeError`` means the launcher signature of this
    Triton release differs; it is raised before anything is launched.
    """
    stream = driver.active.get_current_stream(driver.active.get_current_device())
    compiled.run(
        grid[0],
        grid[1],
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *arguments,
    )
