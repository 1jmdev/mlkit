"""Model wrappers, architecture discovery and conversion reports."""

from mlkit.models.architecture import (
    ArchitectureAdapter,
    adapter,
    architecture_adapter,
    identify_siblings,
    normalize_affine_layers,
)
from mlkit.models.loading import load
from mlkit.models.model import Model, QModel
from mlkit.models.module_utilities import (
    extract_hidden,
    module_device,
    preserve_input_processing,
    weight_name,
)
from mlkit.models.reports import BlockPassReport, LayerReport

__all__ = [
    "ArchitectureAdapter",
    "BlockPassReport",
    "LayerReport",
    "Model",
    "QModel",
    "adapter",
    "architecture_adapter",
    "extract_hidden",
    "identify_siblings",
    "load",
    "module_device",
    "normalize_affine_layers",
    "preserve_input_processing",
    "weight_name",
]
