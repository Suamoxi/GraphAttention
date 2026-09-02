"""Data ownership layer: what exists in the CFD source."""

from .avbp import AVBP_FIELD_CATALOG, AVBPHDF5Dataset, AVBPSampleSpec
from .case_definition import CaseDefinition, load_case_definition, load_case_definitions
from .contracts import (
    FieldCatalog,
    FieldRole,
    FieldSpec,
    FieldSupport,
    Mesh,
    ReferenceScale,
    ReferenceScales,
    ReferenceScope,
    Sample,
)
from .nondimensionalization import ConvectiveNondimensionalizer
from .synthetic import SyntheticMeshDataset

__all__ = [
    "AVBP_FIELD_CATALOG",
    "AVBPHDF5Dataset",
    "AVBPSampleSpec",
    "CaseDefinition",
    "ConvectiveNondimensionalizer",
    "FieldCatalog",
    "FieldRole",
    "FieldSpec",
    "FieldSupport",
    "Mesh",
    "ReferenceScale",
    "ReferenceScales",
    "ReferenceScope",
    "Sample",
    "SyntheticMeshDataset",
    "load_case_definition",
    "load_case_definitions",
]
