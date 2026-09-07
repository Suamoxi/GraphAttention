"""Data ownership layer: what exists in the CFD source."""

from .avbp import AVBP_FIELD_CATALOG, AVBPHDF5Dataset, AVBPSampleSpec
from .case_definition import CaseDefinition, load_case_definition, load_case_definitions
from .collate import MicrobatchBudget, PackedBatch, pack_samples, partition_samples_by_budget
from .contracts import (
    FieldCatalog,
    FieldRole,
    FieldSpec,
    FieldSupport,
    Mesh,
    ReferenceScale,
    ReferenceScales,
    ReferenceScope,
    RegimeParameter,
    RegimeParameters,
    Sample,
)
from .nondimensionalization import ConvectiveNondimensionalizer
from .slice_pt import PrecomputedSlicePTDataset, reconstruct_slice_coordinates
from .splits import SplitManifest, SplitName, make_grouped_split_manifest
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
    "MicrobatchBudget",
    "PackedBatch",
    "PrecomputedSlicePTDataset",
    "ReferenceScale",
    "ReferenceScales",
    "ReferenceScope",
    "RegimeParameter",
    "RegimeParameters",
    "Sample",
    "SplitManifest",
    "SplitName",
    "SyntheticMeshDataset",
    "load_case_definition",
    "load_case_definitions",
    "make_grouped_split_manifest",
    "pack_samples",
    "partition_samples_by_budget",
    "reconstruct_slice_coordinates",
]
