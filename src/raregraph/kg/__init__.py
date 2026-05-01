from .kg_loader import load_kg, load_hierarchy, disease_name, disease_phenotypes, disease_genes
from .kg_precompute import (
    KGIndex,
    precompute_kg_index,
    IMPORTANCE_WEIGHT,
    FREQUENCY_WEIGHT,
    HIGH_FREQ,
)

__all__ = [
    "load_kg", "load_hierarchy",
    "disease_name", "disease_phenotypes", "disease_genes",
    "KGIndex", "precompute_kg_index",
    "IMPORTANCE_WEIGHT", "FREQUENCY_WEIGHT", "HIGH_FREQ",
]
