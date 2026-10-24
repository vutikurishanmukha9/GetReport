"""
Understand-Anything Integration Package for GetReport Backend.
Provides:
- Architectural Layer Detection (detect_dataset_layers)
- Guided Audit Tour Generation (generate_dataset_audit_tour)
- Semantic Fingerprinting & Drift Detection (compute_dataset_fingerprint, compare_dataset_fingerprints)
- Schema Exporter (.ua/knowledge-graph.json export)
"""
from app.services.understand_anything.layer_detector import detect_dataset_layers, CANONICAL_DATA_LAYERS
from app.services.understand_anything.tour_generator import generate_dataset_audit_tour
from app.services.understand_anything.fingerprint import compute_dataset_fingerprint, compare_dataset_fingerprints
from app.services.understand_anything.exporter import export_to_understand_anything_schema, write_understand_anything_data_dir

__all__ = [
    "detect_dataset_layers",
    "CANONICAL_DATA_LAYERS",
    "generate_dataset_audit_tour",
    "compute_dataset_fingerprint",
    "compare_dataset_fingerprints",
    "export_to_understand_anything_schema",
    "write_understand_anything_data_dir",
]
