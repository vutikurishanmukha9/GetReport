import os
import json
import logging
from typing import Any, Dict, Tuple, Optional
import joblib

from .schemas import ModelArtifactMetadata

logger = logging.getLogger(__name__)

class ModelArtifactSerializer:
    """
    Safely serializes and deserializes fitted models, feature signatures,
    and preprocessing metadata into self-contained production artifacts.
    """

    def save_artifact(
        self,
        model: Any,
        metadata: ModelArtifactMetadata,
        output_path: str,
        preprocessor_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves a self-contained bundle with model and metadata to output_path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        bundle = {
            "model": model,
            "metadata": metadata.to_dict(),
            "preprocessor_meta": preprocessor_meta or {}
        }
        joblib.dump(bundle, output_path, compress=3)
        logger.info(f"Saved model artifact ({metadata.model_name}) to {output_path}")
        return output_path

    def load_artifact(
        self,
        artifact_path: str
    ) -> Tuple[Any, ModelArtifactMetadata, Dict[str, Any]]:
        """
        Loads an artifact and returns (model, metadata, preprocessor_meta).
        Raises FileNotFoundError or ValueError if invalid.
        """
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")

        try:
            bundle = joblib.load(artifact_path)
        except Exception as e:
            raise ValueError(f"Failed to deserialize artifact: {e}")

        if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
            raise ValueError("Corrupt or invalid artifact format: missing model or metadata.")

        meta_dict = bundle["metadata"]
        metadata = ModelArtifactMetadata(
            model_name=meta_dict.get("model_name", "unknown"),
            task_type=meta_dict.get("task_type", "unknown"),
            target_column=meta_dict.get("target_column", ""),
            feature_names=meta_dict.get("feature_names", []),
            performance_metrics=meta_dict.get("performance_metrics", {}),
            created_at_iso=meta_dict.get("created_at_iso", ""),
            version=meta_dict.get("version", "1.0.0"),
        )

        preprocessor_meta = bundle.get("preprocessor_meta", {})
        return bundle["model"], metadata, preprocessor_meta
