import os
import json
import hmac
import hashlib
import logging
from typing import Any, Dict, Tuple, Optional
import joblib

from app.core.config import settings
from .schemas import ModelArtifactMetadata

logger = logging.getLogger(__name__)

class ModelArtifactSerializer:
    """
    Safely serializes and deserializes fitted models, feature signatures,
    and preprocessing metadata into self-contained production artifacts.
    """

    def _compute_hmac(self, file_path: str) -> str:
        key = settings.SECRET_KEY.encode("utf-8")
        h = hmac.new(key, digestmod=hashlib.sha256)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

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
        try:
            sig = self._compute_hmac(output_path)
            with open(output_path + ".sig", "w", encoding="utf-8") as f:
                f.write(sig)
        except Exception as e:
            logger.warning(f"Could not write artifact HMAC signature: {e}")
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

        sig_path = artifact_path + ".sig"
        if os.path.exists(sig_path):
            try:
                with open(sig_path, "r", encoding="utf-8") as f:
                    expected_sig = f.read().strip()
                computed_sig = self._compute_hmac(artifact_path)
                if not hmac.compare_digest(expected_sig, computed_sig):
                    raise ValueError("Artifact signature verification failed: potential tampering detected.")
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Could not verify artifact signature: {e}")

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
