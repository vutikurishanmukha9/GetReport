import pytest
from app.services.analysis_config import AnalysisConfig

def test_analysis_config_domain_customization():
    """Verify AnalysisConfig supports domain-specific configuration overrides."""
    cfg = AnalysisConfig.for_domain("finance")
    assert cfg.domain == "finance"
    assert cfg.enable_correlation is True
    assert cfg.enable_outliers is True
    
    default_cfg = AnalysisConfig.default()
    assert default_cfg is not None
