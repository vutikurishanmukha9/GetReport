import pytest
from app.services.transformation_dag import _get_reversibility

def test_transformation_dag_reversibility_classification():
    """Verify that every transformation operation is mapped to an accurate reversibility level."""
    assert _get_reversibility("rename_columns")[0] == "full"
    assert _get_reversibility("drop_null_rows")[0] == "partial"
    assert _get_reversibility("remove_duplicates")[0] == "partial"
    assert _get_reversibility("type_conversion")[0] == "partial"
    assert _get_reversibility("drop_empty_columns")[0] == "partial"
    assert _get_reversibility("fill_null_mean")[0] == "none"
    assert _get_reversibility("fill_null_median")[0] == "none"
    assert _get_reversibility("fill_null_mode")[0] == "none"
    assert _get_reversibility("fill_null_value")[0] == "none"
    assert _get_reversibility("replace_outliers")[0] == "none"
    assert _get_reversibility("custom")[0] == "none"
    assert _get_reversibility("unknown_random_op")[0] == "none"
