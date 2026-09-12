import pytest
import polars as pl
from app.services.concept_synthesizer import (
    VirtualConceptSynthesizer,
    ConceptDeclaration,
)
from app.services.transformation_dag import create_dag


def test_synthesize_polars_expression_simple_arithmetic():
    synth = VirtualConceptSynthesizer()
    schema = {"revenue": "Float64", "cost": "Float64"}
    decl = ConceptDeclaration(
        concept_name="profit",
        formula_or_intent="revenue - cost",
    )

    expr = synth.synthesize_polars_expression(schema, decl)
    assert "pl.col('revenue')" in expr
    assert "pl.col('cost')" in expr


def test_synthesize_polars_expression_complex_formula():
    synth = VirtualConceptSynthesizer()
    schema = {"price": "Float64", "qty": "Int64", "discount": "Float64"}
    decl = ConceptDeclaration(
        concept_name="net_sales",
        formula_or_intent="(price * qty) * (1 - discount)",
    )

    expr = synth.synthesize_polars_expression(schema, decl)
    assert "pl.col('price')" in expr
    assert "pl.col('qty')" in expr
    assert "pl.col('discount')" in expr


def test_synthesize_polars_expression_blocks_unsafe_calls():
    synth = VirtualConceptSynthesizer()
    schema = {"col1": "Float64"}
    decl = ConceptDeclaration(
        concept_name="evil_col",
        formula_or_intent="__import__('os').system('echo pwned')",
    )

    with pytest.raises((ValueError, Exception)):
        synth.synthesize_polars_expression(schema, decl)


def test_apply_and_record_concept_augments_df_and_dag():
    synth = VirtualConceptSynthesizer()
    df = pl.DataFrame({
        "revenue": [100.0, 200.0, 300.0],
        "cogs": [40.0, 90.0, 150.0],
    })

    dag = create_dag(df, dataset_name="sales.csv")
    decl = ConceptDeclaration(
        concept_name="gross_margin",
        formula_or_intent="(revenue - cogs) / revenue",
        description="Gross margin ratio",
    )

    new_df, node = synth.apply_and_record_concept(df, decl, dag)

    assert "gross_margin" in new_df.columns
    assert new_df.width == 3
    assert new_df.height == 3
    assert round(new_df["gross_margin"][0], 2) == 0.60
    assert round(new_df["gross_margin"][1], 2) == 0.55
    assert round(new_df["gross_margin"][2], 2) == 0.50

    # Verify DAG recording
    assert node.operation == "concept_derivation"
    assert node.target_column == "gross_margin"
    assert node.id in dag.nodes
    assert dag.current_node_id == node.id
    assert dag.nodes[node.id].parameters["formula_or_intent"] == "(revenue - cogs) / revenue"
    assert dag.nodes[node.id].reversibility == "full"
