from aereo_water.pipeline.compliance import build_compliance_table


def test_compliance_is_evidence_driven(tmp_path):
    table = build_compliance_table(
        output_root=tmp_path,
        expected_completed_hpo_trials=12,
        expected_total_inference_rows=2841,
        pytest_exit_code=1,
    )
    assert not table["complete"].any()
