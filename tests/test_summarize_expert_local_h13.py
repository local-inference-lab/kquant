from scripts.summarize_expert_local_h13 import _metric_record


def test_metric_record_wraps_variants_for_common_summary_helpers():
    variants = [{"name": "global_teacher"}]
    assert _metric_record({"variants": variants}) == {
        "modes": [{"variants": variants}]
    }
