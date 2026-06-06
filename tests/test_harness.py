from kagura_code_review.review.angles import ANGLE_PROMPTS, CORRECTNESS_ANGLES


def test_angle_catalog_has_seven_angles():
    assert set(ANGLE_PROMPTS) == {
        "correctness-linescan", "removed-behavior", "cross-file",
        "reuse", "simplification", "efficiency", "altitude",
    }
    assert all(isinstance(v, str) and len(v) > 20 for v in ANGLE_PROMPTS.values())


def test_correctness_angles_subset():
    assert CORRECTNESS_ANGLES == {"correctness-linescan", "removed-behavior", "cross-file"}
    assert CORRECTNESS_ANGLES <= set(ANGLE_PROMPTS)
