"""Tests for local quantum-circuit execution."""

import pytest

from tellekynezix_qml import bell_state_probabilities


def test_bell_state_has_expected_probabilities() -> None:
    probabilities = tuple(float(value) for value in bell_state_probabilities())

    assert len(probabilities) == 4
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-8)
    assert probabilities == pytest.approx((0.5, 0.0, 0.0, 0.5), abs=1e-8)

