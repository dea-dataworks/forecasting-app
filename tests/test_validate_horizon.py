import pytest
from src.compare import validate_horizon  # expected signature: (horizon: int, test_len: int) -> int

def test_clips_when_h_exceeds_test_len():
    assert validate_horizon(20, 10) == 10  # clips to test_len

def test_raises_for_non_positive_h():
    with pytest.raises(ValueError):
        validate_horizon(0, 10)
    with pytest.raises(ValueError):
        validate_horizon(-3, 10)

