import pytest

from gpustack.utils.runtime import (
    DEFAULT_GPU_MEMORY_UTILIZATION_ENV,
    get_default_gpu_memory_utilization_override,
)


@pytest.mark.parametrize(
    "raw,expected",
    [(None, None), ("", None), ("abc", None), ("0", None), ("1.5", None), ("0.5", 0.5)],
)
def test_override(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(DEFAULT_GPU_MEMORY_UTILIZATION_ENV, raising=False)
    else:
        monkeypatch.setenv(DEFAULT_GPU_MEMORY_UTILIZATION_ENV, raw)
    assert get_default_gpu_memory_utilization_override() == expected
