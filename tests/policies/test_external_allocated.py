from types import SimpleNamespace

from gpustack.policies.resource_view import ResourceView
from gpustack.schemas.workers import GPUDeviceStatus, MemoryInfo


def test_external_allocated_gpu_has_no_allocatable_vram():
    gpus = [
        GPUDeviceStatus(index=i, type="oneapi", memory=MemoryInfo(total=100, used=0))
        for i in range(2)
    ]
    gpus[1].external_allocated = True
    worker = SimpleNamespace(
        id=1,
        name="w",
        status=SimpleNamespace(
            gpu_devices=gpus,
            memory=SimpleNamespace(is_unified_memory=False, total=1000),
        ),
        system_reserved=SimpleNamespace(ram=0, vram=0),
    )
    allocatable = ResourceView([]).allocatable(worker)
    assert allocatable.vram == {0: 100, 1: 0}
