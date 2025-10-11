"""Buffer creation and pool management"""

from typing import Optional, Tuple

import numpy as np
from gpu_device import wgpu
from gpu_types import (
    BufferInfo,
    BufferPool,
    Device,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUBuffer3D,
    GPUBufferAny,
    StagingPool,
    WGPUBuffer,
)

# ============================================================================
# BASIC BUFFER OPERATIONS
# ============================================================================


def _create_gpu_buffer_internal(
    device: Device, shape: Tuple[int, ...], data: Optional[np.ndarray] = None
) -> WGPUBuffer:
    """Internal: Create raw GPU buffer of any shape"""
    size = int(np.prod(shape))
    buffer_size = size * 4  # 4 bytes per float32

    if data is not None:
        data_np = np.ascontiguousarray(data, dtype=np.float32).flatten()
        buffer = device.wgpu_device.create_buffer_with_data(
            data=data_np,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
    else:
        buffer = device.wgpu_device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

    return buffer


def create_gpu_buffer_1d(
    device: Device, size: int, data: Optional[np.ndarray] = None
) -> GPUBuffer1D:
    """Create 1D GPU buffer for vectors"""
    shape = (size,)
    buffer = _create_gpu_buffer_internal(device, shape, data)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size, device=device)


def create_gpu_buffer_2d(
    device: Device, rows: int, cols: int, data: Optional[np.ndarray] = None
) -> GPUBuffer2D:
    """Create 2D GPU buffer for matrices"""
    shape = (rows, cols)
    size = rows * cols
    buffer = _create_gpu_buffer_internal(device, shape, data)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size, device=device)


def create_gpu_buffer_3d(
    device: Device,
    dim1: int,
    dim2: int,
    dim3: int,
    data: Optional[np.ndarray] = None,
) -> GPUBuffer3D:
    """Create 3D GPU buffer for batched sequences"""
    shape = (dim1, dim2, dim3)
    size = dim1 * dim2 * dim3
    buffer = _create_gpu_buffer_internal(device, shape, data)
    return GPUBuffer3D(buffer=buffer, shape=shape, size=size, device=device)


def gpu_to_numpy(gpu_buffer: GPUBufferAny) -> np.ndarray:
    """Read GPU buffer back to CPU"""
    buffer_size = gpu_buffer.size * 4
    read_buffer = gpu_buffer.device.wgpu_device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )

    encoder = gpu_buffer.device.wgpu_device.create_command_encoder()
    encoder.copy_buffer_to_buffer(gpu_buffer.buffer, 0, read_buffer, 0, buffer_size)
    gpu_buffer.device.wgpu_device.queue.submit([encoder.finish()])

    read_buffer.map_sync(wgpu.MapMode.READ)
    data = np.frombuffer(read_buffer.read_mapped(), dtype=np.float32).copy()
    read_buffer.unmap()

    return data.reshape(gpu_buffer.shape)


def clear_buffer(gpu_buffer: GPUBufferAny) -> None:
    """Zero-initialize a GPU buffer"""
    device = gpu_buffer.device
    zero_data = np.zeros(gpu_buffer.size, dtype=np.float32)
    device.wgpu_device.queue.write_buffer(gpu_buffer.buffer, 0, zero_data)


# ============================================================================
# BUFFER POOL
# ============================================================================


def create_buffer_pool(device: Device, max_buffer_size_mb: int = 512) -> BufferPool:
    """Create a memory pool state for reusable GPU buffers"""
    max_size = max_buffer_size_mb * 1024 * 1024 // 4  # Convert to float32 count
    return BufferPool(device=device, max_size=max_size, pools={}, in_use=set())


def _pool_take_buffer_internal(
    pool_state: BufferPool, shape: Tuple[int, ...], size: int
) -> WGPUBuffer:
    """
    Internal: Take buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer is MOVED out of pool.
    The returned buffer is removed from pool.pools and added to pool.in_use.
    """
    if size in pool_state.pools and len(pool_state.pools[size]) > 0:
        buffer_info = pool_state.pools[size].pop()
        pool_state.in_use.add(id(buffer_info.buffer))
        return buffer_info.buffer

    # Create new buffer
    buffer_size = size * 4
    buffer = pool_state.device.wgpu_device.create_buffer(
        size=buffer_size,
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST,
    )
    pool_state.in_use.add(id(buffer))
    return buffer


def pool_take_buffer_1d(pool_state: BufferPool, size: int) -> GPUBuffer1D:
    """
    Take a 1D buffer from pool or create new.

    Buffer is MOVED out of pool (not shared). Caller owns the buffer.
    Must call pool_release_buffer() to return it.
    """
    shape = (size,)
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_take_buffer_2d(pool_state: BufferPool, rows: int, cols: int) -> GPUBuffer2D:
    """
    Take a 2D buffer from pool or create new.

    Buffer is MOVED out of pool (not shared). Caller owns the buffer.
    Must call pool_release_buffer() to return it.
    """
    shape = (rows, cols)
    size = rows * cols
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_take_buffer_3d(
    pool_state: BufferPool, dim1: int, dim2: int, dim3: int
) -> GPUBuffer3D:
    """
    Take a 3D buffer from pool or create new.

    Buffer is MOVED out of pool (not shared). Caller owns the buffer.
    Must call pool_release_buffer() to return it.
    """
    shape = (dim1, dim2, dim3)
    size = dim1 * dim2 * dim3
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
    return GPUBuffer3D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_release_buffer(pool_state: BufferPool, gpu_buffer: GPUBufferAny) -> None:
    """Return buffer to pool for reuse"""
    buffer_id = id(gpu_buffer.buffer)
    if buffer_id in pool_state.in_use:
        pool_state.in_use.remove(buffer_id)

    size = gpu_buffer.size
    if size not in pool_state.pools:
        pool_state.pools[size] = []

    buffer_info = BufferInfo(buffer=gpu_buffer.buffer)
    pool_state.pools[size].append(buffer_info)


# ============================================================================
# STAGING BUFFER POOL
# ============================================================================


def create_staging_pool(device: Device, initial_size_mb: int = 64) -> StagingPool:
    """Create staging buffer pool state for CPU-GPU transfers"""
    return StagingPool(
        device=device,
        staging_buffers={},
        max_size=initial_size_mb * 1024 * 1024,
    )


def _staging_get_buffer(pool_state: StagingPool, size_bytes: int) -> WGPUBuffer:
    """Get or create staging buffer for CPU-GPU or GPU-CPU transfers"""
    # Round up to next power of 2 for better pooling
    rounded_size = 2 ** (size_bytes - 1).bit_length()
    rounded_size = min(rounded_size, pool_state.max_size)

    if rounded_size not in pool_state.staging_buffers:
        pool_state.staging_buffers[rounded_size] = (
            pool_state.device.wgpu_device.create_buffer(
                size=rounded_size,
                usage=wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
                | wgpu.BufferUsage.MAP_READ
                | wgpu.BufferUsage.MAP_WRITE,
            )
        )

    return pool_state.staging_buffers[rounded_size]


def staging_upload_data(
    pool_state: StagingPool, gpu_buffer: GPUBufferAny, data_np: np.ndarray
) -> None:
    """Upload data to GPU using persistent staging buffer. Returns None (mutates GPU state)."""
    size_bytes = data_np.nbytes

    # For small transfers, use direct write
    if size_bytes <= 256 * 1024:  # 256KB threshold
        pool_state.device.wgpu_device.queue.write_buffer(
            gpu_buffer.buffer,
            0,
            np.ascontiguousarray(data_np, dtype=np.float32),
        )
        return

    # For large transfers, use staging buffer
    staging = _staging_get_buffer(pool_state, size_bytes)
    staging.map_sync(wgpu.MapMode.WRITE)
    staging.write_mapped(np.ascontiguousarray(data_np, dtype=np.float32))
    staging.unmap()

    encoder = pool_state.device.wgpu_device.create_command_encoder()
    encoder.copy_buffer_to_buffer(staging, 0, gpu_buffer.buffer, 0, size_bytes)
    pool_state.device.wgpu_device.queue.submit([encoder.finish()])


def staging_download_data(
    pool_state: StagingPool, gpu_buffer: GPUBufferAny
) -> np.ndarray:
    """Download data from GPU using persistent staging buffer"""
    size_bytes = gpu_buffer.size * 4
    staging = _staging_get_buffer(pool_state, size_bytes)

    encoder = pool_state.device.wgpu_device.create_command_encoder()
    encoder.copy_buffer_to_buffer(gpu_buffer.buffer, 0, staging, 0, size_bytes)
    pool_state.device.wgpu_device.queue.submit([encoder.finish()])

    staging.map_sync(wgpu.MapMode.READ)
    data = np.frombuffer(
        staging.read_mapped(), dtype=np.float32, count=gpu_buffer.size
    ).copy()
    staging.unmap()

    return data.reshape(gpu_buffer.shape)


def gpu_to_numpy_optimized(
    pool_state: Optional[StagingPool], gpu_buffer: GPUBufferAny
) -> np.ndarray:
    """Optimized GPU to numpy with staging buffer pool"""
    if pool_state is not None:
        return staging_download_data(pool_state, gpu_buffer)
    else:
        # Fallback to original implementation
        return gpu_to_numpy(gpu_buffer)
