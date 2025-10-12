"""Buffer creation and pool management"""

from typing import Optional, Tuple, Union

import numpy as np
from gpu_device import wgpu
from gpu_types import (
    BufferInfo,
    BufferPool,
    Device,
    GPUBuffer1D,
    GPUBuffer2D,
    StagingPool,
    WGPUBuffer,
)

GPUBufferAny = Union[GPUBuffer1D | GPUBuffer2D]

# ============================================================================
# BASIC BUFFER OPERATIONS
# ============================================================================


def INTERNAL__create_gpu_buffer(
    device: Device, shape: Tuple[int, ...], data: Optional[np.ndarray] = None
) -> WGPUBuffer:
    """Internal: Create raw GPU buffer of any shape.

    Args:
        device: GPU device state
        shape: Buffer shape (any dimensionality)
        data: Optional numpy array to initialize buffer contents

    Returns:
        Raw WGPU buffer object
    """
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
    """Create 1D GPU buffer for vectors (biases, layer norm params).

    Args:
        device: GPU device state
        size: Number of elements
        data: Optional numpy array to initialize buffer

    Returns:
        Typed 1D GPU buffer

    Raises:
        ValueError: If size <= 0 or data shape doesn't match
    """
    if size <= 0:
        raise ValueError(f"Buffer size must be positive, got {size}")

    if data is not None:
        if data.shape != (size,):
            raise ValueError(
                f"Data shape {data.shape} doesn't match buffer size ({size},)"
            )

    shape = (size,)
    buffer = INTERNAL__create_gpu_buffer(device, shape, data)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size, device=device)


def create_gpu_buffer_2d(
    device: Device, rows: int, cols: int, data: Optional[np.ndarray] = None
) -> GPUBuffer2D:
    """Create 2D GPU buffer for matrices (weight matrices, activations).

    Args:
        device: GPU device state
        rows: Number of rows
        cols: Number of columns
        data: Optional numpy array to initialize buffer

    Returns:
        Typed 2D GPU buffer

    Raises:
        ValueError: If dimensions <= 0 or data shape doesn't match
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Buffer dimensions must be positive, got ({rows}, {cols})")

    if data is not None:
        if data.shape != (rows, cols):
            raise ValueError(
                f"Data shape {data.shape} doesn't match buffer shape ({rows}, {cols})"
            )

    shape = (rows, cols)
    size = rows * cols
    buffer = INTERNAL__create_gpu_buffer(device, shape, data)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size, device=device)


def clear_buffer(gpu_buffer: GPUBufferAny) -> None:
    """Zero-initialize a GPU buffer (mutation).

    Args:
        gpu_buffer: GPU buffer to clear (MUTATED)
    """
    device = gpu_buffer.device
    zero_data = np.zeros(gpu_buffer.size, dtype=np.float32)
    device.wgpu_device.queue.write_buffer(gpu_buffer.buffer, 0, zero_data)


# ============================================================================
# BUFFER POOL
# ============================================================================
def pool_create(
    device: Device,
    max_buffer_size_mb: Optional[int] = None,
    max_total_memory_mb: Optional[int] = None,
) -> BufferPool:
    """
    Create a memory pool state for reusable GPU buffers

    Buffers are pooled by size for efficient reuse without reallocation.

    Args:
        device: GPU device state
        max_buffer_size_mb: Maximum size of individual pooled buffers in MB.
                           If None, uses device.config.buffer_pool_max_buffer_mb
        max_total_memory_mb: Maximum total pool memory in MB (0 = unlimited).
                             If None, uses device.config.buffer_pool_max_mb

    Returns:
        Buffer pool state with memory limits enforced
    """
    # Use config defaults if not provided
    if max_buffer_size_mb is None:
        max_buffer_size_mb = device.config.buffer_pool_max_buffer_mb

    if max_total_memory_mb is None:
        max_total_memory_mb = device.config.buffer_pool_max_mb

    max_size = max_buffer_size_mb * 1024 * 1024 // 4  # Convert to float32 count
    max_total_bytes = (
        max_total_memory_mb * 1024 * 1024 if max_total_memory_mb > 0 else 0
    )

    return BufferPool(
        device=device,
        maxsize=max_size,
        pools={},
        inuse=set(),
        totalmemorybytes=0,
        maxtotalmemorybytes=max_total_bytes,
    )


def INTERNAL__pool_take_buffer(
    pool_state: BufferPool, shape: Tuple[int, ...], size: int
) -> WGPUBuffer:
    """Internal: Take buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer is MOVED out of pool.
    The returned buffer is removed from pool.pools and added to pool.in_use.
    Caller owns the buffer and must return it with pool_release_buffer.

    Args:
        pool_state: Buffer pool state (MUTATED)
        shape: Buffer shape (for documentation only)
        size: Buffer size in float32 elements

    Returns:
        Raw WGPU buffer owned by caller

    Raises:
        MemoryError: If creating new buffer would exceed max_total_memory_bytes
    """
    # Check if we have a buffer of this size in the pool
    if size in pool_state.pools and pool_state.pools[size]:
        # Take buffer from pool
        buffer_info = pool_state.pools[size].pop()
        buffer = buffer_info.buffer
        pool_state.in_use.add(id(buffer))
        return buffer

    # Need to create new buffer
    buffer_bytes = size * 4

    # Check memory limits
    if pool_state.max_total_memory_bytes > 0:
        if (
            pool_state.total_memory_bytes + buffer_bytes
            > pool_state.max_total_memory_bytes
        ):
            raise MemoryError(
                f"Buffer pool memory limit exceeded: "
                f"{pool_state.total_memory_bytes + buffer_bytes} > {pool_state.max_total_memory_bytes}"
            )

    # Create new buffer
    if size > pool_state.max_size:
        # Too large to pool - create non-pooled buffer
        buffer = pool_state.device.wgpu_device.create_buffer(
            size=buffer_bytes,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
    else:
        # Create pooled buffer
        buffer = pool_state.device.wgpu_device.create_buffer(
            size=buffer_bytes,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
        pool_state.in_use.add(id(buffer))
        pool_state.total_memory_bytes += buffer_bytes

    return buffer


def pool_take_buffer_1d(pool_state: BufferPool, size: int) -> GPUBuffer1D:
    """Take 1D buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer ownership transfers to caller.
    Caller must return buffer with pool_release_buffer when done.

    Args:
        pool_state: Buffer pool state (MUTATED)
        size: Number of elements

    Returns:
        1D GPU buffer owned by caller

    Raises:
        ValueError: If size <= 0
        MemoryError: If pool memory limit exceeded
    """
    if size <= 0:
        raise ValueError(f"Buffer size must be positive, got {size}")

    shape = (size,)
    buffer = INTERNAL__pool_take_buffer(pool_state, shape, size)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_take_buffer_2d(pool_state: BufferPool, rows: int, cols: int) -> GPUBuffer2D:
    """Take 2D buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer ownership transfers to caller.
    Caller must return buffer with pool_release_buffer when done.

    Args:
        pool_state: Buffer pool state (MUTATED)
        rows: Number of rows
        cols: Number of columns

    Returns:
        2D GPU buffer owned by caller

    Raises:
        ValueError: If dimensions <= 0
        MemoryError: If pool memory limit exceeded
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Buffer dimensions must be positive, got ({rows}, {cols})")

    shape = (rows, cols)
    size = rows * cols
    buffer = INTERNAL__pool_take_buffer(pool_state, shape, size)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_release_buffer(pool_state: BufferPool, gpu_buffer: GPUBufferAny) -> None:
    """Return buffer to pool for reuse

    SEMANTICS: This is a RELEASE operation - buffer ownership returns to pool.
    Caller must not use the buffer after calling this function.

    Args:
        pool_state: Buffer pool state (MUTATED)
        gpu_buffer: GPU buffer to return to pool
    """
    buffer_id = id(gpu_buffer.buffer)
    size = gpu_buffer.size

    # Only pool if size is within limits
    if size <= pool_state.max_size:
        if buffer_id in pool_state.in_use:
            pool_state.in_use.remove(buffer_id)

        # Add to appropriate pool
        if size not in pool_state.pools:
            pool_state.pools[size] = []

        pool_state.pools[size].append(BufferInfo(buffer=gpu_buffer.buffer))


def pool_clear(pool_state: BufferPool) -> None:
    """Clear all pooled buffers and reset state (mutation).

    This releases all buffers back to WGPU for cleanup.
    In-use buffers are not affected.

    Args:
        pool_state: Buffer pool state (MUTATED)
    """
    pool_state.pools.clear()
    pool_state.total_memory_bytes = 0


# ============================================================================
# STAGING BUFFER POOL
# ============================================================================


def staging_pool_create(
    device: Device,
    initial_size_mb: Optional[int] = None,
    max_entries: Optional[int] = None,
) -> StagingPool:
    """
    Create staging buffer pool state for CPU-GPU transfers

    Staging buffers are persistent and reused across transfers for efficiency.

    Args:
        device: GPU device state
        initial_size_mb: Initial maximum staging buffer size in MB.
                        If None, uses 64 MB default
        max_entries: Maximum number of different-sized buffers to cache.
                    If None, uses device.config.staging_buffer_max_entries

    Returns:
        Staging pool state
    """
    if initial_size_mb is None:
        initial_size_mb = 64

    if max_entries is None:
        max_entries = device.config.staging_buffer_max_entries

    return StagingPool(
        device=device,
        stagingbuffers={},
        maxsize=initial_size_mb * 1024 * 1024,
        maxentries=max_entries,
    )


def INTERNAL__get_staging_buffer(
    pool_state: StagingPool, size_bytes: int
) -> WGPUBuffer:
    """Internal: Get or create staging buffer for CPU-GPU or GPU-CPU transfers.

    Staging buffers are reused and persist in the pool (not taken out).
    Implements LRU eviction when max_entries is exceeded.

    Args:
        pool_state: Staging pool state (MUTATED if new buffer created)
        size_bytes: Required size in bytes

    Returns:
        Staging buffer (still owned by pool)
    """
    # Round up to next power of 2 for better pooling
    rounded_size = 2 ** (size_bytes - 1).bit_length()
    rounded_size = min(rounded_size, pool_state.max_size)

    if rounded_size not in pool_state.staging_buffers:
        # Check if we need to evict
        if len(pool_state.staging_buffers) >= pool_state.max_entries:
            # Evict smallest buffer (least likely to be reused)
            smallest_size = min(pool_state.staging_buffers.keys())
            del pool_state.staging_buffers[smallest_size]

        # Create new buffer
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


def staging_pool_upload(
    poolstate: StagingPool, gpubuffer: GPUBufferAny, data_np: np.ndarray
) -> None:
    """
    Upload data to GPU using persistent staging buffer (mutation)

    This function MUTATES gpubuffer contents by writing data to it.
    Returns None to signal mutation.

    For small transfers (<= threshold), uses direct write.
    For large transfers, uses staging buffer for better performance.

    Args:
        poolstate: Staging pool state (may be mutated if new staging buffer created)
        gpubuffer: Target GPU buffer (MUTATED)
        data_np: Source numpy array
    """
    size_bytes = data_np.nbytes

    # Use config threshold for deciding staging vs direct transfer
    threshold_bytes = poolstate.device.config.staging_buffer_threshold_kb * 1024

    # For small transfers, use direct write
    if size_bytes <= threshold_bytes:
        poolstate.device.wgpu_device.queue.write_buffer(
            gpubuffer.buffer,
            0,
            np.ascontiguousarray(data_np, dtype=np.float32),
        )
        return

    # For large transfers, use staging buffer
    staging = INTERNAL__get_staging_buffer(poolstate, size_bytes)
    staging.map_sync(wgpu.MapMode.WRITE)
    staging.write_mapped(np.ascontiguousarray(data_np, dtype=np.float32))
    staging.unmap()

    encoder = poolstate.device.wgpu_device.create_command_encoder()
    encoder.copy_buffer_to_buffer(staging, 0, gpubuffer.buffer, 0, size_bytes)
    poolstate.device.wgpu_device.queue.submit([encoder.finish()])


def staging_pool_download(
    pool_state: StagingPool, gpu_buffer: GPUBufferAny
) -> np.ndarray:
    """Download data from GPU using persistent staging buffer.

    More efficient than using numpy for repeated downloads.

    Args:
        pool_state: Staging pool state (may be mutated if new staging buffer created)
        gpu_buffer: Source GPU buffer

    Returns:
        Numpy array with buffer contents
    """
    size_bytes = gpu_buffer.size * 4
    staging = INTERNAL__get_staging_buffer(pool_state, size_bytes)

    encoder = pool_state.device.wgpu_device.create_command_encoder()
    encoder.copy_buffer_to_buffer(gpu_buffer.buffer, 0, staging, 0, size_bytes)
    pool_state.device.wgpu_device.queue.submit([encoder.finish()])

    staging.map_sync(wgpu.MapMode.READ)
    data = np.frombuffer(
        staging.read_mapped(), dtype=np.float32, count=gpu_buffer.size
    ).copy()
    staging.unmap()

    return data.reshape(gpu_buffer.shape)
