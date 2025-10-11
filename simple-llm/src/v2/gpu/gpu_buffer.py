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
    """
    Internal: Create raw GPU buffer of any shape.

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
    """
    Create 1D GPU buffer for vectors (biases, layer norm params).

    Args:
        device: GPU device state
        size: Number of elements
        data: Optional numpy array to initialize buffer

    Returns:
        Typed 1D GPU buffer
    """
    shape = (size,)
    buffer = _create_gpu_buffer_internal(device, shape, data)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size, device=device)


def create_gpu_buffer_2d(
    device: Device, rows: int, cols: int, data: Optional[np.ndarray] = None
) -> GPUBuffer2D:
    """
    Create 2D GPU buffer for matrices (weight matrices, activations).

    Args:
        device: GPU device state
        rows: Number of rows
        cols: Number of columns
        data: Optional numpy array to initialize buffer

    Returns:
        Typed 2D GPU buffer
    """
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
    """
    Create 3D GPU buffer for batched sequences [batch, seq, dim].

    Args:
        device: GPU device state
        dim1: First dimension (typically batch size)
        dim2: Second dimension (typically sequence length)
        dim3: Third dimension (typically embedding dimension)
        data: Optional numpy array to initialize buffer

    Returns:
        Typed 3D GPU buffer
    """
    shape = (dim1, dim2, dim3)
    size = dim1 * dim2 * dim3
    buffer = _create_gpu_buffer_internal(device, shape, data)
    return GPUBuffer3D(buffer=buffer, shape=shape, size=size, device=device)


def gpu_to_numpy(gpu_buffer: GPUBufferAny) -> np.ndarray:
    """
    Read GPU buffer back to CPU as numpy array.

    Creates a temporary staging buffer for readback.
    For repeated downloads, use staging_download_data() instead.

    Args:
        gpu_buffer: GPU buffer to read

    Returns:
        Numpy array with buffer contents reshaped to original shape
    """
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
    """
    Zero-initialize a GPU buffer (mutation).

    This function MUTATES the GPU buffer contents by writing zeros.
    Returns None to signal mutation.

    Args:
        gpu_buffer: GPU buffer to clear
    """
    device = gpu_buffer.device
    zero_data = np.zeros(gpu_buffer.size, dtype=np.float32)
    device.wgpu_device.queue.write_buffer(gpu_buffer.buffer, 0, zero_data)


# ============================================================================
# BUFFER POOL
# ============================================================================


def create_buffer_pool(
    device: Device, max_buffer_size_mb: int = 512, max_total_memory_mb: int = 2048
) -> BufferPool:
    """
    Create a memory pool state for reusable GPU buffers.

    Buffers are pooled by size for efficient reuse without reallocation.

    Args:
        device: GPU device state
        max_buffer_size_mb: Maximum size of individual pooled buffers in MB
        max_total_memory_mb: Maximum total pool memory in MB (0 = unlimited)

    Returns:
        Buffer pool state with memory limits enforced
    """
    max_size = max_buffer_size_mb * 1024 * 1024 // 4  # Convert to float32 count
    max_total_bytes = (
        max_total_memory_mb * 1024 * 1024 if max_total_memory_mb > 0 else 0
    )

    return BufferPool(
        device=device,
        max_size=max_size,
        pools={},
        in_use=set(),
        total_memory_bytes=0,
        max_total_memory_bytes=max_total_bytes,
    )


def _pool_take_buffer_internal(
    pool_state: BufferPool, shape: Tuple[int, ...], size: int
) -> WGPUBuffer:
    """
    Internal: Take buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer is MOVED out of pool.
    The returned buffer is removed from pool.pools and added to pool.in_use.
    Caller owns the buffer and must return it with pool_release_buffer().

    Raises:
        MemoryError: If creating new buffer would exceed max_total_memory_bytes

    Args:
        pool_state: Buffer pool state (MUTATED)
        shape: Buffer shape (for documentation only)
        size: Buffer size in float32 elements

    Returns:
        Raw WGPU buffer
    """
    # Try to reuse from pool first
    if size in pool_state.pools and len(pool_state.pools[size]) > 0:
        buffer_info = pool_state.pools[size].pop()
        pool_state.in_use.add(id(buffer_info.buffer))
        return buffer_info.buffer

    # Need to create new buffer - check memory limit
    buffer_size_bytes = size * 4

    if pool_state.max_total_memory_bytes > 0:
        if (
            pool_state.total_memory_bytes + buffer_size_bytes
            > pool_state.max_total_memory_bytes
        ):
            # Try to free unused buffers
            freed = _pool_evict_unused_buffers_internal(pool_state, buffer_size_bytes)
            if freed < buffer_size_bytes:
                raise MemoryError(
                    f"Cannot allocate {buffer_size_bytes} bytes. "
                    f"Current: {pool_state.total_memory_bytes}, "
                    f"Max: {pool_state.max_total_memory_bytes}, "
                    f"Available: {pool_state.max_total_memory_bytes - pool_state.total_memory_bytes}"
                )

    # Create new buffer
    buffer = pool_state.device.wgpu_device.create_buffer(
        size=buffer_size_bytes,
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST,
    )

    pool_state.in_use.add(id(buffer))
    pool_state.total_memory_bytes += buffer_size_bytes

    return buffer


def _pool_evict_unused_buffers_internal(
    pool_state: BufferPool, needed_bytes: int
) -> int:
    """
    Internal: Evict unused buffers from pool to free memory.

    Eviction strategy: Remove smallest buffers first (they're least likely to be reused).

    Args:
        pool_state: Buffer pool state (MUTATED)
        needed_bytes: Minimum bytes to free

    Returns:
        Total bytes freed
    """
    freed_bytes = 0

    # Sort sizes smallest first
    sorted_sizes = sorted(pool_state.pools.keys())

    for size in sorted_sizes:
        if freed_bytes >= needed_bytes:
            break

        buffer_list = pool_state.pools[size]
        while buffer_list and freed_bytes < needed_bytes:
            buffer_info = buffer_list.pop()
            # Buffer is automatically freed by WGPU when no references exist
            freed_bytes += size * 4
            pool_state.total_memory_bytes -= size * 4

        # Remove empty size categories
        if not buffer_list:
            del pool_state.pools[size]

    return freed_bytes


def pool_release_buffer(pool_state: BufferPool, gpu_buffer: GPUBufferAny) -> None:
    """
    Return buffer to pool for reuse (mutation).

    This function MUTATES pool_state by adding buffer back to the pool.
    After calling this, the buffer should not be used by the caller.

    Args:
        pool_state: Buffer pool state (MUTATED)
        gpu_buffer: GPU buffer to return to pool
    """
    buffer_id = id(gpu_buffer.buffer)
    if buffer_id in pool_state.in_use:
        pool_state.in_use.remove(buffer_id)

    size = gpu_buffer.size
    if size not in pool_state.pools:
        pool_state.pools[size] = []

    buffer_info = BufferInfo(buffer=gpu_buffer.buffer)
    pool_state.pools[size].append(buffer_info)


def get_pool_memory_stats(pool_state: BufferPool) -> Dict[str, int]:
    """
    Get memory statistics for buffer pool.

    Returns:
        Dictionary with memory statistics in bytes:
        - total_allocated: Total memory currently allocated
        - in_use: Memory for buffers currently in use
        - pooled: Memory for buffers available in pool
        - num_pooled_buffers: Number of buffers in pool
        - num_in_use_buffers: Number of buffers in use
    """
    pooled_bytes = sum(
        len(buffers) * size * 4 for size, buffers in pool_state.pools.items()
    )

    in_use_bytes = pool_state.total_memory_bytes - pooled_bytes

    num_pooled = sum(len(buffers) for buffers in pool_state.pools.values())
    num_in_use = len(pool_state.in_use)

    return {
        "total_allocated": pool_state.total_memory_bytes,
        "in_use": in_use_bytes,
        "pooled": pooled_bytes,
        "num_pooled_buffers": num_pooled,
        "num_in_use_buffers": num_in_use,
        "max_total": pool_state.max_total_memory_bytes,
    }


def pool_take_buffer_1d(pool_state: BufferPool, size: int) -> GPUBuffer1D:
    """
    Take a 1D buffer from pool or create new.

    Buffer is MOVED out of pool (not shared). Caller owns the buffer.
    Must call pool_release_buffer() to return it when done.

    Args:
        pool_state: Buffer pool state (MUTATED)
        size: Number of elements

    Returns:
        1D GPU buffer owned by caller
    """
    shape = (size,)
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_take_buffer_2d(pool_state: BufferPool, rows: int, cols: int) -> GPUBuffer2D:
    """
    Take a 2D buffer from pool or create new.

    Buffer is MOVED out of pool (not shared). Caller owns the buffer.
    Must call pool_release_buffer() to return it when done.

    Args:
        pool_state: Buffer pool state (MUTATED)
        rows: Number of rows
        cols: Number of columns

    Returns:
        2D GPU buffer owned by caller
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
    Must call pool_release_buffer() to return it when done.

    Args:
        pool_state: Buffer pool state (MUTATED)
        dim1: First dimension
        dim2: Second dimension
        dim3: Third dimension

    Returns:
        3D GPU buffer owned by caller
    """
    shape = (dim1, dim2, dim3)
    size = dim1 * dim2 * dim3
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
    return GPUBuffer3D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


# ============================================================================
# STAGING BUFFER POOL
# ============================================================================


def create_staging_pool(
    device: Device, initial_size_mb: int = 64, max_entries: int = 8
) -> StagingPool:
    """
    Create staging buffer pool state for CPU-GPU transfers.

    Staging buffers are persistent and reused across transfers for efficiency.

    Args:
        device: GPU device state
        initial_size_mb: Initial maximum staging buffer size in MB
        max_entries: Maximum number of different-sized buffers to cache

    Returns:
        Staging pool state
    """
    return StagingPool(
        device=device,
        staging_buffers={},
        max_size=initial_size_mb * 1024 * 1024,
        max_entries=max_entries,
    )


def _get_staging_buffer_internal(
    pool_state: StagingPool, size_bytes: int
) -> WGPUBuffer:
    """
    Internal: Get or create staging buffer for CPU-GPU or GPU-CPU transfers.

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


def staging_upload_data(
    pool_state: StagingPool, gpu_buffer: GPUBufferAny, data_np: np.ndarray
) -> None:
    """
    Upload data to GPU using persistent staging buffer (mutation).

    This function MUTATES gpu_buffer contents by writing data to it.
    Returns None to signal mutation.

    For small transfers (<256KB), uses direct write.
    For large transfers, uses staging buffer for better performance.

    Args:
        pool_state: Staging pool state (may be mutated if new staging buffer created)
        gpu_buffer: Target GPU buffer (MUTATED)
        data_np: Source numpy array
    """
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
    staging = _get_staging_buffer_internal(pool_state, size_bytes)
    staging.map_sync(wgpu.MapMode.WRITE)
    staging.write_mapped(np.ascontiguousarray(data_np, dtype=np.float32))
    staging.unmap()

    encoder = pool_state.device.wgpu_device.create_command_encoder()
    encoder.copy_buffer_to_buffer(staging, 0, gpu_buffer.buffer, 0, size_bytes)
    pool_state.device.wgpu_device.queue.submit([encoder.finish()])


def staging_download_data(
    pool_state: StagingPool, gpu_buffer: GPUBufferAny
) -> np.ndarray:
    """
    Download data from GPU using persistent staging buffer.

    More efficient than gpu_to_numpy() for repeated downloads.

    Args:
        pool_state: Staging pool state (may be mutated if new staging buffer created)
        gpu_buffer: Source GPU buffer

    Returns:
        Numpy array with buffer contents
    """
    size_bytes = gpu_buffer.size * 4
    staging = _get_staging_buffer_internal(pool_state, size_bytes)

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
    """
    Optimized GPU to numpy with staging buffer pool.

    Uses staging pool if provided, otherwise falls back to basic download.

    Args:
        pool_state: Optional staging pool state (for efficiency)
        gpu_buffer: GPU buffer to read

    Returns:
        Numpy array with buffer contents
    """
    if pool_state is not None:
        return staging_download_data(pool_state, gpu_buffer)
    else:
        # Fallback to original implementation
        return gpu_to_numpy(gpu_buffer)
