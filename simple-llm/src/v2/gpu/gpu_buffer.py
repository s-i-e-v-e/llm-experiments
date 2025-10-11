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


def create_gpu_buffer_internal(
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
    buffer = create_gpu_buffer_internal(device, shape, data)
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
    buffer = create_gpu_buffer_internal(device, shape, data)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size, device=device)


def create_gpu_buffer_3d(
    device: Device,
    dim1: int,
    dim2: int,
    dim3: int,
    data: Optional[np.ndarray] = None,
) -> GPUBuffer3D:
    """Create 3D GPU buffer for batched sequences (batch, seq, dim).

    Args:
        device: GPU device state
        dim1: First dimension (typically batch size)
        dim2: Second dimension (typically sequence length)
        dim3: Third dimension (typically embedding dimension)
        data: Optional numpy array to initialize buffer

    Returns:
        Typed 3D GPU buffer

    Raises:
        ValueError: If dimensions <= 0 or data shape doesn't match
    """
    if dim1 <= 0 or dim2 <= 0 or dim3 <= 0:
        raise ValueError(
            f"Buffer dimensions must be positive, got ({dim1}, {dim2}, {dim3})"
        )

    if data is not None:
        if data.shape != (dim1, dim2, dim3):
            raise ValueError(
                f"Data shape {data.shape} doesn't match buffer shape ({dim1}, {dim2}, {dim3})"
            )

    shape = (dim1, dim2, dim3)
    size = dim1 * dim2 * dim3
    buffer = create_gpu_buffer_internal(device, shape, data)
    return GPUBuffer3D(buffer=buffer, shape=shape, size=size, device=device)


def gpu_to_numpy(gpu_buffer: GPUBufferAny) -> np.ndarray:
    """Read GPU buffer back to CPU as numpy array.

    Creates a temporary staging buffer for readback.
    For repeated downloads, use staging_download_data instead.

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
    """Zero-initialize a GPU buffer (mutation).

    This function MUTATES the GPU buffer contents by writing zeros.
    Returns None to signal mutation.

    Args:
        gpu_buffer: GPU buffer to clear (MUTATED)
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
    """Create a memory pool state for reusable GPU buffers.

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
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
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
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_take_buffer_3d(
    pool_state: BufferPool, dim1: int, dim2: int, dim3: int
) -> GPUBuffer3D:
    """Take 3D buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer ownership transfers to caller.
    Caller must return buffer with pool_release_buffer when done.

    Args:
        pool_state: Buffer pool state (MUTATED)
        dim1: First dimension
        dim2: Second dimension
        dim3: Third dimension

    Returns:
        3D GPU buffer owned by caller

    Raises:
        ValueError: If dimensions <= 0
        MemoryError: If pool memory limit exceeded
    """
    if dim1 <= 0 or dim2 <= 0 or dim3 <= 0:
        raise ValueError(
            f"Buffer dimensions must be positive, got ({dim1}, {dim2}, {dim3})"
        )

    shape = (dim1, dim2, dim3)
    size = dim1 * dim2 * dim3
    buffer = _pool_take_buffer_internal(pool_state, shape, size)
    return GPUBuffer3D(buffer=buffer, shape=shape, size=size, device=pool_state.device)


def pool_release_buffer(pool_state: BufferPool, gpu_buffer: GPUBufferAny) -> None:
    """Return buffer to pool for reuse (mutation).

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


def create_staging_pool(
    device: Device, initial_size_mb: int = 64, max_entries: int = 8
) -> StagingPool:
    """Create staging buffer pool state for CPU-GPU transfers.

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


def staging_upload_data(
    pool_state: StagingPool, gpu_buffer: GPUBufferAny, data_np: np.ndarray
) -> None:
    """Upload data to GPU using persistent staging buffer (mutation).

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
    if size_bytes < 256 * 1024:  # 256KB threshold
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
    """Download data from GPU using persistent staging buffer.

    More efficient than gpu_to_numpy for repeated downloads.

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
    """Optimized GPU to numpy with staging buffer pool.

    Uses staging pool if provided, otherwise falls back to basic download.

    Args:
        pool_state: Optional staging pool state for efficiency
        gpu_buffer: GPU buffer to read

    Returns:
        Numpy array with buffer contents
    """
    if pool_state is not None:
        return staging_download_data(pool_state, gpu_buffer)
    else:
        # Fallback to original implementation
        return gpu_to_numpy(gpu_buffer)
