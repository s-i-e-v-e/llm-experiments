"""Buffer creation and pool management"""

from typing import Optional, Tuple, Union

import numpy as np

from .gpu_device import wgpu
from .gpu_types import (
    BatchState,
    BufferInfo,
    BufferPool,
    GPUBuffer,
    GPUBuffer1D,
    GPUBuffer2D,
    GPUConfig,
    GPUDevice,
    StagingPool,
)

GPUBufferAny = Union[GPUBuffer1D | GPUBuffer2D]

# ============================================================================
# BASIC BUFFER OPERATIONS
# ============================================================================


def INTERNAL__create_gpu_buffer(
    device: GPUDevice, shape: Tuple[int, ...], data: Optional[np.ndarray] = None
) -> GPUBuffer:
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
        buffer = device.create_buffer_with_data(
            data=data_np,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
    else:
        buffer = device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

    return buffer


def create_gpu_buffer_1d(
    device: GPUDevice, size: int, data: Optional[np.ndarray] = None
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
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size)


def create_gpu_buffer_2d(
    device: GPUDevice, rows: int, cols: int, data: Optional[np.ndarray] = None
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
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size)


def clear_buffer(device: GPUDevice, gpu_buffer: GPUBufferAny) -> None:
    """Zero-initialize a GPU buffer.

    Args:
        gpu_buffer: GPU buffer to clear
    """
    device = device
    zero_data = np.zeros(gpu_buffer.size, dtype=np.float32)
    device.queue.write_buffer(gpu_buffer.buffer, 0, zero_data)


# ============================================================================
# BUFFER POOL
# ============================================================================
def pool_create(
    config: GPUConfig,
    max_buffer_size_mb: Optional[int] = None,
    max_total_memory_mb: Optional[int] = None,
) -> BufferPool:
    """
    Create a memory pool state for reusable GPU buffers

    Buffers are pooled by size for efficient reuse without reallocation.

    Args:
        device: GPU device state
        max_buffer_size_mb: Maximum size of individual pooled buffers in MB.
                           If None, uses config.buffer_pool_max_buffer_mb
        max_total_memory_mb: Maximum total pool memory in MB (0 = unlimited).
                             If None, uses config.buffer_pool_max_mb

    Returns:
        Buffer pool state with memory limits enforced
    """
    # Use config defaults if not provided
    if max_buffer_size_mb is None:
        max_buffer_size_mb = config.buffer_pool_max_buffer_mb

    if max_total_memory_mb is None:
        max_total_memory_mb = config.buffer_pool_max_mb

    max_size = max_buffer_size_mb * 1024 * 1024 // 4  # Convert to float32 count
    max_total_bytes = (
        max_total_memory_mb * 1024 * 1024 if max_total_memory_mb > 0 else 0
    )

    return BufferPool(
        max_size=max_size,
        pools={},
        in_use=set(),
        total_memory_bytes=0,
        max_total_memory_bytes=max_total_bytes,
    )


def INTERNAL__pool_take_buffer(
    device: GPUDevice, pool_state: BufferPool, shape: Tuple[int, ...], size: int
) -> GPUBuffer:
    """Internal: Take buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer is MOVED out of pool.
    The returned buffer is removed from pool.pools and added to pool.in_use.
    Caller owns the buffer and must return it with pool_release_buffer.

    Args:
        pool_state: Buffer pool state
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
        buffer = device.create_buffer(
            size=buffer_bytes,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
    else:
        # Create pooled buffer
        buffer = device.create_buffer(
            size=buffer_bytes,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
        pool_state.in_use.add(id(buffer))
        pool_state.total_memory_bytes += buffer_bytes

    return buffer


def pool_take_buffer_1d(
    device: GPUDevice, pool_state: BufferPool, size: int
) -> GPUBuffer1D:
    """Take 1D buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer ownership transfers to caller.
    Caller must return buffer with pool_release_buffer when done.

    Args:
        pool_state: Buffer pool state
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
    buffer = INTERNAL__pool_take_buffer(device, pool_state, shape, size)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size)


def pool_take_buffer_2d(
    device: GPUDevice, pool_state: BufferPool, rows: int, cols: int
) -> GPUBuffer2D:
    """Take 2D buffer from pool or create new.

    SEMANTICS: This is a TAKE operation - buffer ownership transfers to caller.
    Caller must return buffer with pool_release_buffer when done.

    Args:
        pool_state: Buffer pool state
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
    buffer = INTERNAL__pool_take_buffer(device, pool_state, shape, size)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size)


def pool_release_buffer(pool_state: BufferPool, gpu_buffer: GPUBufferAny) -> None:
    """Return buffer to pool for reuse

    SEMANTICS: This is a RELEASE operation - buffer ownership returns to pool.
    Caller must not use the buffer after calling this function.

    Args:
        pool_state: Buffer pool state
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
    """Clear all pooled buffers and reset state.

    This releases all buffers back to WGPU for cleanup.
    In-use buffers are not affected.

    Args:
        pool_state: Buffer pool state
    """
    pool_state.pools.clear()
    pool_state.total_memory_bytes = 0


# ============================================================================
# STAGING BUFFER POOL
# ============================================================================


def staging_pool_create(
    config: GPUConfig,
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
                    If None, uses config.staging_buffer_max_entries

    Returns:
        Staging pool state
    """
    if initial_size_mb is None:
        initial_size_mb = 64

    if max_entries is None:
        max_entries = config.staging_buffer_max_entries

    return StagingPool(
        staging_buffers={},
        max_size=initial_size_mb * 1024 * 1024,
        max_entries=max_entries,
    )


def INTERNAL__get_staging_buffer(
    device: GPUDevice, pool_state: StagingPool, size_bytes: int
) -> GPUBuffer:
    """Internal: Get or create staging buffer for CPU-GPU or GPU-CPU transfers.

    Staging buffers are reused and persist in the pool (not taken out).
    Implements LRU eviction when max_entries is exceeded.

    Args:
        pool_state: Staging pool state
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
        pool_state.staging_buffers[rounded_size] = device.create_buffer(
            size=rounded_size,
            usage=wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.MAP_READ
            | wgpu.BufferUsage.MAP_WRITE,
        )

    return pool_state.staging_buffers[rounded_size]


def staging_pool_upload(
    device: GPUDevice,
    config: GPUConfig,
    poolstate: StagingPool,
    gpubuffer: GPUBufferAny,
    data_np: np.ndarray,
) -> None:
    """
    Upload data to GPU using persistent staging buffer

    For small transfers (<= threshold), uses direct write.
    For large transfers, uses staging buffer for better performance.

    Args:
        poolstate: Staging pool state
        gpubuffer: Target GPU buffer
        data_np: Source numpy array
    """
    size_bytes = data_np.nbytes

    # Use config threshold for deciding staging vs direct transfer
    threshold_bytes = config.staging_buffer_threshold_kb * 1024

    # For small transfers, use direct write
    if size_bytes <= threshold_bytes:
        device.queue.write_buffer(
            gpubuffer.buffer,
            0,
            np.ascontiguousarray(data_np, dtype=np.float32),
        )
        return

    # For large transfers, use staging buffer
    staging = INTERNAL__get_staging_buffer(device, poolstate, size_bytes)
    staging.map_sync(wgpu.MapMode.WRITE)
    staging.write_mapped(np.ascontiguousarray(data_np, dtype=np.float32))
    staging.unmap()

    encoder = device.create_command_encoder()
    encoder.copy_buffer_to_buffer(staging, 0, gpubuffer.buffer, 0, size_bytes)
    device.queue.submit([encoder.finish()])


def staging_pool_download(
    device: GPUDevice, pool_state: StagingPool, gpu_buffer: GPUBufferAny
) -> np.ndarray:
    """Download data from GPU using persistent staging buffer.

    More efficient than using numpy for repeated downloads.

    Args:
        pool_state: Staging pool state
        gpu_buffer: Source GPU buffer

    Returns:
        Numpy array with buffer contents
    """
    size_bytes = gpu_buffer.size * 4
    staging = INTERNAL__get_staging_buffer(device, pool_state, size_bytes)

    encoder = device.create_command_encoder()
    encoder.copy_buffer_to_buffer(gpu_buffer.buffer, 0, staging, 0, size_bytes)
    device.queue.submit([encoder.finish()])

    staging.map_sync(wgpu.MapMode.READ)
    data = np.frombuffer(
        staging.read_mapped(), dtype=np.float32, count=gpu_buffer.size
    ).copy()
    staging.unmap()

    return data.reshape(gpu_buffer.shape)


# ============================================================================
# STAGING BUFFER POOL
# ============================================================================


def copy_buffer_region(
    device: GPUDevice,
    batch_state: BatchState,
    source: Union[GPUBuffer1D, GPUBuffer2D],
    dest: Union[GPUBuffer1D, GPUBuffer2D],
    source_offset: int = 0,
    dest_offset: int = 0,
    size: int = None,
) -> None:
    """Copy a region from source buffer to destination buffer.

    Args:
        device: GPU device
        batch_state: Batch state for command encoder
        source: Source buffer
        dest: Destination buffer
        source_offset: Offset in source buffer (in elements)
        dest_offset: Offset in destination buffer (in elements)
        size: Number of elements to copy (None = all remaining)

    Raises:
        ValueError: If offsets/size invalid
        RuntimeError: If batch not initialized
    """
    if batch_state.encoder is None:
        raise RuntimeError("Must call begin_batch before copy operations")

    source_size = (
        source.shape[0]
        if isinstance(source, GPUBuffer1D)
        else source.shape[0] * source.shape[1]
    )
    dest_size = (
        dest.shape[0]
        if isinstance(dest, GPUBuffer1D)
        else dest.shape[0] * dest.shape[1]
    )

    if size is None:
        size = source_size - source_offset

    if source_offset + size > source_size:
        raise ValueError("Copy would read past end of source buffer")
    if dest_offset + size > dest_size:
        raise ValueError("Copy would write past end of dest buffer")

    bytes_per_element = 4  # f32
    source_byte_offset = source_offset * bytes_per_element
    dest_byte_offset = dest_offset * bytes_per_element
    byte_size = size * bytes_per_element

    batch_state.encoder.copy_buffer_to_buffer(
        source.buffer, source_byte_offset, dest.buffer, dest_byte_offset, byte_size
    )
    batch_state.operation_count += 1


def concatenate_buffers_1d(
    device: GPUDevice,
    batch_state: BatchState,
    buffer_pool: BufferPool,
    sources: list,
) -> GPUBuffer1D:
    """Concatenate multiple 1D buffers into a single buffer.

    Args:
        device: GPU device
        batch_state: Batch state for command encoder
        buffer_pool: Buffer pool for allocating output
        sources: List of source buffers to concatenate

    Returns:
        New buffer containing concatenated data

    Raises:
        ValueError: If sources is empty
    """
    if not sources:
        raise ValueError("Cannot concatenate empty list of buffers")

    total_size = sum(buf.shape[0] for buf in sources)
    output = pool_take_buffer_1d(device, buffer_pool, total_size)

    offset = 0
    for source in sources:
        copy_buffer_region(
            device, batch_state, source, output, 0, offset, source.shape[0]
        )
        offset += source.shape[0]

    return output


def read_buffer_to_numpy(
    device: GPUDevice,
    buffer: Union[GPUBuffer1D, GPUBuffer2D],
    dtype=np.float32,
) -> np.ndarray:
    """Read GPU buffer contents to numpy array (blocking operation).

    This creates a staging buffer, copies GPU data to it, maps it for reading,
    and copies to a numpy array. This is a synchronous operation that will
    block until data is available.

    Args:
        device: GPU device
        buffer: Buffer to read from
        dtype: Numpy dtype (default: np.float32)

    Returns:
        Numpy array containing buffer data

    Raises:
        RuntimeError: If read operation fails
    """
    if isinstance(buffer, GPUBuffer1D):
        size = buffer.shape[0]
        shape = (size,)
    else:  # GPUBuffer2D
        rows, cols = buffer.shape
        size = rows * cols
        shape = (rows, cols)

    byte_size = size * 4  # f32 = 4 bytes

    staging_buffer = device.create_buffer(
        size=byte_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    encoder = device.create_command_encoder()
    encoder.copy_buffer_to_buffer(buffer.buffer, 0, staging_buffer, 0, byte_size)
    command_buffer = encoder.finish()
    device.queue.submit([command_buffer])

    device.queue.wait_until_idle()

    staging_buffer.map(wgpu.MapMode.READ)
    mapped_data = staging_buffer.read_mapped()
    array = np.frombuffer(mapped_data, dtype=dtype).copy()

    staging_buffer.unmap()
    staging_buffer.destroy()

    return array.reshape(shape)


def write_numpy_to_buffer(
    device: GPUDevice,
    data: np.ndarray,
    buffer: Union[GPUBuffer1D, GPUBuffer2D],
) -> None:
    """Write numpy array to GPU buffer (blocking operation).

    Args:
        device: GPU device
        data: Numpy array to write
        buffer: Destination buffer

    Raises:
        ValueError: If data shape doesn't match buffer shape
    """
    if isinstance(buffer, GPUBuffer1D):
        if data.shape != (buffer.shape[0],):
            raise ValueError(
                f"Data shape {data.shape} doesn't match buffer shape {buffer.shape}"
            )
    else:  # GPUBuffer2D
        if data.shape != buffer.shape:
            raise ValueError(
                f"Data shape {data.shape} doesn't match buffer shape {buffer.shape}"
            )

    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    device.queue.write_buffer(buffer.buffer, 0, data_f32.tobytes())


def wait_for_gpu_idle(device: GPUDevice) -> None:
    """Wait for all pending GPU operations to complete."""
    device.queue.wait_until_idle()


def poll_gpu(device: GPUDevice, timeout_ms: int = 1000) -> bool:
    """Poll GPU for completion with timeout.

    Returns:
        True if operations completed, False if timeout
    """
    import time

    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        if device.poll():
            return True
        time.sleep(0.001)
    return False
