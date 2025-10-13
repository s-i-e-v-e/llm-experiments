"""Buffer creation and pool management"""

from typing import Optional, Tuple, Union

import numpy as np
import wgpu

from .gpu_types import GPUBuffer1D, GPUBuffer2D, GPUBufferAny, GPUContext

# ============================================================================
# BASIC BUFFER OPERATIONS
# ============================================================================


def __gpu_buffer_create(
    ctx: GPUContext, shape: Tuple[int, ...], data: Optional[np.ndarray] = None
) -> wgpu.GPUBuffer:
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
        buffer = ctx.device.create_buffer_with_data(
            data=data_np,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )
    else:
        buffer = ctx.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

    return buffer


def gpu_buffer_1d_create(
    ctx: GPUContext, size: int, data: Optional[np.ndarray] = None
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
    buffer = __gpu_buffer_create(ctx, shape, data)
    return GPUBuffer1D(buffer=buffer, shape=shape, size=size)


def gpu_buffer_2d_create(
    ctx: GPUContext, rows: int, cols: int, data: Optional[np.ndarray] = None
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
    buffer = __gpu_buffer_create(ctx, shape, data)
    return GPUBuffer2D(buffer=buffer, shape=shape, size=size)


def gpu_buffer_zerofy(ctx: GPUContext, gpu_buffer: GPUBufferAny) -> None:
    """Zero-initialize a GPU buffer.

    Args:
        device: GPU device
        gpu_buffer: GPU buffer to clear
    """
    zero_data = np.zeros(gpu_buffer.size, dtype=np.float32)
    ctx.device.queue.write_buffer(gpu_buffer.buffer, 0, zero_data)


def __gpu_buffer_write(
    ctx: GPUContext, in_data: np.ndarray, buffer: Union[GPUBuffer1D, GPUBuffer2D]
) -> None:
    data_f32 = np.ascontiguousarray(in_data, dtype=np.float32)
    ctx.device.queue.write_buffer(buffer.buffer, 0, data_f32.tobytes())


def gpu_buffer_1d_write(
    ctx: GPUContext, in_data: np.ndarray, buffer: GPUBuffer1D
) -> None:
    __gpu_buffer_write(ctx, in_data, buffer)


def gpu_buffer_2d_write(
    ctx: GPUContext, in_data: np.ndarray, buffer: GPUBuffer2D
) -> None:
    __gpu_buffer_write(ctx, in_data, buffer)


def __gpu_buffer_read(
    ctx: GPUContext, buffer: Union[GPUBuffer1D, GPUBuffer2D], out_data: np.ndarray
) -> None:
    """Read GPU buffer to numpy array

    Creates temporary staging buffer, copies GPU data to it, maps and reads.
    The staging buffer is destroyed after reading.
    """
    size_bytes = out_data.nbytes

    # Create temporary staging buffer for this read
    staging = ctx.device.create_buffer(
        size=size_bytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    # Copy GPU buffer to staging buffer
    encoder = ctx.device.create_command_encoder()
    encoder.copy_buffer_to_buffer(buffer.buffer, 0, staging, 0, size_bytes)
    ctx.device.queue.submit([encoder.finish()])

    # Map staging buffer for reading (synchronous)
    staging.map_sync(wgpu.MapMode.READ)

    # Read data from mapped buffer into numpy array
    mapped_data = staging.read_mapped()
    np.copyto(
        out_data, np.frombuffer(mapped_data, dtype=np.float32).reshape(out_data.shape)
    )

    # Unmap and destroy staging buffer
    staging.unmap()
    staging.destroy()


def gpu_buffer_1d_read(
    ctx: GPUContext, buffer: GPUBuffer1D, out_data: np.ndarray
) -> None:
    """Read 1D GPU buffer to numpy array

    Args:
        device: GPU device
        buffer: Source GPU buffer
        out_data: Pre-allocated numpy array to fill (shape must match buffer.size)
    """
    __gpu_buffer_read(ctx, buffer, out_data)


def gpu_buffer_2d_read(
    ctx: GPUContext, buffer: GPUBuffer2D, out_data: np.ndarray
) -> None:
    """Read 2D GPU buffer to numpy array

    Args:
        device: GPU device
        buffer: Source GPU buffer
        out_data: Pre-allocated numpy array to fill (shape must match buffer rows x cols)
    """
    __gpu_buffer_read(ctx, buffer, out_data)
