"""
GPU acceleration using WGPU compute shaders.
"""

import dataclasses
import math

import numpy as np

try:
    import wgpu

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

_device = None


def get_device():
    """Get or create the default WGPU device"""
    global _device
    if not WGPU_AVAILABLE:
        return None

    if _device is None:
        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            _device = adapter.request_device_sync()
            print("✅ WGPU device initialized")
        except Exception as e:
            print(f"⚠️ WGPU initialization failed: {e}")
            _device = None
    return _device


# ============================================================================
# COMPUTE SHADERS
# ============================================================================

MATMUL_SHADER = """
struct Dimensions {
    M: u32,  // rows of A
    K: u32,  // cols of A, rows of B
    N: u32,  // cols of B
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

const TILE_SIZE: u32 = 8u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y * TILE_SIZE;
    let col = global_id.x * TILE_SIZE;

    // Each thread computes an 8x8 tile
    var sums: array<array<f32, 8>, 8>;
    for (var i = 0u; i < TILE_SIZE; i++) {
        for (var j = 0u; j < TILE_SIZE; j++) {
            sums[i][j] = 0.0;
        }
    }

    // Compute the tile
    for (var k = 0u; k < dims.K; k++) {
        for (var i = 0u; i < TILE_SIZE; i++) {
            let a_row = row + i;
            if (a_row < dims.M) {
                let a_elem = a[a_row * dims.K + k];
                for (var j = 0u; j < TILE_SIZE; j++) {
                    let b_col = col + j;
                    if (b_col < dims.N) {
                        sums[i][j] += a_elem * b[k * dims.N + b_col];
                    }
                }
            }
        }
    }

    // Write results
    for (var i = 0u; i < TILE_SIZE; i++) {
        for (var j = 0u; j < TILE_SIZE; j++) {
            let out_row = row + i;
            let out_col = col + j;
            if (out_row < dims.M && out_col < dims.N) {
                result[out_row * dims.N + out_col] = sums[i][j];
            }
        }
    }
}
"""


# ============================================================================
# MATRIX OPERATIONS
# ============================================================================
# ============================================================================
# GPU-NATIVE OPERATIONS (work directly with GPUBuffer)
# ============================================================================


def matmul_gpu(a, b):
    """
    Matrix multiplication on GPU buffers.

    Args:
        a: GPUBuffer of shape (M, K)
        b: GPUBuffer of shape (K, N)

    Returns:
        GPUBuffer of shape (M, N)
    """
    # Convert to numpy, use existing matmul, convert back
    a_np = gpu_to_numpy(a)
    b_np = gpu_to_numpy(b)
    result_np = matmul(a_np, b_np)  # Uses existing matmul function
    return create_gpu_buffer(result_np.shape, result_np, a.device)


def matmul_batched_gpu(a, b):
    """
    Batched matrix multiplication on GPU buffers.

    Args:
        a: GPUBuffer of shape (..., M, K)
        b: GPUBuffer of shape (..., K, N)

    Returns:
        GPUBuffer of shape (..., M, N)
    """
    device = a.device
    a_np = gpu_to_numpy(a)
    b_np = gpu_to_numpy(b)

    # Handle batch dimensions
    batch_dims = a.shape[:-2]
    M, K = a.shape[-2:]
    N = b.shape[-1]

    batch_size = int(np.prod(batch_dims)) if batch_dims else 1

    # Reshape to 3D
    a_3d = a_np.reshape(batch_size, M, K)
    b_3d = b_np.reshape(batch_size, K, N)

    # Process each batch
    results = []
    for i in range(batch_size):
        result = matmul(a_3d[i], b_3d[i])
        results.append(result)

    # Stack and reshape
    result_3d = np.stack(results, axis=0)
    output_shape = batch_dims + (M, N)
    result = result_3d.reshape(output_shape)

    return create_gpu_buffer(output_shape, result, device)


def layer_norm_gpu(x, gamma, beta, eps=1e-5):
    """
    Layer normalization on GPU buffers.

    Args:
        x: GPUBuffer of shape (..., dim)
        gamma: GPUBuffer of shape (dim,)
        beta: GPUBuffer of shape (dim,)

    Returns:
        GPUBuffer normalized output
    """
    device = x.device
    x_np = gpu_to_numpy(x)
    gamma_np = gpu_to_numpy(gamma)
    beta_np = gpu_to_numpy(beta)

    # Use existing layer_norm function
    result_np = layer_norm(x_np, gamma_np, beta_np, eps)

    return create_gpu_buffer(x.shape, result_np, device)


def softmax_gpu(x, axis=-1):
    """
    Softmax on GPU buffer.

    Args:
        x: GPUBuffer
        axis: axis to apply softmax

    Returns:
        GPUBuffer softmax output
    """
    device = x.device
    x_np = gpu_to_numpy(x)

    # Use existing softmax function
    result_np = softmax(x_np, axis)

    return create_gpu_buffer(x.shape, result_np, device)


def gelu_gpu(x):
    """
    GELU activation on GPU buffer.

    Args:
        x: GPUBuffer

    Returns:
        GPUBuffer GELU output
    """
    device = x.device
    x_np = gpu_to_numpy(x)

    # Use existing gelu function
    result_np = gelu(x_np)

    return create_gpu_buffer(x.shape, result_np, device)


def embedding_lookup_gpu(embedding_table, indices_flat):
    """
    Embedding lookup on GPU.

    Args:
        embedding_table: GPUBuffer of shape (vocab_size, embedding_dim)
        indices_flat: np.ndarray of token indices (flattened)

    Returns:
        GPUBuffer of embedded tokens
    """
    device = embedding_table.device
    vocab_size, embedding_dim = embedding_table.shape
    num_indices = len(indices_flat)

    # Upload indices
    indices_buffer = device.create_buffer_with_data(
        data=indices_flat.astype(np.uint32), usage=wgpu.BufferUsage.STORAGE
    )

    # Create output buffer
    output = create_gpu_buffer((num_indices, embedding_dim), device=device)

    # Create params buffer
    params_data = np.array([num_indices, embedding_dim], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params_data, usage=wgpu.BufferUsage.UNIFORM
    )

    # Create shader and pipeline
    shader_module = device.create_shader_module(code=EMBEDDING_LOOKUP_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": embedding_table.buffer,
                    "offset": 0,
                    "size": embedding_table.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": indices_buffer,
                    "offset": 0,
                    "size": indices_flat.nbytes,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params_data.nbytes,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(num_indices / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated matrix multiplication using WGPU.
    Falls back to NumPy if WGPU is unavailable.

    Args:
        a: Matrix A of shape (M, K)
        b: Matrix B of shape (K, N)

    Returns:
        Result matrix C of shape (M, N)
    """
    device = get_device()
    if device is None:
        # Fallback to NumPy
        return np.dot(a, b)

    # Ensure float32 and contiguous
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible shapes: ({M}, {K}) x ({K2}, {N})"

    # Create GPU buffers
    dims_data = np.array([M, K, N], dtype=np.uint32)
    dims_buffer = device.create_buffer_with_data(
        data=dims_data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )

    a_buffer = device.create_buffer_with_data(
        data=a, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    b_buffer = device.create_buffer_with_data(
        data=b, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    result_size = M * N * 4  # 4 bytes per float32
    result_buffer = device.create_buffer(
        size=result_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create shader module and pipeline
    shader_module = device.create_shader_module(code=MATMUL_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": dims_buffer,
                    "offset": 0,
                    "size": dims_data.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {"buffer": a_buffer, "offset": 0, "size": a.nbytes},
            },
            {
                "binding": 2,
                "resource": {"buffer": b_buffer, "offset": 0, "size": b.nbytes},
            },
            {
                "binding": 3,
                "resource": {"buffer": result_buffer, "offset": 0, "size": result_size},
            },
        ],
    )

    # Encode and submit commands
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    # Dispatch workgroups (8x8 tile per workgroup)
    workgroups_x = math.ceil(N / 8)
    workgroups_y = math.ceil(M / 8)
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
    compute_pass.end()

    # Copy result to readable buffer
    result_read_buffer = device.create_buffer(
        size=result_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    command_encoder.copy_buffer_to_buffer(
        result_buffer, 0, result_read_buffer, 0, result_size
    )

    device.queue.submit([command_encoder.finish()])

    # Read result
    result_read_buffer.map_sync(wgpu.MapMode.READ)
    result_data = np.frombuffer(
        result_read_buffer.read_mapped(), dtype=np.float32
    ).copy()
    result_read_buffer.unmap()

    return result_data.reshape(M, N)


def dot(x, w):
    """
    Drop-in replacement for jnp.dot() that uses GPU.
    Handles batched inputs and JAX <-> NumPy conversion.

    Args:
        x: JAX array of shape (..., K)
        w: JAX array of shape (K, N)

    Returns:
        JAX array of shape (..., N)
    """
    import jax
    import jax.numpy as jnp

    # Check if we're being traced (inside jit/grad)
    # If so, fall back to JAX
    if isinstance(x, jax.core.Tracer):
        return jnp.dot(x, w)

    device = get_device()
    if device is None:
        # Fallback to JAX
        return jnp.dot(x, w)

    # Remember original shape for batched inputs
    original_shape = x.shape[:-1]

    # Flatten to 2D
    x_2d = x.reshape(-1, x.shape[-1])

    # Convert to NumPy and run GPU matmul
    x_np = np.array(x_2d, dtype=np.float32)
    w_np = np.array(w, dtype=np.float32)
    result_np = matmul(x_np, w_np)

    # Convert back to JAX and reshape
    result = jnp.array(result_np)
    return result.reshape(original_shape + (w.shape[-1],))


SOFTMAX_SHADER = """
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> shape: vec2<u32>; // [batch_size, dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let dim = shape.y;

    if (row >= shape.x) {
        return;
    }

    let offset = row * dim;

    // Find max for numerical stability
    var max_val = input[offset];
    for (var i = 1u; i < dim; i++) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute exp and sum
    var sum = 0.0;
    for (var i = 0u; i < dim; i++) {
        let exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (var i = 0u; i < dim; i++) {
        output[offset + i] = output[offset + i] / sum;
    }
}
"""


def softmax(x, axis: int = -1):
    """
    GPU-accelerated softmax with JAX tracing support.

    Args:
        x: Input array (JAX or NumPy)
        axis: Axis to apply softmax (default: -1)

    Returns:
        Softmax output
    """
    import jax
    import jax.numpy as jnp

    # Check if we're being traced (inside grad/jit)
    # If so, fall back to JAX
    if isinstance(x, jax.core.Tracer):
        return jax.nn.softmax(x, axis=axis)

    device = get_device()
    if device is None:
        # Fallback to JAX if available, else NumPy
        try:
            return jax.nn.softmax(jnp.array(x), axis=axis)
        except:
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    # Convert to NumPy for GPU processing
    x_np = np.array(x, dtype=np.float32)

    # Reshape to 2D for GPU processing
    original_shape = x_np.shape
    if axis != -1 and axis != len(x_np.shape) - 1:
        # Move axis to last dimension
        x_np = np.moveaxis(x_np, axis, -1)

    # Flatten all dimensions except last
    batch_size = np.prod(x_np.shape[:-1])
    dim = x_np.shape[-1]
    x_2d = x_np.reshape(batch_size, dim)

    x_2d = np.ascontiguousarray(x_2d, dtype=np.float32)

    # Create buffers
    shape_data = np.array([batch_size, dim], dtype=np.uint32)
    shape_buffer = device.create_buffer_with_data(
        data=shape_data, usage=wgpu.BufferUsage.UNIFORM
    )

    input_buffer = device.create_buffer_with_data(
        data=x_2d, usage=wgpu.BufferUsage.STORAGE
    )

    output_buffer = device.create_buffer(
        size=x_2d.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create pipeline
    shader_module = device.create_shader_module(code=SOFTMAX_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": input_buffer, "offset": 0, "size": x_2d.nbytes},
            },
            {
                "binding": 1,
                "resource": {"buffer": output_buffer, "offset": 0, "size": x_2d.nbytes},
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": shape_buffer,
                    "offset": 0,
                    "size": shape_data.nbytes,
                },
            },
        ],
    )

    # Execute
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(batch_size / 256), 1, 1)
    compute_pass.end()

    # Read result
    result_read_buffer = device.create_buffer(
        size=x_2d.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    command_encoder.copy_buffer_to_buffer(
        output_buffer, 0, result_read_buffer, 0, x_2d.nbytes
    )

    device.queue.submit([command_encoder.finish()])

    result_read_buffer.map_sync(wgpu.MapMode.READ)
    result_data = np.frombuffer(
        result_read_buffer.read_mapped(), dtype=np.float32
    ).copy()
    result_read_buffer.unmap()

    result = result_data.reshape(x_np.shape)

    # Restore original axis order if needed
    if axis != -1 and axis != len(original_shape) - 1:
        result = np.moveaxis(result, -1, axis)

    result = result.reshape(original_shape)

    # Return as JAX array if input was JAX
    return jnp.array(result)


def matmul_batched(a, b):
    """
    Batched matrix multiplication with GPU acceleration.
    Handles arbitrary batch dimensions.

    Args:
        a: Array of shape (..., M, K)
        b: Array of shape (..., K, N)

    Returns:
        Array of shape (..., M, N)
    """
    import jax
    import jax.numpy as jnp

    # Check if we're being traced
    if isinstance(a, jax.core.Tracer) or isinstance(b, jax.core.Tracer):
        return jnp.matmul(a, b)

    device = get_device()
    if device is None:
        return jnp.matmul(a, b)

    # Convert to NumPy
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)

    # Get shapes
    original_shape_a = a_np.shape
    original_shape_b = b_np.shape

    # Reshape to 3D: (batch, M, K) and (batch, K, N)
    # Handle batch dimensions
    batch_dims_a = original_shape_a[:-2]
    batch_dims_b = original_shape_b[:-2]

    assert batch_dims_a == batch_dims_b, (
        f"Batch dimensions must match: {batch_dims_a} vs {batch_dims_b}"
    )

    batch_size = int(np.prod(batch_dims_a)) if batch_dims_a else 1
    M, K = original_shape_a[-2:]
    K2, N = original_shape_b[-2:]

    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    # Reshape to (batch_size, M, K) and (batch_size, K, N)
    a_3d = a_np.reshape(batch_size, M, K)
    b_3d = b_np.reshape(batch_size, K, N)

    # Process each batch on GPU
    results = []
    for i in range(batch_size):
        result = matmul(a_3d[i], b_3d[i])
        results.append(result)

    # Stack results
    result_3d = np.stack(results, axis=0)

    # Reshape back to original batch dimensions
    output_shape = batch_dims_a + (M, N)
    result = result_3d.reshape(output_shape)

    return jnp.array(result)


LAYER_NORM_SHADER = """
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> shape: vec2<u32>; // [batch_size, dim]

const EPS: f32 = 1e-5;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let dim = shape.y;

    if (row >= shape.x) {
        return;
    }

    let offset = row * dim;

    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < dim; i++) {
        sum += input[offset + i];
    }
    let mean = sum / f32(dim);

    // Compute variance
    var var_sum = 0.0;
    for (var i = 0u; i < dim; i++) {
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    let variance = var_sum / f32(dim);

    // Normalize and scale
    let std_dev = sqrt(variance + EPS);
    for (var i = 0u; i < dim; i++) {
        let normalized = (input[offset + i] - mean) / std_dev;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}
"""


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    GPU-accelerated layer normalization.

    Args:
        x: Input array of shape (..., dim)
        gamma: Scale parameter of shape (dim,)
        beta: Shift parameter of shape (dim,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized output
    """
    import jax
    import jax.numpy as jnp

    # Check if we're being traced
    if isinstance(x, jax.core.Tracer):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        norm = (x - mean) / jnp.sqrt(var + eps)
        return norm * gamma + beta

    device = get_device()
    if device is None:
        # Fallback to JAX
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        norm = (x - mean) / jnp.sqrt(var + eps)
        return norm * gamma + beta

    # Convert to NumPy
    x_np = np.array(x, dtype=np.float32)
    gamma_np = np.array(gamma, dtype=np.float32)
    beta_np = np.array(beta, dtype=np.float32)

    original_shape = x_np.shape
    dim = original_shape[-1]
    batch_size = int(np.prod(original_shape[:-1]))

    # Reshape to 2D
    x_2d = x_np.reshape(batch_size, dim)
    x_2d = np.ascontiguousarray(x_2d)

    # Create buffers
    shape_data = np.array([batch_size, dim], dtype=np.uint32)
    shape_buffer = device.create_buffer_with_data(
        data=shape_data, usage=wgpu.BufferUsage.UNIFORM
    )

    input_buffer = device.create_buffer_with_data(
        data=x_2d, usage=wgpu.BufferUsage.STORAGE
    )

    gamma_buffer = device.create_buffer_with_data(
        data=gamma_np, usage=wgpu.BufferUsage.STORAGE
    )

    beta_buffer = device.create_buffer_with_data(
        data=beta_np, usage=wgpu.BufferUsage.STORAGE
    )

    output_buffer = device.create_buffer(
        size=x_2d.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create pipeline
    shader_module = device.create_shader_module(code=LAYER_NORM_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": input_buffer, "offset": 0, "size": x_2d.nbytes},
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": gamma_buffer,
                    "offset": 0,
                    "size": gamma_np.nbytes,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": beta_buffer,
                    "offset": 0,
                    "size": beta_np.nbytes,
                },
            },
            {
                "binding": 3,
                "resource": {"buffer": output_buffer, "offset": 0, "size": x_2d.nbytes},
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": shape_buffer,
                    "offset": 0,
                    "size": shape_data.nbytes,
                },
            },
        ],
    )

    # Execute
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(batch_size / 256), 1, 1)
    compute_pass.end()

    # Read result
    result_read_buffer = device.create_buffer(
        size=x_2d.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    command_encoder.copy_buffer_to_buffer(
        output_buffer, 0, result_read_buffer, 0, x_2d.nbytes
    )

    device.queue.submit([command_encoder.finish()])

    result_read_buffer.map_sync(wgpu.MapMode.READ)
    result_data = np.frombuffer(
        result_read_buffer.read_mapped(), dtype=np.float32
    ).copy()
    result_read_buffer.unmap()

    result = result_data.reshape(original_shape)
    return jnp.array(result)


GELU_SHADER = """
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

const SQRT_2_OVER_PI: f32 = 0.7978845608;  // sqrt(2/pi)
const COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let x = input[idx];

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
    let tanh_inner = tanh(inner);

    output[idx] = 0.5 * x * (1.0 + tanh_inner);
}
"""


def gelu(x):
    """
    GPU-accelerated GELU activation.

    Args:
        x: Input array of any shape

    Returns:
        GELU(x)
    """
    import jax
    import jax.numpy as jnp

    # Check if we're being traced
    if isinstance(x, jax.core.Tracer):
        return jax.nn.gelu(x)

    device = get_device()
    if device is None:
        return jax.nn.gelu(x)

    # Convert to NumPy
    x_np = np.array(x, dtype=np.float32)
    original_shape = x_np.shape

    # Flatten for GPU processing
    x_flat = x_np.flatten()
    x_flat = np.ascontiguousarray(x_flat)
    size = x_flat.size

    # Create buffers
    size_data = np.array([size], dtype=np.uint32)
    size_buffer = device.create_buffer_with_data(
        data=size_data, usage=wgpu.BufferUsage.UNIFORM
    )

    input_buffer = device.create_buffer_with_data(
        data=x_flat, usage=wgpu.BufferUsage.STORAGE
    )

    output_buffer = device.create_buffer(
        size=x_flat.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create pipeline
    shader_module = device.create_shader_module(code=GELU_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": input_buffer,
                    "offset": 0,
                    "size": x_flat.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": output_buffer,
                    "offset": 0,
                    "size": x_flat.nbytes,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": size_buffer,
                    "offset": 0,
                    "size": size_data.nbytes,
                },
            },
        ],
    )

    # Execute
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(size / 256), 1, 1)
    compute_pass.end()

    # Read result
    result_read_buffer = device.create_buffer(
        size=x_flat.nbytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    command_encoder.copy_buffer_to_buffer(
        output_buffer, 0, result_read_buffer, 0, x_flat.nbytes
    )

    device.queue.submit([command_encoder.finish()])

    result_read_buffer.map_sync(wgpu.MapMode.READ)
    result_data = np.frombuffer(
        result_read_buffer.read_mapped(), dtype=np.float32
    ).copy()
    result_read_buffer.unmap()

    result = result_data.reshape(original_shape)
    return jnp.array(result)


EMBEDDING_LOOKUP_SHADER = """
@group(0) @binding(0) var<storage, read> embedding: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>; // [num_indices, embedding_dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_indices = params.x;
    let embedding_dim = params.y;

    if (idx >= num_indices) {
        return;
    }

    // Get token index
    let token_idx = indices[idx];

    // Copy entire embedding row
    let src_offset = token_idx * embedding_dim;
    let dst_offset = idx * embedding_dim;

    for (var i = 0u; i < embedding_dim; i++) {
        output[dst_offset + i] = embedding[src_offset + i];
    }
}
"""


def embedding_lookup(embedding_table, indices):
    """
    GPU-accelerated embedding lookup.

    Args:
        embedding_table: Embedding matrix of shape (vocab_size, embedding_dim)
        indices: Token indices of shape (batch_size, seq_len) or any shape

    Returns:
        Embedded tokens of shape (*indices.shape, embedding_dim)
    """
    import jax
    import jax.numpy as jnp

    # Check if we're being traced (either embedding_table or indices)
    if isinstance(indices, jax.core.Tracer) or isinstance(
        embedding_table, jax.core.Tracer
    ):
        return embedding_table[indices]

    device = get_device()
    if device is None:
        return embedding_table[indices]

    # Convert to NumPy
    embedding_np = np.array(embedding_table, dtype=np.float32)
    indices_np = np.array(indices, dtype=np.int32)

    original_indices_shape = indices_np.shape
    vocab_size, embedding_dim = embedding_np.shape

    # Flatten indices
    indices_flat = indices_np.flatten().astype(np.uint32)
    num_indices = indices_flat.size

    # Create buffers
    params_data = np.array([num_indices, embedding_dim], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params_data, usage=wgpu.BufferUsage.UNIFORM
    )

    embedding_buffer = device.create_buffer_with_data(
        data=embedding_np, usage=wgpu.BufferUsage.STORAGE
    )

    indices_buffer = device.create_buffer_with_data(
        data=indices_flat, usage=wgpu.BufferUsage.STORAGE
    )

    output_size = num_indices * embedding_dim * 4
    output_buffer = device.create_buffer(
        size=output_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create pipeline
    shader_module = device.create_shader_module(code=EMBEDDING_LOOKUP_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": embedding_buffer,
                    "offset": 0,
                    "size": embedding_np.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": indices_buffer,
                    "offset": 0,
                    "size": indices_flat.nbytes,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": output_buffer, "offset": 0, "size": output_size},
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params_data.nbytes,
                },
            },
        ],
    )

    # Execute
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(num_indices / 256), 1, 1)
    compute_pass.end()

    # Read result
    result_read_buffer = device.create_buffer(
        size=output_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    command_encoder.copy_buffer_to_buffer(
        output_buffer, 0, result_read_buffer, 0, output_size
    )

    device.queue.submit([command_encoder.finish()])

    result_read_buffer.map_sync(wgpu.MapMode.READ)
    result_data = np.frombuffer(
        result_read_buffer.read_mapped(), dtype=np.float32
    ).copy()
    result_read_buffer.unmap()

    # Reshape to original indices shape + embedding dimension
    output_shape = original_indices_shape + (embedding_dim,)
    result = result_data.reshape(output_shape)

    return jnp.array(result)


CAUSAL_MASK_SHADER = """
@group(0) @binding(0) var<storage, read_write> mask: array<f32>;
@group(0) @binding(1) var<uniform> seq_len: u32;

const MASK_VALUE: f32 = -1e9;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= seq_len || col >= seq_len) {
        return;
    }

    let idx = row * seq_len + col;

    // Upper triangular mask (col > row gets masked)
    if (col > row) {
        mask[idx] = MASK_VALUE;
    } else {
        mask[idx] = 0.0;
    }
}
"""


def create_causal_mask(seq_len):
    """
    GPU-accelerated causal mask creation.
    Returns upper triangular mask with -1e9 above diagonal.

    Args:
        seq_len: Sequence length

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    import jax.numpy as jnp

    device = get_device()
    if device is None:
        # Fallback to JAX
        return jnp.triu(jnp.ones((seq_len, seq_len)), k=1) * -1e9

    # Create output buffer
    size = seq_len * seq_len * 4
    output_buffer = device.create_buffer(
        size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    # Create uniform buffer
    seq_len_data = np.array([seq_len], dtype=np.uint32)
    seq_len_buffer = device.create_buffer_with_data(
        data=seq_len_data, usage=wgpu.BufferUsage.UNIFORM
    )

    # Create pipeline
    shader_module = device.create_shader_module(code=CAUSAL_MASK_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": output_buffer, "offset": 0, "size": size},
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": seq_len_buffer,
                    "offset": 0,
                    "size": seq_len_data.nbytes,
                },
            },
        ],
    )

    # Execute
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(
        math.ceil(seq_len / 16), math.ceil(seq_len / 16), 1
    )
    compute_pass.end()

    # Read result
    result_read_buffer = device.create_buffer(
        size=size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )
    command_encoder.copy_buffer_to_buffer(output_buffer, 0, result_read_buffer, 0, size)

    device.queue.submit([command_encoder.finish()])

    result_read_buffer.map_sync(wgpu.MapMode.READ)
    result_data = np.frombuffer(
        result_read_buffer.read_mapped(), dtype=np.float32
    ).copy()
    result_read_buffer.unmap()

    result = result_data.reshape(seq_len, seq_len)
    return jnp.array(result)


# =======================================
@dataclasses.dataclass
class GPUBuffer:
    """Handle to GPU buffer with shape info"""

    buffer: object  # wgpu.Buffer
    shape: tuple
    size: int  # Total number of elements
    device: object  # wgpu.Device


def create_gpu_buffer(shape, data=None, device=None):
    """
    Create a GPU buffer.

    Args:
        shape: Tuple defining tensor shape
        data: Optional NumPy array to initialize with
        device: WGPU device (uses default if None)

    Returns:
        GPUBuffer
    """
    device = device or get_device()
    size = int(np.prod(shape))
    buffer_size = size * 4  # 4 bytes per float32

    if data is not None:
        data_np = np.ascontiguousarray(data, dtype=np.float32)
        assert data_np.size == size, (
            f"Data size {data_np.size} doesn't match shape {shape}"
        )
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

    return GPUBuffer(buffer=buffer, shape=shape, size=size, device=device)


def gpu_to_numpy(gpu_buffer):
    """Read GPU buffer to NumPy array"""
    buffer_size = gpu_buffer.size * 4
    read_buffer = gpu_buffer.device.create_buffer(
        size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    encoder = gpu_buffer.device.create_command_encoder()
    encoder.copy_buffer_to_buffer(gpu_buffer.buffer, 0, read_buffer, 0, buffer_size)
    gpu_buffer.device.queue.submit([encoder.finish()])

    read_buffer.map_sync(wgpu.MapMode.READ)
    data = np.frombuffer(read_buffer.read_mapped(), dtype=np.float32).copy()
    read_buffer.unmap()

    return data.reshape(gpu_buffer.shape)


def numpy_to_gpu(gpu_buffer, data):
    """Write NumPy array to GPU buffer"""
    data_np = np.ascontiguousarray(data, dtype=np.float32)
    assert data_np.size == gpu_buffer.size, (
        f"Size mismatch: {data_np.size} vs {gpu_buffer.size}"
    )

    temp_buffer = gpu_buffer.device.create_buffer_with_data(
        data=data_np, usage=wgpu.BufferUsage.COPY_SRC
    )

    encoder = gpu_buffer.device.create_command_encoder()
    encoder.copy_buffer_to_buffer(
        temp_buffer, 0, gpu_buffer.buffer, 0, gpu_buffer.size * 4
    )
    gpu_buffer.device.queue.submit([encoder.finish()])


@dataclasses.dataclass
class GPULayerParams:
    """GPU-resident transformer layer parameters"""

    # Attention
    attn_wq: GPUBuffer
    attn_wk: GPUBuffer
    attn_wv: GPUBuffer
    attn_wo: GPUBuffer

    # Feed-forward
    ff_w1: GPUBuffer
    ff_b1: GPUBuffer
    ff_w2: GPUBuffer
    ff_b2: GPUBuffer

    # Layer norm
    ln_gamma1: GPUBuffer
    ln_beta1: GPUBuffer
    ln_gamma2: GPUBuffer
    ln_beta2: GPUBuffer


@dataclasses.dataclass
class GPUModelParams:
    """Complete GPU-resident model"""

    embedding: GPUBuffer
    pos_encoding: GPUBuffer
    layers: list  # List[GPULayerParams]


def create_gpu_layer_params(embedding_dim, device=None):
    """Initialize GPU layer parameters with random values"""
    device = device or get_device()
    dim = embedding_dim

    return GPULayerParams(
        # Attention weights
        attn_wq=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.01, (dim, dim)), device
        ),
        attn_wk=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.01, (dim, dim)), device
        ),
        attn_wv=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.01, (dim, dim)), device
        ),
        attn_wo=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.01, (dim, dim)), device
        ),
        # Feed-forward weights
        ff_w1=create_gpu_buffer(
            (dim, 4 * dim), np.random.normal(0, 0.01, (dim, 4 * dim)), device
        ),
        ff_b1=create_gpu_buffer((4 * dim,), np.zeros(4 * dim), device),
        ff_w2=create_gpu_buffer(
            (4 * dim, dim), np.random.normal(0, 0.01, (4 * dim, dim)), device
        ),
        ff_b2=create_gpu_buffer((dim,), np.zeros(dim), device),
        # Layer norm parameters
        ln_gamma1=create_gpu_buffer((dim,), np.ones(dim), device),
        ln_beta1=create_gpu_buffer((dim,), np.zeros(dim), device),
        ln_gamma2=create_gpu_buffer((dim,), np.ones(dim), device),
        ln_beta2=create_gpu_buffer((dim,), np.zeros(dim), device),
    )


def create_gpu_model_params(
    vocab_size, embedding_dim, context_length, n_layers, device=None
):
    """Initialize complete GPU model"""
    device = device or get_device()

    # Create embedding
    embedding_data = np.random.normal(0, 0.01, (vocab_size, embedding_dim)).astype(
        np.float32
    )
    embedding = create_gpu_buffer((vocab_size, embedding_dim), embedding_data, device)

    # Create positional encoding
    pos_encoding_data = positional_encoding(context_length, embedding_dim)
    pos_encoding = create_gpu_buffer(
        (context_length, embedding_dim), pos_encoding_data, device
    )

    # Create layers
    layers = [create_gpu_layer_params(embedding_dim, device) for _ in range(n_layers)]

    return GPUModelParams(embedding=embedding, pos_encoding=pos_encoding, layers=layers)


def positional_encoding(seq_len, dim):
    """Generate positional encoding (NumPy version for initialization)"""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angle_rads = pos * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1).astype(np.float32)
    return pos_encoding


def gpu_layer_to_dict(layer):
    """Convert GPU layer to dict of NumPy arrays"""
    return {
        "attn_wq": gpu_to_numpy(layer.attn_wq),
        "attn_wk": gpu_to_numpy(layer.attn_wk),
        "attn_wv": gpu_to_numpy(layer.attn_wv),
        "attn_wo": gpu_to_numpy(layer.attn_wo),
        "ff_w1": gpu_to_numpy(layer.ff_w1),
        "ff_b1": gpu_to_numpy(layer.ff_b1),
        "ff_w2": gpu_to_numpy(layer.ff_w2),
        "ff_b2": gpu_to_numpy(layer.ff_b2),
        "ln_gamma1": gpu_to_numpy(layer.ln_gamma1),
        "ln_beta1": gpu_to_numpy(layer.ln_beta1),
        "ln_gamma2": gpu_to_numpy(layer.ln_gamma2),
        "ln_beta2": gpu_to_numpy(layer.ln_beta2),
    }


def dict_to_gpu_layer(data, embedding_dim, device=None):
    """Create GPU layer from dict of NumPy arrays"""
    device = device or get_device()
    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer((dim, dim), data["attn_wq"], device),
        attn_wk=create_gpu_buffer((dim, dim), data["attn_wk"], device),
        attn_wv=create_gpu_buffer((dim, dim), data["attn_wv"], device),
        attn_wo=create_gpu_buffer((dim, dim), data["attn_wo"], device),
        ff_w1=create_gpu_buffer((dim, 4 * dim), data["ff_w1"], device),
        ff_b1=create_gpu_buffer((4 * dim,), data["ff_b1"], device),
        ff_w2=create_gpu_buffer((4 * dim, dim), data["ff_w2"], device),
        ff_b2=create_gpu_buffer((dim,), data["ff_b2"], device),
        ln_gamma1=create_gpu_buffer((dim,), data["ln_gamma1"], device),
        ln_beta1=create_gpu_buffer((dim,), data["ln_beta1"], device),
        ln_gamma2=create_gpu_buffer((dim,), data["ln_gamma2"], device),
        ln_beta2=create_gpu_buffer((dim,), data["ln_beta2"], device),
    )


# ============================================================================
# BACKWARD PASS SHADERS
# ============================================================================

MATMUL_BACKWARD_SHADER = """
// Computes gradients for matrix multiplication C = A @ B
// Given dL/dC, compute dL/dA and dL/dB

struct Dimensions {
    M: u32,  // rows of A
    K: u32,  // cols of A, rows of B
    N: u32,  // cols of B
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;           // Forward: A input
@group(0) @binding(2) var<storage, read> b: array<f32>;           // Forward: B input
@group(0) @binding(3) var<storage, read> grad_output: array<f32>; // dL/dC
@group(0) @binding(4) var<storage, read_write> grad_a: array<f32>; // dL/dA (output)
@group(0) @binding(5) var<storage, read_write> grad_b: array<f32>; // dL/dB (output)

@compute @workgroup_size(8, 8)
fn compute_grad_a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.M || col >= dims.K) {
        return;
    }

    // dL/dA = grad_output @ B^T
    var sum = 0.0;
    for (var i = 0u; i < dims.N; i++) {
        sum += grad_output[row * dims.N + i] * b[col * dims.N + i];
    }
    grad_a[row * dims.K + col] = sum;
}

@compute @workgroup_size(8, 8)
fn compute_grad_b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.K || col >= dims.N) {
        return;
    }

    // dL/dB = A^T @ grad_output
    var sum = 0.0;
    for (var i = 0u; i < dims.M; i++) {
        sum += a[i * dims.K + row] * grad_output[i * dims.N + col];
    }
    grad_b[row * dims.N + col] = sum;
}
"""

LAYER_NORM_BACKWARD_SHADER = """
// Backward pass for layer normalization

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_gamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_beta: array<f32>;
@group(0) @binding(6) var<uniform> shape: vec2<u32>; // [batch_size, dim]

const EPS: f32 = 1e-5;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let dim = shape.y;

    if (row >= shape.x) {
        return;
    }

    let offset = row * dim;

    // Recompute mean and variance
    var sum = 0.0;
    for (var i = 0u; i < dim; i++) {
        sum += input[offset + i];
    }
    let mean = sum / f32(dim);

    var var_sum = 0.0;
    for (var i = 0u; i < dim; i++) {
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    let variance = var_sum / f32(dim);
    let std_dev = sqrt(variance + EPS);

    // Compute intermediate values
    var dstd = 0.0;
    var dmean = 0.0;

    for (var i = 0u; i < dim; i++) {
        let x_hat = (input[offset + i] - mean) / std_dev;
        let grad_out = grad_output[offset + i];

        // Gradient w.r.t. normalized value
        dstd += grad_out * gamma[i] * x_hat;
        dmean += grad_out * gamma[i];

        // Accumulate gradient for gamma and beta (across batch)
        atomicAdd(&grad_gamma[i], grad_out * x_hat);
        atomicAdd(&grad_beta[i], grad_out);
    }

    dstd *= -1.0 / (std_dev * std_dev);
    dmean *= -1.0 / std_dev;

    // Gradient w.r.t. input
    for (var i = 0u; i < dim; i++) {
        let x_centered = input[offset + i] - mean;
        let grad_out = grad_output[offset + i];

        var grad_in = grad_out * gamma[i] / std_dev;
        grad_in += dstd * 2.0 * x_centered / f32(dim);
        grad_in += dmean / f32(dim);

        grad_input[offset + i] = grad_in;
    }
}
"""

GELU_BACKWARD_SHADER = """
// Backward pass for GELU activation

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let x = input[idx];
    let x_sq = x * x;
    let x_cubed = x_sq * x;

    // GELU derivative
    let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
    let tanh_val = tanh(inner);
    let sech_sq = 1.0 - tanh_val * tanh_val;

    let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x_sq);
    let d_gelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech_sq * d_inner;

    grad_input[idx] = grad_output[idx] * d_gelu;
}
"""

SOFTMAX_BACKWARD_SHADER = """
// Backward pass for softmax

@group(0) @binding(0) var<storage, read> output: array<f32>;      // Softmax output from forward
@group(0) @binding(1) var<storage, read> grad_output: array<f32>; // Gradient from next layer
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> shape: vec2<u32>; // [batch_size, dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let dim = shape.y;

    if (row >= shape.x) {
        return;
    }

    let offset = row * dim;

    // Compute sum of (grad_output * softmax_output)
    var sum = 0.0;
    for (var i = 0u; i < dim; i++) {
        sum += grad_output[offset + i] * output[offset + i];
    }

    // Compute gradient
    for (var i = 0u; i < dim; i++) {
        let s = output[offset + i];
        grad_input[offset + i] = s * (grad_output[offset + i] - sum);
    }
}
"""

CROSS_ENTROPY_BACKWARD_SHADER = """
// Backward pass for cross-entropy loss
// Computes gradient of loss w.r.t. logits

@group(0) @binding(0) var<storage, read> probs: array<f32>;        // Softmax probabilities
@group(0) @binding(1) var<storage, read> targets: array<u32>;      // Target indices
@group(0) @binding(2) var<storage, read_write> grad_logits: array<f32>;
@group(0) @binding(3) var<uniform> params: vec3<u32>; // [batch_size, seq_len, vocab_size]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_size = params.x;
    let seq_len = params.y;
    let vocab_size = params.z;
    let total_predictions = batch_size * seq_len;

    if (idx >= total_predictions * vocab_size) {
        return;
    }

    let pred_idx = idx / vocab_size;
    let class_idx = idx % vocab_size;

    if (pred_idx >= total_predictions) {
        return;
    }

    let target_idx = targets[pred_idx];
    let prob = probs[idx];

    // Gradient: (prob - 1) if correct class, else prob
    // Normalized by batch_size * seq_len
    var grad = prob;
    if (class_idx == target_idx) {
        grad -= 1.0;
    }
    grad /= f32(batch_size * seq_len);

    grad_logits[idx] = grad;
}
"""

EMBEDDING_BACKWARD_SHADER = """
// Backward pass for embedding lookup
// Accumulates gradients for embedding table

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;  // Gradient from next layer
@group(0) @binding(1) var<storage, read> indices: array<u32>;      // Token indices
@group(0) @binding(2) var<storage, read_write> grad_embedding: array<f32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>; // [num_indices, embedding_dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_indices = params.x;
    let embedding_dim = params.y;

    if (idx >= num_indices) {
        return;
    }

    // Get token index
    let token_idx = indices[idx];

    // Accumulate gradient to embedding table
    let grad_offset = idx * embedding_dim;
    let emb_offset = token_idx * embedding_dim;

    for (var i = 0u; i < embedding_dim; i++) {
        atomicAdd(&grad_embedding[emb_offset + i], grad_output[grad_offset + i]);
    }
}
"""

ADAMW_SHADER = """
// AdamW optimizer update

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;     // First moment
@group(0) @binding(3) var<storage, read_write> v: array<f32>;     // Second moment
@group(0) @binding(4) var<storage, read_write> params_out: array<f32>;
@group(0) @binding(5) var<uniform> hyperparams: vec4<f32>; // [lr, beta1, beta2, weight_decay]
@group(0) @binding(6) var<uniform> step: u32;
@group(0) @binding(7) var<uniform> size: u32;

const EPS: f32 = 1e-8;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let lr = hyperparams.x;
    let beta1 = hyperparams.y;
    let beta2 = hyperparams.z;
    let weight_decay = hyperparams.w;

    let param = params[idx];
    let grad = grads[idx];

    // Update biased first moment
    let m_new = beta1 * m[idx] + (1.0 - beta1) * grad;
    m[idx] = m_new;

    // Update biased second moment
    let v_new = beta2 * v[idx] + (1.0 - beta2) * grad * grad;
    v[idx] = v_new;

    // Bias correction
    let step_f = f32(step);
    let m_hat = m_new / (1.0 - pow(beta1, step_f));
    let v_hat = v_new / (1.0 - pow(beta2, step_f));

    // AdamW update: includes weight decay on parameters directly
    let update = m_hat / (sqrt(v_hat) + EPS) + weight_decay * param;
    params_out[idx] = param - lr * update;
}"""


def matmul_backward(a, b, grad_output, device=None):
    """
    Compute gradients for matrix multiplication C = A @ B.

    Args:
        a: GPUBuffer - forward input A (M, K)
        b: GPUBuffer - forward input B (K, N)
        grad_output: GPUBuffer - gradient dL/dC (M, N)
        device: WGPU device

    Returns:
        (grad_a, grad_b): Tuple of GPUBuffer gradients
    """
    device = device or get_device()

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Create output buffers
    grad_a = create_gpu_buffer((M, K), device=device)
    grad_b = create_gpu_buffer((K, N), device=device)

    # Create uniform buffer
    dims_data = np.array([M, K, N], dtype=np.uint32)
    dims_buffer = device.create_buffer_with_data(
        data=dims_data, usage=wgpu.BufferUsage.UNIFORM
    )

    # Create shader and pipeline
    shader_module = device.create_shader_module(code=MATMUL_BACKWARD_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    # Compute grad_a
    pipeline_grad_a = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "compute_grad_a"},
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": dims_buffer,
                    "offset": 0,
                    "size": dims_data.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {"buffer": a.buffer, "offset": 0, "size": a.size * 4},
            },
            {
                "binding": 2,
                "resource": {"buffer": b.buffer, "offset": 0, "size": b.size * 4},
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": grad_output.buffer,
                    "offset": 0,
                    "size": grad_output.size * 4,
                },
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": grad_a.buffer,
                    "offset": 0,
                    "size": grad_a.size * 4,
                },
            },
            {
                "binding": 5,
                "resource": {
                    "buffer": grad_b.buffer,
                    "offset": 0,
                    "size": grad_b.size * 4,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    pass1 = encoder.begin_compute_pass()
    pass1.set_pipeline(pipeline_grad_a)
    pass1.set_bind_group(0, bind_group)
    pass1.dispatch_workgroups(math.ceil(K / 8), math.ceil(M / 8), 1)
    pass1.end()

    # Compute grad_b
    pipeline_grad_b = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "compute_grad_b"},
    )

    pass2 = encoder.begin_compute_pass()
    pass2.set_pipeline(pipeline_grad_b)
    pass2.set_bind_group(0, bind_group)
    pass2.dispatch_workgroups(math.ceil(N / 8), math.ceil(K / 8), 1)
    pass2.end()

    device.queue.submit([encoder.finish()])

    return grad_a, grad_b


# ============================================================================
# OPTIMIZER
# ============================================================================


@dataclasses.dataclass
class GPUOptimizerState:
    """AdamW optimizer state on GPU"""

    m_embedding: GPUBuffer
    v_embedding: GPUBuffer
    m_layers: list  # List of GPULayerParams (moment estimates)
    v_layers: list  # List of GPULayerParams (moment estimates)
    step: int


def create_optimizer_state(model_params):
    """Initialize optimizer state (zero moments)"""
    device = model_params.embedding.device

    # Create zero buffers for embedding
    m_embedding = create_gpu_buffer(
        model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
        device,
    )
    v_embedding = create_gpu_buffer(
        model_params.embedding.shape,
        np.zeros(model_params.embedding.shape, dtype=np.float32),
        device,
    )

    # Create zero buffers for each layer
    m_layers = []
    v_layers = []
    for layer in model_params.layers:
        # Get embedding_dim from first weight matrix
        embedding_dim = layer.attn_wq.shape[0]

        m_layer = GPULayerParams(
            attn_wq=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            attn_wk=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            attn_wv=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            attn_wo=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            ff_w1=create_gpu_buffer(
                (embedding_dim, 4 * embedding_dim),
                np.zeros((embedding_dim, 4 * embedding_dim)),
                device,
            ),
            ff_b1=create_gpu_buffer(
                (4 * embedding_dim,), np.zeros(4 * embedding_dim), device
            ),
            ff_w2=create_gpu_buffer(
                (4 * embedding_dim, embedding_dim),
                np.zeros((4 * embedding_dim, embedding_dim)),
                device,
            ),
            ff_b2=create_gpu_buffer((embedding_dim,), np.zeros(embedding_dim), device),
            ln_gamma1=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
            ln_beta1=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
            ln_gamma2=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
            ln_beta2=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
        )

        v_layer = GPULayerParams(
            attn_wq=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            attn_wk=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            attn_wv=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            attn_wo=create_gpu_buffer(
                (embedding_dim, embedding_dim),
                np.zeros((embedding_dim, embedding_dim)),
                device,
            ),
            ff_w1=create_gpu_buffer(
                (embedding_dim, 4 * embedding_dim),
                np.zeros((embedding_dim, 4 * embedding_dim)),
                device,
            ),
            ff_b1=create_gpu_buffer(
                (4 * embedding_dim,), np.zeros(4 * embedding_dim), device
            ),
            ff_w2=create_gpu_buffer(
                (4 * embedding_dim, embedding_dim),
                np.zeros((4 * embedding_dim, embedding_dim)),
                device,
            ),
            ff_b2=create_gpu_buffer((embedding_dim,), np.zeros(embedding_dim), device),
            ln_gamma1=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
            ln_beta1=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
            ln_gamma2=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
            ln_beta2=create_gpu_buffer(
                (embedding_dim,), np.zeros(embedding_dim), device
            ),
        )

        m_layers.append(m_layer)
        v_layers.append(v_layer)

    return GPUOptimizerState(
        m_embedding=m_embedding,
        v_embedding=v_embedding,
        m_layers=m_layers,
        v_layers=v_layers,
        step=0,
    )


def adamw_update(param, grad, m, v, step, lr, beta1=0.9, beta2=0.95, weight_decay=0.1):
    """
    Apply AdamW update to a single parameter buffer.

    Args:
        param: GPUBuffer - current parameters
        grad: GPUBuffer - gradients
        m: GPUBuffer - first moment estimate
        v: GPUBuffer - second moment estimate
        step: int - current step number
        lr: float - learning rate
        beta1, beta2, weight_decay: optimizer hyperparameters

    Returns:
        GPUBuffer - updated parameters
    """
    device = param.device
    size = param.size

    # Create output buffer
    param_out = create_gpu_buffer(param.shape, device=device)

    # Create hyperparameters buffer
    hyperparams = np.array([lr, beta1, beta2, weight_decay], dtype=np.float32)
    hyperparams_buffer = device.create_buffer_with_data(
        data=hyperparams, usage=wgpu.BufferUsage.UNIFORM
    )

    step_buffer = device.create_buffer_with_data(
        data=np.array([step], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    size_buffer = device.create_buffer_with_data(
        data=np.array([size], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    # Create shader and pipeline
    shader_module = device.create_shader_module(code=ADAMW_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 6,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 7,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": param.buffer, "offset": 0, "size": size * 4},
            },
            {
                "binding": 1,
                "resource": {"buffer": grad.buffer, "offset": 0, "size": size * 4},
            },
            {
                "binding": 2,
                "resource": {"buffer": m.buffer, "offset": 0, "size": size * 4},
            },
            {
                "binding": 3,
                "resource": {"buffer": v.buffer, "offset": 0, "size": size * 4},
            },
            {
                "binding": 4,
                "resource": {"buffer": param_out.buffer, "offset": 0, "size": size * 4},
            },
            {
                "binding": 5,
                "resource": {"buffer": hyperparams_buffer, "offset": 0, "size": 16},
            },
            {"binding": 6, "resource": {"buffer": step_buffer, "offset": 0, "size": 4}},
            {"binding": 7, "resource": {"buffer": size_buffer, "offset": 0, "size": 4}},
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return param_out


def update_model_with_optimizer(model_params, gradients, opt_state, lr):
    """
    Update all model parameters using AdamW optimizer.

    Args:
        model_params: GPUModelParams - current model
        gradients: GPUModelParams - computed gradients
        opt_state: GPUOptimizerState - optimizer state
        lr: float - learning rate

    Returns:
        (new_model_params, new_opt_state)
    """
    device = model_params.embedding.device
    step = opt_state.step + 1

    # Update embedding
    new_embedding = adamw_update(
        model_params.embedding,
        gradients.embedding,
        opt_state.m_embedding,
        opt_state.v_embedding,
        step,
        lr,
    )

    # Update each layer
    new_layers = []
    for layer_param, layer_grad, m_layer, v_layer in zip(
        model_params.layers, gradients.layers, opt_state.m_layers, opt_state.v_layers
    ):
        new_layer = GPULayerParams(
            attn_wq=adamw_update(
                layer_param.attn_wq,
                layer_grad.attn_wq,
                m_layer.attn_wq,
                v_layer.attn_wq,
                step,
                lr,
            ),
            attn_wk=adamw_update(
                layer_param.attn_wk,
                layer_grad.attn_wk,
                m_layer.attn_wk,
                v_layer.attn_wk,
                step,
                lr,
            ),
            attn_wv=adamw_update(
                layer_param.attn_wv,
                layer_grad.attn_wv,
                m_layer.attn_wv,
                v_layer.attn_wv,
                step,
                lr,
            ),
            attn_wo=adamw_update(
                layer_param.attn_wo,
                layer_grad.attn_wo,
                m_layer.attn_wo,
                v_layer.attn_wo,
                step,
                lr,
            ),
            ff_w1=adamw_update(
                layer_param.ff_w1,
                layer_grad.ff_w1,
                m_layer.ff_w1,
                v_layer.ff_w1,
                step,
                lr,
            ),
            ff_b1=adamw_update(
                layer_param.ff_b1,
                layer_grad.ff_b1,
                m_layer.ff_b1,
                v_layer.ff_b1,
                step,
                lr,
            ),
            ff_w2=adamw_update(
                layer_param.ff_w2,
                layer_grad.ff_w2,
                m_layer.ff_w2,
                v_layer.ff_w2,
                step,
                lr,
            ),
            ff_b2=adamw_update(
                layer_param.ff_b2,
                layer_grad.ff_b2,
                m_layer.ff_b2,
                v_layer.ff_b2,
                step,
                lr,
            ),
            ln_gamma1=adamw_update(
                layer_param.ln_gamma1,
                layer_grad.ln_gamma1,
                m_layer.ln_gamma1,
                v_layer.ln_gamma1,
                step,
                lr,
            ),
            ln_beta1=adamw_update(
                layer_param.ln_beta1,
                layer_grad.ln_beta1,
                m_layer.ln_beta1,
                v_layer.ln_beta1,
                step,
                lr,
            ),
            ln_gamma2=adamw_update(
                layer_param.ln_gamma2,
                layer_grad.ln_gamma2,
                m_layer.ln_gamma2,
                v_layer.ln_gamma2,
                step,
                lr,
            ),
            ln_beta2=adamw_update(
                layer_param.ln_beta2,
                layer_grad.ln_beta2,
                m_layer.ln_beta2,
                v_layer.ln_beta2,
                step,
                lr,
            ),
        )
        new_layers.append(new_layer)

    new_model = GPUModelParams(
        embedding=new_embedding,
        pos_encoding=model_params.pos_encoding,  # Not trainable
        layers=new_layers,
    )

    new_opt_state = GPUOptimizerState(
        m_embedding=opt_state.m_embedding,
        v_embedding=opt_state.v_embedding,
        m_layers=opt_state.m_layers,
        v_layers=opt_state.v_layers,
        step=step,
    )

    return new_model, new_opt_state


# ============================================================================
# COMPLETE TRAINING STEP
# ============================================================================


@dataclasses.dataclass
class ForwardCache:
    """Cache of intermediate values needed for backward pass"""

    # Input
    input_embedded: GPUBuffer

    # Per-layer caches
    layer_caches: list  # List of dicts with layer-specific activations


def forward_pass_with_cache(model_params, input_tokens, n_heads):
    """
    Forward pass that caches all values needed for backward pass.

    Args:
        model_params: GPUModelParams
        input_tokens: np.ndarray of shape (batch_size, seq_len)
        n_heads: number of attention heads

    Returns:
        (logits_buffer, cache)
    """
    device = model_params.embedding.device
    batch_size, seq_len = input_tokens.shape
    embedding_dim = model_params.embedding.shape[1]

    # TODO: Implement full forward pass with caching
    # This is a simplified version - you'll need to implement:
    # 1. Embedding lookup
    # 2. Add positional encoding
    # 3. Each transformer layer (attention + FF)
    # 4. Output projection

    # For now, return placeholder
    logits = create_gpu_buffer(
        (batch_size, seq_len, model_params.embedding.shape[0]), device=device
    )
    cache = ForwardCache(
        input_embedded=create_gpu_buffer(
            (batch_size, seq_len, embedding_dim), device=device
        ),
        layer_caches=[],
    )

    return logits, cache


def backward_pass(model_params, cache, grad_logits):
    """
    Backward pass that computes all gradients.

    Args:
        model_params: GPUModelParams
        cache: ForwardCache from forward pass
        grad_logits: GPUBuffer - gradient of loss w.r.t. logits

    Returns:
        gradients: GPUModelParams structure containing all gradients
    """
    # TODO: Implement full backward pass
    # This will chain together all the backward shader functions

    pass


def train_step_pure_gpu(
    model_params, opt_state, batch_inputs, batch_targets, n_heads, lr
):
    """
    Complete training step on pure GPU.

    Args:
        model_params: GPUModelParams
        opt_state: GPUOptimizerState
        batch_inputs: np.ndarray (batch_size, seq_len)
        batch_targets: np.ndarray (batch_size, seq_len)
        n_heads: int
        lr: float

    Returns:
        (new_model_params, new_opt_state, loss_value)
    """
    # 1. Forward pass with caching
    logits, cache = forward_pass_with_cache(model_params, batch_inputs, n_heads)

    # 2. Compute loss and initial gradient
    loss_value, grad_logits = compute_loss_and_grad(logits, batch_targets)

    # 3. Backward pass to compute all gradients
    gradients = backward_pass(model_params, cache, grad_logits, n_heads)

    # 4. Optimizer step
    new_model_params, new_opt_state = update_model_with_optimizer(
        model_params, gradients, opt_state, lr
    )

    return new_model_params, new_opt_state, loss_value


CROSS_ENTROPY_LOSS_SHADER = """
// Compute cross-entropy loss and initial gradient

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> loss_output: array<f32>;  // Array to accumulate losses per thread
@group(0) @binding(3) var<storage, read_write> grad_logits: array<f32>;
@group(0) @binding(4) var<uniform> params: vec3<u32>; // [batch_size, seq_len, vocab_size]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_size = params.x;
    let seq_len = params.y;
    let vocab_size = params.z;
    let total_predictions = batch_size * seq_len;

    let pred_idx = global_id.x;

    if (pred_idx >= total_predictions) {
        return;
    }

    let target_idx = targets[pred_idx];
    let logit_offset = pred_idx * vocab_size;

    // Find max for numerical stability
    var max_logit = logits[logit_offset];
    for (var i = 1u; i < vocab_size; i++) {
        max_logit = max(max_logit, logits[logit_offset + i]);
    }

    // Compute softmax and loss
    var sum_exp = 0.0;
    for (var i = 0u; i < vocab_size; i++) {
        sum_exp += exp(logits[logit_offset + i] - max_logit);
    }

    let log_sum_exp = log(sum_exp);
    let target_logit = logits[logit_offset + target_idx];
    let loss = -(target_logit - max_logit - log_sum_exp);

    // Store individual loss (will sum on CPU)
    loss_output[pred_idx] = loss;

    // Compute gradients: softmax(logits) - one_hot(target)
    let normalization = 1.0 / f32(total_predictions);
    for (var i = 0u; i < vocab_size; i++) {
        let prob = exp(logits[logit_offset + i] - max_logit) / sum_exp;
        var grad = prob;
        if (i == target_idx) {
            grad -= 1.0;
        }
        grad_logits[logit_offset + i] = grad * normalization;
    }
}
"""


def compute_loss_and_grad(logits, targets):
    """
    Compute cross-entropy loss and initial gradient on GPU.

    Args:
        logits: GPUBuffer of shape (batch_size, seq_len, vocab_size)
        targets: np.ndarray of shape (batch_size, seq_len) with target token IDs

    Returns:
        (loss_value, grad_logits_buffer)
    """
    device = logits.device
    batch_size, seq_len, vocab_size = logits.shape
    total_predictions = batch_size * seq_len

    # Upload targets to GPU
    targets_flat = targets.flatten().astype(np.uint32)
    targets_buffer = device.create_buffer_with_data(
        data=targets_flat, usage=wgpu.BufferUsage.STORAGE
    )

    # Create loss output buffer (one value per prediction for summing on CPU)
    loss_buffer = device.create_buffer(
        size=total_predictions * 4,  # CHANGED: array of losses, not single value
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    # Create gradient output buffer
    grad_logits = create_gpu_buffer(logits.shape, device=device)

    # Create params buffer
    params_data = np.array([batch_size, seq_len, vocab_size], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params_data, usage=wgpu.BufferUsage.UNIFORM
    )

    # Create shader and pipeline
    shader_module = device.create_shader_module(code=CROSS_ENTROPY_LOSS_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": logits.buffer,
                    "offset": 0,
                    "size": logits.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": targets_buffer,
                    "offset": 0,
                    "size": targets_flat.nbytes,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": loss_buffer,
                    "offset": 0,
                    "size": total_predictions * 4,
                },
            },  # CHANGED
            {
                "binding": 3,
                "resource": {
                    "buffer": grad_logits.buffer,
                    "offset": 0,
                    "size": grad_logits.size * 4,
                },
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params_data.nbytes,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(batch_size * seq_len / 256), 1, 1)
    compute_pass.end()

    # Read loss values
    loss_read_buffer = device.create_buffer(
        size=total_predictions * 4,  # CHANGED
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder.copy_buffer_to_buffer(
        loss_buffer, 0, loss_read_buffer, 0, total_predictions * 4
    )  # CHANGED

    device.queue.submit([encoder.finish()])

    loss_read_buffer.map_sync(wgpu.MapMode.READ)
    loss_data = np.frombuffer(loss_read_buffer.read_mapped(), dtype=np.float32).copy()
    loss_read_buffer.unmap()

    # Sum losses on CPU
    loss_value = np.sum(loss_data) / total_predictions

    return loss_value, grad_logits


@dataclasses.dataclass
class LayerCache:
    """Cache for one transformer layer"""

    # Pre-attention
    x_pre_attn: GPUBuffer
    ln1_output: GPUBuffer

    # Attention
    q: GPUBuffer
    k: GPUBuffer
    v: GPUBuffer
    attn_scores: GPUBuffer  # Before softmax
    attn_weights: GPUBuffer  # After softmax
    attn_output: GPUBuffer

    # Pre-feedforward
    x_pre_ff: GPUBuffer
    ln2_output: GPUBuffer

    # Feedforward
    ff_hidden: GPUBuffer  # Before GELU
    ff_output: GPUBuffer


def forward_pass_with_cache(model_params, input_tokens, n_heads):
    """
    Complete forward pass with caching for backward pass.

    Args:
        model_params: GPUModelParams
        input_tokens: np.ndarray of shape (batch_size, seq_len)
        n_heads: number of attention heads

    Returns:
        (logits_buffer, cache)
    """
    device = model_params.embedding.device
    batch_size, seq_len = input_tokens.shape
    embedding_dim = model_params.embedding.shape[1]
    head_dim = embedding_dim // n_heads

    # 1. Embedding lookup
    input_tokens_flat = input_tokens.flatten().astype(np.uint32)
    x = embedding_lookup_gpu(model_params.embedding, input_tokens_flat)
    x = reshape_buffer(x, (batch_size, seq_len, embedding_dim))

    # 2. Add positional encoding
    # Broadcast pos_encoding across batch dimension
    pos_encoding_broadcasted = create_gpu_buffer(
        (batch_size, seq_len, embedding_dim),
        np.tile(gpu_to_numpy(model_params.pos_encoding), (batch_size, 1, 1)),
        device,
    )
    x = add_buffers(x, pos_encoding_broadcasted)

    input_embedded = x
    layer_caches = []

    # 3. Process each transformer layer
    for layer_idx, layer_params in enumerate(model_params.layers):
        # Pre-attention LayerNorm
        x_pre_attn = x
        ln1_output = layer_norm_gpu(x, layer_params.ln_gamma1, layer_params.ln_beta1)

        # Multi-head attention projections
        ln1_2d = reshape_buffer(ln1_output, (batch_size * seq_len, embedding_dim))

        q_2d = matmul_gpu(ln1_2d, layer_params.attn_wq)
        k_2d = matmul_gpu(ln1_2d, layer_params.attn_wk)
        v_2d = matmul_gpu(ln1_2d, layer_params.attn_wv)

        q = reshape_buffer(q_2d, (batch_size, seq_len, embedding_dim))
        k = reshape_buffer(k_2d, (batch_size, seq_len, embedding_dim))
        v = reshape_buffer(v_2d, (batch_size, seq_len, embedding_dim))

        # Split heads
        q = split_heads_gpu(q, n_heads, head_dim)
        k = split_heads_gpu(k, n_heads, head_dim)
        v = split_heads_gpu(v, n_heads, head_dim)

        # Scaled dot-product attention
        k_transposed = transpose_last_two_dims(k)
        attn_scores = matmul_batched_gpu(q, k_transposed)

        # Scale
        attn_scores = scale_buffer(attn_scores, 1.0 / math.sqrt(head_dim))

        # Add causal mask
        attn_scores = add_causal_mask_gpu(attn_scores, seq_len)

        # Softmax
        attn_weights = softmax_gpu(attn_scores, axis=-1)

        # Attention @ V
        attn_values = matmul_batched_gpu(attn_weights, v)

        # Combine heads
        attn_combined = combine_heads_gpu(attn_values)

        # Output projection
        attn_combined_2d = reshape_buffer(
            attn_combined, (batch_size * seq_len, embedding_dim)
        )
        attn_output_2d = matmul_gpu(attn_combined_2d, layer_params.attn_wo)
        attn_output = reshape_buffer(
            attn_output_2d, (batch_size, seq_len, embedding_dim)
        )

        # Residual connection
        x = add_buffers(x_pre_attn, attn_output)

        # Pre-feedforward LayerNorm
        x_pre_ff = x
        ln2_output = layer_norm_gpu(x, layer_params.ln_gamma2, layer_params.ln_beta2)

        # Feedforward network
        ln2_2d = reshape_buffer(ln2_output, (batch_size * seq_len, embedding_dim))

        ff_hidden_2d = matmul_gpu(ln2_2d, layer_params.ff_w1)
        ff_hidden_2d = add_bias_gpu(ff_hidden_2d, layer_params.ff_b1)
        ff_hidden = reshape_buffer(
            ff_hidden_2d, (batch_size, seq_len, 4 * embedding_dim)
        )

        ff_activated = gelu_gpu(ff_hidden)

        ff_activated_2d = reshape_buffer(
            ff_activated, (batch_size * seq_len, 4 * embedding_dim)
        )
        ff_output_2d = matmul_gpu(ff_activated_2d, layer_params.ff_w2)
        ff_output_2d = add_bias_gpu(ff_output_2d, layer_params.ff_b2)
        ff_output = reshape_buffer(ff_output_2d, (batch_size, seq_len, embedding_dim))

        # Residual connection
        x = add_buffers(x_pre_ff, ff_output)

        # Cache layer activations
        layer_cache = LayerCache(
            x_pre_attn=x_pre_attn,
            ln1_output=ln1_output,
            q=q,
            k=k,
            v=v,
            attn_scores=attn_scores,
            attn_weights=attn_weights,
            attn_output=attn_output,
            x_pre_ff=x_pre_ff,
            ln2_output=ln2_output,
            ff_hidden=ff_hidden,
            ff_output=ff_output,
        )
        layer_caches.append(layer_cache)

    # 4. Output projection (logits = x @ embedding^T)
    x_2d = reshape_buffer(x, (batch_size * seq_len, embedding_dim))
    embedding_transposed = transpose_buffer(model_params.embedding)
    logits_2d = matmul_gpu(x_2d, embedding_transposed)
    logits = reshape_buffer(
        logits_2d, (batch_size, seq_len, model_params.embedding.shape[0])
    )

    cache = ForwardCache(input_embedded=input_embedded, layer_caches=layer_caches)

    return logits, cache


def reshape_buffer(buffer, new_shape):
    """Reshape GPU buffer (just metadata change, no data copy)"""
    assert int(np.prod(buffer.shape)) == int(np.prod(new_shape))
    return GPUBuffer(
        buffer=buffer.buffer, shape=new_shape, size=buffer.size, device=buffer.device
    )


def add_buffers(a, b):
    """Element-wise addition on GPU (with broadcasting support)"""
    # TODO: Implement element-wise add shader
    # For now, simplified version
    return a


def scale_buffer(buffer, scale):
    """Multiply all elements by scalar"""
    # TODO: Implement scalar multiply shader
    return buffer


def add_causal_mask_gpu(attn_scores, seq_len):
    """Add causal mask to attention scores"""
    # TODO: Use the causal mask shader we defined earlier
    return attn_scores


def transpose_buffer(buffer):
    """Transpose 2D buffer"""
    assert len(buffer.shape) == 2
    M, N = buffer.shape
    return GPUBuffer(
        buffer=buffer.buffer, shape=(N, M), size=buffer.size, device=buffer.device
    )


def transpose_last_two_dims(buffer):
    """Transpose last two dimensions"""
    # TODO: Implement transpose shader
    return buffer


def split_heads_gpu(x, n_heads, head_dim):
    """Split embedding dimension into multiple heads"""
    batch_size, seq_len, embedding_dim = x.shape
    # Reshape: (batch, seq, dim) -> (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
    # TODO: Implement reshape/transpose shader
    return reshape_buffer(x, (batch_size, n_heads, seq_len, head_dim))


def combine_heads_gpu(x):
    """Combine multiple heads back into single embedding"""
    batch_size, n_heads, seq_len, head_dim = x.shape
    # TODO: Implement reshape/transpose shader
    return reshape_buffer(x, (batch_size, seq_len, n_heads * head_dim))


def add_bias_gpu(x, bias):
    """Add bias vector to matrix"""
    # TODO: Implement bias add shader
    return x


# ============================================================================
# UTILITY SHADERS
# ============================================================================

ELEMENT_WISE_ADD_SHADER = """
// Element-wise addition with broadcasting support

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // [size, a_stride, b_stride, broadcast_b]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let size = params.x;
    let a_stride = params.y;
    let b_stride = params.z;
    let broadcast_b = params.w;

    if (idx >= size) {
        return;
    }

    let a_idx = idx / a_stride * a_stride + (idx % a_stride);
    var b_idx = idx / b_stride * b_stride + (idx % b_stride);

    // If broadcasting, compute correct b index
    if (broadcast_b == 1u) {
        b_idx = idx % b_stride;
    }

    output[idx] = a[a_idx] + b[b_idx];
}
"""

SCALAR_MULTIPLY_SHADER = """
// Multiply all elements by a scalar

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> scalar: f32;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    output[idx] = input[idx] * scalar;
}
"""

TRANSPOSE_2D_SHADER = """
// Transpose a 2D matrix

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> shape: vec2<u32>; // [rows, cols]

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let rows = shape.x;
    let cols = shape.y;

    if (row >= rows || col >= cols) {
        return;
    }

    let in_idx = row * cols + col;
    let out_idx = col * rows + row;

    output[out_idx] = input[in_idx];
}
"""

TRANSPOSE_LAST_TWO_DIMS_SHADER = """
// Transpose last two dimensions of 4D tensor
// Shape: (batch, heads, seq1, seq2) -> (batch, heads, seq2, seq1)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> shape: vec4<u32>; // [dim0, dim1, dim2, dim3]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = shape.x * shape.y * shape.z * shape.w;

    if (idx >= total_size) {
        return;
    }

    let dim0 = shape.x;
    let dim1 = shape.y;
    let dim2 = shape.z;
    let dim3 = shape.w;

    // Decode input indices
    let i0 = idx / (dim1 * dim2 * dim3);
    let rem1 = idx % (dim1 * dim2 * dim3);
    let i1 = rem1 / (dim2 * dim3);
    let rem2 = rem1 % (dim2 * dim3);
    let i2 = rem2 / dim3;
    let i3 = rem2 % dim3;

    // Swap last two dimensions
    let out_idx = i0 * (dim1 * dim3 * dim2) + i1 * (dim3 * dim2) + i3 * dim2 + i2;

    output[out_idx] = input[idx];
}
"""

ADD_BIAS_SHADER = """
// Add bias vector to each row of matrix

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> shape: vec2<u32>; // [num_rows, num_cols]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_rows = shape.x;
    let num_cols = shape.y;
    let total_size = num_rows * num_cols;

    if (idx >= total_size) {
        return;
    }

    let col = idx % num_cols;
    output[idx] = input[idx] + bias[col];
}
"""

ADD_CAUSAL_MASK_SHADER = """
// Add causal mask to attention scores
// Adds -1e9 to upper triangle

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> shape: vec4<u32>; // [batch, heads, seq_len, seq_len]

const MASK_VALUE: f32 = -1e9;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_head_idx = global_id.z;
    let row = global_id.y;
    let col = global_id.x;

    let batch = shape.x;
    let heads = shape.y;
    let seq_len = shape.z;

    if (batch_head_idx >= batch * heads || row >= seq_len || col >= seq_len) {
        return;
    }

    let idx = batch_head_idx * seq_len * seq_len + row * seq_len + col;

    var value = input[idx];
    if (col > row) {
        value += MASK_VALUE;
    }

    output[idx] = value;
}
"""

SPLIT_HEADS_SHADER = """
// Split: (batch, seq, dim) -> (batch, heads, seq, head_dim)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // [batch, seq, heads, head_dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let batch = params.x;
    let seq = params.y;
    let heads = params.z;
    let head_dim = params.w;

    let total_size = batch * seq * heads * head_dim;

    if (idx >= total_size) {
        return;
    }

    // Input layout: [batch][seq][heads * head_dim]
    // Output layout: [batch][heads][seq][head_dim]

    let b = idx / (seq * heads * head_dim);
    let rem1 = idx % (seq * heads * head_dim);
    let s = rem1 / (heads * head_dim);
    let rem2 = rem1 % (heads * head_dim);
    let h = rem2 / head_dim;
    let d = rem2 % head_dim;

    let in_idx = b * (seq * heads * head_dim) + s * (heads * head_dim) + h * head_dim + d;
    let out_idx = b * (heads * seq * head_dim) + h * (seq * head_dim) + s * head_dim + d;

    output[out_idx] = input[in_idx];
}
"""

COMBINE_HEADS_SHADER = """
// Combine: (batch, heads, seq, head_dim) -> (batch, seq, heads * head_dim)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // [batch, heads, seq, head_dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let batch = params.x;
    let heads = params.y;
    let seq = params.z;
    let head_dim = params.w;

    let total_size = batch * heads * seq * head_dim;

    if (idx >= total_size) {
        return;
    }

    // Input layout: [batch][heads][seq][head_dim]
    // Output layout: [batch][seq][heads * head_dim]

    let b = idx / (heads * seq * head_dim);
    let rem1 = idx % (heads * seq * head_dim);
    let h = rem1 / (seq * head_dim);
    let rem2 = rem1 % (seq * head_dim);
    let s = rem2 / head_dim;
    let d = rem2 % head_dim;

    let out_idx = b * (seq * heads * head_dim) + s * (heads * head_dim) + h * head_dim + d;

    output[out_idx] = input[idx];
}
"""

# ============================================================================
# UTILITY GPU OPERATIONS
# ============================================================================


def add_buffers(a, b, broadcast_b=False):
    """
    Element-wise addition with optional broadcasting.

    Args:
        a: GPUBuffer - first operand
        b: GPUBuffer - second operand
        broadcast_b: bool - if True, broadcast b along first dimensions

    Returns:
        GPUBuffer - result
    """
    device = a.device

    # Determine output size
    output_size = a.size
    output_shape = a.shape

    # Create output buffer
    output = create_gpu_buffer(output_shape, device=device)

    # Calculate strides
    a_stride = 1
    b_stride = 1 if not broadcast_b else b.size

    # Create params buffer
    params = np.array(
        [output_size, a_stride, b_stride, 1 if broadcast_b else 0], dtype=np.uint32
    )
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    # Create shader and pipeline
    shader_module = device.create_shader_module(code=ELEMENT_WISE_ADD_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": a.buffer, "offset": 0, "size": a.size * 4},
            },
            {
                "binding": 1,
                "resource": {"buffer": b.buffer, "offset": 0, "size": b.size * 4},
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(output_size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def scale_buffer(buffer, scalar):
    """Multiply all elements by scalar"""
    device = buffer.device
    output = create_gpu_buffer(buffer.shape, device=device)

    scalar_buffer = device.create_buffer_with_data(
        data=np.array([scalar], dtype=np.float32), usage=wgpu.BufferUsage.UNIFORM
    )

    size_buffer = device.create_buffer_with_data(
        data=np.array([buffer.size], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    shader_module = device.create_shader_module(code=SCALAR_MULTIPLY_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": buffer.buffer,
                    "offset": 0,
                    "size": buffer.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": scalar_buffer, "offset": 0, "size": 4},
            },
            {"binding": 3, "resource": {"buffer": size_buffer, "offset": 0, "size": 4}},
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(buffer.size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def transpose_buffer(buffer):
    """Transpose 2D buffer"""
    assert len(buffer.shape) == 2, "Only 2D transpose supported"
    device = buffer.device
    rows, cols = buffer.shape
    output = create_gpu_buffer((cols, rows), device=device)

    shape_buffer = device.create_buffer_with_data(
        data=np.array([rows, cols], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    shader_module = device.create_shader_module(code=TRANSPOSE_2D_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": buffer.buffer,
                    "offset": 0,
                    "size": buffer.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": shape_buffer, "offset": 0, "size": 8},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(cols / 16), math.ceil(rows / 16), 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def transpose_last_two_dims(buffer):
    """Transpose last two dimensions of 4D tensor"""
    assert len(buffer.shape) == 4
    device = buffer.device
    d0, d1, d2, d3 = buffer.shape
    output = create_gpu_buffer((d0, d1, d3, d2), device=device)

    shape_buffer = device.create_buffer_with_data(
        data=np.array([d0, d1, d2, d3], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    shader_module = device.create_shader_module(code=TRANSPOSE_LAST_TWO_DIMS_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": buffer.buffer,
                    "offset": 0,
                    "size": buffer.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": shape_buffer, "offset": 0, "size": 16},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(buffer.size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def add_bias_gpu(x, bias):
    """Add bias vector to each row of matrix"""
    device = x.device
    # x is 2D: (num_rows, num_cols)
    # bias is 1D: (num_cols,)

    assert len(x.shape) == 2
    assert len(bias.shape) == 1
    assert x.shape[1] == bias.shape[0]

    output = create_gpu_buffer(x.shape, device=device)

    shape_buffer = device.create_buffer_with_data(
        data=np.array(x.shape, dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    shader_module = device.create_shader_module(code=ADD_BIAS_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": x.buffer, "offset": 0, "size": x.size * 4},
            },
            {
                "binding": 1,
                "resource": {"buffer": bias.buffer, "offset": 0, "size": bias.size * 4},
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {"buffer": shape_buffer, "offset": 0, "size": 8},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(x.size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def add_causal_mask_gpu(attn_scores, seq_len):
    """Add causal mask to attention scores"""
    device = attn_scores.device
    # attn_scores shape: (batch, heads, seq_len, seq_len)
    assert len(attn_scores.shape) == 4

    output = create_gpu_buffer(attn_scores.shape, device=device)

    shape_buffer = device.create_buffer_with_data(
        data=np.array(attn_scores.shape, dtype=np.uint32),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader_module = device.create_shader_module(code=ADD_CAUSAL_MASK_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    batch, heads, seq_len, _ = attn_scores.shape

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": attn_scores.buffer,
                    "offset": 0,
                    "size": attn_scores.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": shape_buffer, "offset": 0, "size": 16},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(
        math.ceil(seq_len / 16), math.ceil(seq_len / 16), batch * heads
    )
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def split_heads_gpu(x, n_heads, head_dim):
    """Split embedding dimension into multiple heads"""
    device = x.device
    batch_size, seq_len, embedding_dim = x.shape
    assert embedding_dim == n_heads * head_dim

    output = create_gpu_buffer((batch_size, n_heads, seq_len, head_dim), device=device)

    params_buffer = device.create_buffer_with_data(
        data=np.array([batch_size, seq_len, n_heads, head_dim], dtype=np.uint32),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader_module = device.create_shader_module(code=SPLIT_HEADS_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": x.buffer, "offset": 0, "size": x.size * 4},
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": params_buffer, "offset": 0, "size": 16},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(x.size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


def combine_heads_gpu(x):
    """Combine multiple heads back into single embedding"""
    device = x.device
    batch_size, n_heads, seq_len, head_dim = x.shape

    output = create_gpu_buffer((batch_size, seq_len, n_heads * head_dim), device=device)

    params_buffer = device.create_buffer_with_data(
        data=np.array([batch_size, n_heads, seq_len, head_dim], dtype=np.uint32),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader_module = device.create_shader_module(code=COMBINE_HEADS_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": x.buffer, "offset": 0, "size": x.size * 4},
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": params_buffer, "offset": 0, "size": 16},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(x.size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return output


# ============================================================================
# BACKWARD PASS - COMPLETE IMPLEMENTATION
# ============================================================================


def backward_pass(model_params, cache, grad_logits, n_heads):
    """
    Complete backward pass that computes all gradients.

    Args:
        model_params: GPUModelParams
        cache: ForwardCache from forward pass
        grad_logits: GPUBuffer - gradient of loss w.r.t. logits
        n_heads: number of attention heads

    Returns:
        gradients: GPUModelParams structure containing all gradients
    """
    device = model_params.embedding.device
    embedding_dim = model_params.embedding.shape[1]
    head_dim = embedding_dim // n_heads

    # Initialize gradient structure
    grad_layers = []
    for _ in model_params.layers:
        grad_layer = create_zero_layer_grads(embedding_dim, device)
        grad_layers.append(grad_layer)

    grad_embedding = create_gpu_buffer(model_params.embedding.shape, device=device)

    # Get final layer output
    final_layer_cache = cache.layer_caches[-1]
    x_final = add_buffers(final_layer_cache.x_pre_ff, final_layer_cache.ff_output)

    # 1. Backprop through output projection: logits = x @ embedding^T
    batch_size, seq_len, vocab_size = grad_logits.shape

    grad_logits_2d = reshape_buffer(grad_logits, (batch_size * seq_len, vocab_size))
    x_final_2d = reshape_buffer(x_final, (batch_size * seq_len, embedding_dim))

    # grad_embedding = x_final^T @ grad_logits
    grad_embedding_2d = matmul_gpu(transpose_buffer(x_final_2d), grad_logits_2d)
    numpy_to_gpu(grad_embedding, gpu_to_numpy(grad_embedding_2d))

    # grad_x = grad_logits @ embedding
    grad_x_2d = matmul_gpu(grad_logits_2d, model_params.embedding)
    grad_x = reshape_buffer(grad_x_2d, (batch_size, seq_len, embedding_dim))

    # 2. Backprop through each transformer layer (in reverse)
    for layer_idx in reversed(range(len(model_params.layers))):
        layer_params = model_params.layers[layer_idx]
        layer_cache = cache.layer_caches[layer_idx]
        grad_layer = grad_layers[layer_idx]

        # FEEDFORWARD BACKWARD
        grad_ff_output = grad_x
        grad_x_pre_ff = grad_x

        ff_activated_2d = reshape_buffer(
            layer_cache.ff_hidden, (batch_size * seq_len, 4 * embedding_dim)
        )
        grad_ff_output_2d = reshape_buffer(
            grad_ff_output, (batch_size * seq_len, embedding_dim)
        )

        # grad_ff_w2 = ff_activated^T @ grad_ff_output
        grad_ff_w2 = matmul_gpu(transpose_buffer(ff_activated_2d), grad_ff_output_2d)
        numpy_to_gpu(grad_layer.ff_w2, gpu_to_numpy(grad_ff_w2))

        # grad_ff_b2 = sum over batch/seq
        grad_ff_b2_np = np.sum(gpu_to_numpy(grad_ff_output_2d), axis=0)
        numpy_to_gpu(grad_layer.ff_b2, grad_ff_b2_np)

        # grad_ff_activated = grad_ff_output @ ff_w2^T
        grad_ff_activated_2d = matmul_gpu(
            grad_ff_output_2d, transpose_buffer(layer_params.ff_w2)
        )
        grad_ff_activated = reshape_buffer(
            grad_ff_activated_2d, (batch_size, seq_len, 4 * embedding_dim)
        )

        # Backprop through GELU
        grad_ff_hidden = gelu_backward_gpu(layer_cache.ff_hidden, grad_ff_activated)

        # Backprop through FF input projection
        ln2_output_2d = reshape_buffer(
            layer_cache.ln2_output, (batch_size * seq_len, embedding_dim)
        )
        grad_ff_hidden_2d = reshape_buffer(
            grad_ff_hidden, (batch_size * seq_len, 4 * embedding_dim)
        )

        # grad_ff_w1 = ln2_output^T @ grad_ff_hidden
        grad_ff_w1 = matmul_gpu(transpose_buffer(ln2_output_2d), grad_ff_hidden_2d)
        numpy_to_gpu(grad_layer.ff_w1, gpu_to_numpy(grad_ff_w1))

        # grad_ff_b1 = sum
        grad_ff_b1_np = np.sum(gpu_to_numpy(grad_ff_hidden_2d), axis=0)
        numpy_to_gpu(grad_layer.ff_b1, grad_ff_b1_np)

        # grad_ln2_output = grad_ff_hidden @ ff_w1^T
        grad_ln2_output_2d = matmul_gpu(
            grad_ff_hidden_2d, transpose_buffer(layer_params.ff_w1)
        )
        grad_ln2_output = reshape_buffer(
            grad_ln2_output_2d, (batch_size, seq_len, embedding_dim)
        )

        # Backprop through LayerNorm 2
        grad_x_pre_ff_from_ln2, grad_ln_gamma2, grad_ln_beta2 = layer_norm_backward_gpu(
            layer_cache.x_pre_ff, layer_params.ln_gamma2, grad_ln2_output
        )

        numpy_to_gpu(grad_layer.ln_gamma2, gpu_to_numpy(grad_ln_gamma2))
        numpy_to_gpu(grad_layer.ln_beta2, gpu_to_numpy(grad_ln_beta2))

        # Add residual gradient
        grad_x_pre_attn = add_buffers(grad_x_pre_ff, grad_x_pre_ff_from_ln2)

        # ATTENTION BACKWARD
        grad_attn_output = grad_x_pre_attn
        grad_x_pre_attn_residual = grad_x_pre_attn

        attn_combined_2d = reshape_buffer(
            layer_cache.attn_output, (batch_size * seq_len, embedding_dim)
        )
        grad_attn_output_2d = reshape_buffer(
            grad_attn_output, (batch_size * seq_len, embedding_dim)
        )

        # grad_attn_wo = attn_combined^T @ grad_attn_output
        grad_attn_wo = matmul_gpu(
            transpose_buffer(attn_combined_2d), grad_attn_output_2d
        )
        numpy_to_gpu(grad_layer.attn_wo, gpu_to_numpy(grad_attn_wo))

        # grad_attn_combined = grad_attn_output @ attn_wo^T
        grad_attn_combined_2d = matmul_gpu(
            grad_attn_output_2d, transpose_buffer(layer_params.attn_wo)
        )
        grad_attn_combined = reshape_buffer(
            grad_attn_combined_2d, (batch_size, seq_len, embedding_dim)
        )

        # Backprop through combine_heads
        grad_attn_values = split_heads_gpu(grad_attn_combined, n_heads, head_dim)

        # Backprop through attention @ V
        v = layer_cache.v
        attn_weights = layer_cache.attn_weights

        grad_attn_weights = matmul_batched_gpu(
            grad_attn_values, transpose_last_two_dims(v)
        )
        grad_v = matmul_batched_gpu(
            transpose_last_two_dims(attn_weights), grad_attn_values
        )

        # Backprop through softmax
        grad_attn_scores = softmax_backward_gpu(attn_weights, grad_attn_weights)

        # Backprop through scale
        grad_attn_scores = scale_buffer(grad_attn_scores, 1.0 / math.sqrt(head_dim))

        # Backprop through Q @ K^T
        q = layer_cache.q
        k = layer_cache.k

        grad_q = matmul_batched_gpu(grad_attn_scores, k)
        grad_k = matmul_batched_gpu(transpose_last_two_dims(grad_attn_scores), q)

        # Backprop through split_heads
        grad_q_combined = combine_heads_gpu(grad_q)
        grad_k_combined = combine_heads_gpu(grad_k)
        grad_v_combined = combine_heads_gpu(grad_v)

        # Backprop through Q, K, V projections
        ln1_output_2d = reshape_buffer(
            layer_cache.ln1_output, (batch_size * seq_len, embedding_dim)
        )

        grad_q_2d = reshape_buffer(
            grad_q_combined, (batch_size * seq_len, embedding_dim)
        )
        grad_k_2d = reshape_buffer(
            grad_k_combined, (batch_size * seq_len, embedding_dim)
        )
        grad_v_2d = reshape_buffer(
            grad_v_combined, (batch_size * seq_len, embedding_dim)
        )

        # Compute gradients for weight matrices
        grad_attn_wq = matmul_gpu(transpose_buffer(ln1_output_2d), grad_q_2d)
        numpy_to_gpu(grad_layer.attn_wq, gpu_to_numpy(grad_attn_wq))

        grad_attn_wk = matmul_gpu(transpose_buffer(ln1_output_2d), grad_k_2d)
        numpy_to_gpu(grad_layer.attn_wk, gpu_to_numpy(grad_attn_wk))

        grad_attn_wv = matmul_gpu(transpose_buffer(ln1_output_2d), grad_v_2d)
        numpy_to_gpu(grad_layer.attn_wv, gpu_to_numpy(grad_attn_wv))

        # grad_ln1_output
        grad_ln1_from_q = matmul_gpu(grad_q_2d, transpose_buffer(layer_params.attn_wq))
        grad_ln1_from_k = matmul_gpu(grad_k_2d, transpose_buffer(layer_params.attn_wk))
        grad_ln1_from_v = matmul_gpu(grad_v_2d, transpose_buffer(layer_params.attn_wv))

        grad_ln1_output_2d = add_buffers(grad_ln1_from_q, grad_ln1_from_k)
        grad_ln1_output_2d = add_buffers(grad_ln1_output_2d, grad_ln1_from_v)
        grad_ln1_output = reshape_buffer(
            grad_ln1_output_2d, (batch_size, seq_len, embedding_dim)
        )

        # Backprop through LayerNorm 1
        grad_x_from_ln1, grad_ln_gamma1, grad_ln_beta1 = layer_norm_backward_gpu(
            layer_cache.x_pre_attn, layer_params.ln_gamma1, grad_ln1_output
        )

        numpy_to_gpu(grad_layer.ln_gamma1, gpu_to_numpy(grad_ln_gamma1))
        numpy_to_gpu(grad_layer.ln_beta1, gpu_to_numpy(grad_ln_beta1))

        # Add residual gradient
        grad_x = add_buffers(grad_x_pre_attn_residual, grad_x_from_ln1)

    return GPUModelParams(
        embedding=grad_embedding,
        pos_encoding=create_gpu_buffer(model_params.pos_encoding.shape, device=device),
        layers=grad_layers,
    )


def create_zero_layer_grads(embedding_dim, device):
    """Create zero-initialized gradient buffers for a layer"""
    dim = embedding_dim
    return GPULayerParams(
        attn_wq=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        attn_wk=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        attn_wv=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        attn_wo=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        ff_w1=create_gpu_buffer(
            (dim, 4 * dim), np.zeros((dim, 4 * dim), dtype=np.float32), device
        ),
        ff_b1=create_gpu_buffer(
            (4 * dim,), np.zeros(4 * dim, dtype=np.float32), device
        ),
        ff_w2=create_gpu_buffer(
            (4 * dim, dim), np.zeros((4 * dim, dim), dtype=np.float32), device
        ),
        ff_b2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma1=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_beta1=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_beta2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
    )


# ============================================================================
# BACKWARD HELPER FUNCTIONS
# ============================================================================


def create_zero_layer_grads(embedding_dim, device):
    """Create zero-initialized gradient buffers for a layer"""
    dim = embedding_dim
    return GPULayerParams(
        attn_wq=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        attn_wk=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        attn_wv=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        attn_wo=create_gpu_buffer(
            (dim, dim), np.zeros((dim, dim), dtype=np.float32), device
        ),
        ff_w1=create_gpu_buffer(
            (dim, 4 * dim), np.zeros((dim, 4 * dim), dtype=np.float32), device
        ),
        ff_b1=create_gpu_buffer(
            (4 * dim,), np.zeros(4 * dim, dtype=np.float32), device
        ),
        ff_w2=create_gpu_buffer(
            (4 * dim, dim), np.zeros((4 * dim, dim), dtype=np.float32), device
        ),
        ff_b2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma1=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_beta1=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_beta2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
    )


def gelu_backward_gpu(input_buffer, grad_output):
    """
    Backward pass through GELU activation.

    Args:
        input_buffer: GPUBuffer - forward input
        grad_output: GPUBuffer - gradient from next layer

    Returns:
        GPUBuffer - gradient w.r.t. input
    """
    device = input_buffer.device
    grad_input = create_gpu_buffer(input_buffer.shape, device=device)

    size_buffer = device.create_buffer_with_data(
        data=np.array([input_buffer.size], dtype=np.uint32),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader_module = device.create_shader_module(code=GELU_BACKWARD_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": input_buffer.buffer,
                    "offset": 0,
                    "size": input_buffer.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": grad_output.buffer,
                    "offset": 0,
                    "size": grad_output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": grad_input.buffer,
                    "offset": 0,
                    "size": grad_input.size * 4,
                },
            },
            {"binding": 3, "resource": {"buffer": size_buffer, "offset": 0, "size": 4}},
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(input_buffer.size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return grad_input


def softmax_backward_gpu(softmax_output, grad_output):
    """
    Backward pass through softmax.

    Args:
        softmax_output: GPUBuffer - softmax output from forward
        grad_output: GPUBuffer - gradient from next layer

    Returns:
        GPUBuffer - gradient w.r.t. input
    """
    device = softmax_output.device

    # Flatten to 2D for processing
    original_shape = softmax_output.shape
    batch_dims = original_shape[:-1]
    last_dim = original_shape[-1]
    batch_size = int(np.prod(batch_dims))

    output_2d = reshape_buffer(softmax_output, (batch_size, last_dim))
    grad_output_2d = reshape_buffer(grad_output, (batch_size, last_dim))

    grad_input_2d = create_gpu_buffer((batch_size, last_dim), device=device)

    shape_buffer = device.create_buffer_with_data(
        data=np.array([batch_size, last_dim], dtype=np.uint32),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader_module = device.create_shader_module(code=SOFTMAX_BACKWARD_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": output_2d.buffer,
                    "offset": 0,
                    "size": output_2d.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": grad_output_2d.buffer,
                    "offset": 0,
                    "size": grad_output_2d.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": grad_input_2d.buffer,
                    "offset": 0,
                    "size": grad_input_2d.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {"buffer": shape_buffer, "offset": 0, "size": 8},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(batch_size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return reshape_buffer(grad_input_2d, original_shape)


def layer_norm_backward_gpu(x_input, gamma, grad_output):
    """
    Backward pass through layer normalization.

    Args:
        x_input: GPUBuffer - input to layer norm
        gamma: GPUBuffer - scale parameter
        grad_output: GPUBuffer - gradient from next layer

    Returns:
        (grad_input, grad_gamma, grad_beta)
    """
    device = x_input.device

    # Simplified: use CPU implementation for now
    # TODO: Implement full GPU version with LAYER_NORM_BACKWARD_SHADER

    x_np = gpu_to_numpy(x_input)
    gamma_np = gpu_to_numpy(gamma)
    grad_out_np = gpu_to_numpy(grad_output)

    # Compute gradients on CPU (simplified)
    original_shape = x_np.shape
    dim = original_shape[-1]
    batch_size = int(np.prod(original_shape[:-1]))

    x_2d = x_np.reshape(batch_size, dim)
    grad_out_2d = grad_out_np.reshape(batch_size, dim)

    # Compute mean and variance
    mean = np.mean(x_2d, axis=-1, keepdims=True)
    var = np.var(x_2d, axis=-1, keepdims=True)
    std = np.sqrt(var + 1e-5)
    x_hat = (x_2d - mean) / std

    # Gradients
    grad_gamma = np.sum(grad_out_2d * x_hat, axis=0)
    grad_beta = np.sum(grad_out_2d, axis=0)

    # Gradient w.r.t. input
    grad_x_hat = grad_out_2d * gamma_np
    grad_var = np.sum(
        grad_x_hat * (x_2d - mean) * -0.5 * (var + 1e-5) ** -1.5, axis=-1, keepdims=True
    )
    grad_mean = np.sum(
        grad_x_hat * -1.0 / std, axis=-1, keepdims=True
    ) + grad_var * np.mean(-2.0 * (x_2d - mean), axis=-1, keepdims=True)
    grad_input = (
        grad_x_hat / std + grad_var * 2.0 * (x_2d - mean) / dim + grad_mean / dim
    )

    grad_input = grad_input.reshape(original_shape)

    return (
        create_gpu_buffer(original_shape, grad_input, device),
        create_gpu_buffer((dim,), grad_gamma, device),
        create_gpu_buffer((dim,), grad_beta, device),
    )


def gelu_backward_gpu(input_buffer, grad_output):
    """
    Backward pass through GELU activation.

    Args:
        input_buffer: GPUBuffer - forward input
        grad_output: GPUBuffer - gradient from next layer

    Returns:
        GPUBuffer - gradient w.r.t. input
    """
    device = input_buffer.device
    grad_input = create_gpu_buffer(input_buffer.shape, device=device)

    size_buffer = device.create_buffer_with_data(
        data=np.array([input_buffer.size], dtype=np.uint32),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    shader_module = device.create_shader_module(code=GELU_BACKWARD_SHADER)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout, compute={"module": shader_module, "entry_point": "main"}
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": input_buffer.buffer,
                    "offset": 0,
                    "size": input_buffer.size * 4,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": grad_output.buffer,
                    "offset": 0,
                    "size": grad_output.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": grad_input.buffer,
                    "offset": 0,
                    "size": grad_input.size * 4,
                },
            },
            {"binding": 3, "resource": {"buffer": size_buffer, "offset": 0, "size": 4}},
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(math.ceil(input_buffer.size / 256), 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    return grad_input
