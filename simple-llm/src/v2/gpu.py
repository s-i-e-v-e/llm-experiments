"""
Fixed high-performance GPU transformer using WGSL kernels.
Complete implementation with all data structures and helper functions.
"""

import dataclasses
from typing import Tuple

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
_pipeline_cache = {}


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
# WGSL KERNELS
# ============================================================================

TILED_MATMUL_KERNEL = """
// Tiled matrix multiplication: C = A @ B
// Uses shared memory to reduce global memory accesses

struct MatmulParams {
    M: u32,  // Rows of A
    K: u32,  // Cols of A, Rows of B
    N: u32,  // Cols of B
}

@group(0) @binding(0) var<uniform> params: MatmulParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_A: array<f32, 256>;  // 16x16 tile
var<workgroup> tile_B: array<f32, 256>;  // 16x16 tile

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile of A into shared memory
        let a_row = row;
        let a_col = t * TILE_SIZE + local_col;
        if (a_row < params.M && a_col < params.K) {
            tile_A[local_row * TILE_SIZE + local_col] = A[a_row * params.K + a_col];
        } else {
            tile_A[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        let b_col = col;
        if (b_row < params.K && b_col < params.N) {
            tile_B[local_row * TILE_SIZE + local_col] = B[b_row * params.N + b_col];
        } else {
            tile_B[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize to ensure all threads have loaded their data
        workgroupBarrier();

        // Compute partial dot product using shared memory
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_A[local_row * TILE_SIZE + k] * tile_B[k * TILE_SIZE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}
"""

LAYERNORM_KERNEL = """
struct NormParams {
    size: u32,
    n_elements: u32,
}

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const EPS: f32 = 1e-5;
const BLOCK_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let tid = local_id.x;
    let elem_idx = idx / params.size;

    if (elem_idx >= params.n_elements) {
        return;
    }

    let offset = elem_idx * params.size;

    // Compute mean using parallel reduction
    var sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        sum += input[offset + i];
    }
    shared_data[tid] = sum;
    workgroupBarrier();

    // Reduction
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    let mean = shared_data[0] / f32(params.size);

    // Compute variance
    var var_sum = 0.0;
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    shared_data[tid] = var_sum;
    workgroupBarrier();

    // Reduction
    for (var s = BLOCK_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    let variance = shared_data[0] / f32(params.size);
    let inv_std = 1.0 / sqrt(variance + EPS);

    // Normalize and apply affine transformation
    for (var i = tid; i < params.size; i += BLOCK_SIZE) {
        let normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}
"""

GELU_KERNEL = """
struct GeluParams {
    size: u32,
}

@group(0) @binding(0) var<uniform> params: GeluParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"""

RESIDUAL_ADD_KERNEL = """
struct AddParams {
    size: u32,
}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> input_a: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    output[idx] = input_a[idx] + input_b[idx];
}
"""

EMBEDDING_KERNEL = """
struct EmbedParams {
    batch_size: u32,
    seq_len: u32,
    embedding_dim: u32,
}

@group(0) @binding(0) var<uniform> params: EmbedParams;
@group(0) @binding(1) var<storage, read> embedding_table: array<f32>;
@group(0) @binding(2) var<storage, read> pos_encoding: array<f32>;
@group(0) @binding(3) var<storage, read> input_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.seq_len;

    if (idx >= total) {
        return;
    }

    let seq_idx = idx % params.seq_len;
    let token_id = input_ids[idx];
    let D = params.embedding_dim;

    let emb_offset = token_id * D;
    let pos_offset = seq_idx * D;
    let out_offset = idx * D;

    for (var d = 0u; d < D; d++) {
        output[out_offset + d] = embedding_table[emb_offset + d] + pos_encoding[pos_offset + d];
    }
}
"""

BIAS_ADD_KERNEL = """
struct BiasParams {
    size: u32,
    dim: u32,
}

@group(0) @binding(0) var<uniform> params: BiasParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let row = idx / params.dim;
    let col = idx % params.dim;

    output[idx] = input[idx] + bias[col];
}
"""


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclasses.dataclass
class GPUBuffer:
    buffer: object
    shape: Tuple[int, ...]
    size: int
    device: object


@dataclasses.dataclass
class GPULayerParams:
    attn_wq: GPUBuffer
    attn_wk: GPUBuffer
    attn_wv: GPUBuffer
    attn_wo: GPUBuffer
    ff_w1: GPUBuffer
    ff_b1: GPUBuffer
    ff_w2: GPUBuffer
    ff_b2: GPUBuffer
    ln_gamma1: GPUBuffer
    ln_beta1: GPUBuffer
    ln_gamma2: GPUBuffer
    ln_beta2: GPUBuffer


@dataclasses.dataclass
class GPUModelParams:
    embedding: GPUBuffer
    pos_encoding: GPUBuffer
    layers: list  # List of GPULayerParams


@dataclasses.dataclass
class GPUOptimizerState:
    m_embedding: GPUBuffer
    v_embedding: GPUBuffer
    m_layers: list  # List of GPULayerParams (momentum)
    v_layers: list  # List of GPULayerParams (variance)
    step: int


# ============================================================================
# BUFFER MANAGEMENT
# ============================================================================


def create_gpu_buffer(shape, data=None, device=None):
    """Create GPU buffer"""
    device = device or get_device()
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

    return GPUBuffer(buffer=buffer, shape=shape, size=size, device=device)


def gpu_to_numpy(gpu_buffer):
    """Read GPU buffer back to CPU"""
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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def positional_encoding(seq_len: int, dim: int) -> np.ndarray:
    """Generate sinusoidal positional encoding"""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(dim)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angle_rads = pos * angle_rates

    pos_encoding = np.zeros((seq_len, dim), dtype=np.float32)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pos_encoding


def _get_or_create_pipeline(shader_code: str, device=None):
    """Cache compute pipelines"""
    device = device or get_device()
    cache_key = (id(device), hash(shader_code))

    if cache_key not in _pipeline_cache:
        shader_module = device.create_shader_module(code=shader_code)
        pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": shader_module, "entry_point": "main"}
        )
        _pipeline_cache[cache_key] = pipeline

    return _pipeline_cache[cache_key]


# ============================================================================
# KERNEL EXECUTION FUNCTIONS
# ============================================================================


def run_matmul(A: GPUBuffer, B: GPUBuffer, C: GPUBuffer, device=None):
    """Execute tiled matrix multiplication: C = A @ B"""
    device = device or get_device()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible shapes: {A.shape} @ {B.shape}"
    assert C.shape == (M, N), f"Output shape mismatch: {C.shape} != ({M}, {N})"

    params = np.array([M, K, N], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(TILED_MATMUL_KERNEL, device)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {"buffer": A.buffer, "offset": 0, "size": A.size * 4},
            },
            {
                "binding": 2,
                "resource": {"buffer": B.buffer, "offset": 0, "size": B.size * 4},
            },
            {
                "binding": 3,
                "resource": {"buffer": C.buffer, "offset": 0, "size": C.size * 4},
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((N + 15) // 16, (M + 15) // 16, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_layernorm(
    input_buf: GPUBuffer,
    gamma: GPUBuffer,
    beta: GPUBuffer,
    output: GPUBuffer,
    device=None,
):
    """Execute layer normalization"""
    device = device or get_device()

    n_elements, size = input_buf.shape

    params = np.array([size, n_elements], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(LAYERNORM_KERNEL, device)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": gamma.buffer,
                    "offset": 0,
                    "size": gamma.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {"buffer": beta.buffer, "offset": 0, "size": beta.size * 4},
            },
            {
                "binding": 4,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n_elements, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_gelu(input_buf: GPUBuffer, output: GPUBuffer, device=None):
    """Apply GELU activation"""
    device = device or get_device()

    total_size = input_buf.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(GELU_KERNEL, device)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
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
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_residual_add(
    input_a: GPUBuffer, input_b: GPUBuffer, output: GPUBuffer, device=None
):
    """Element-wise addition for residual connections"""
    device = device or get_device()

    total_size = input_a.size
    assert input_a.size == input_b.size == output.size

    params = np.array([total_size], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(RESIDUAL_ADD_KERNEL, device)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_a.buffer,
                    "offset": 0,
                    "size": input_a.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": input_b.buffer,
                    "offset": 0,
                    "size": input_b.size * 4,
                },
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


def run_bias_add(input_buf: GPUBuffer, bias: GPUBuffer, output: GPUBuffer, device=None):
    """Add bias vector to each row of input matrix"""
    device = device or get_device()

    n_elements, dim = input_buf.shape
    total_size = n_elements * dim

    params = np.array([total_size, dim], dtype=np.uint32)
    params_buffer = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )

    pipeline = _get_or_create_pipeline(BIAS_ADD_KERNEL, device)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {
                "binding": 0,
                "resource": {
                    "buffer": params_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": {
                    "buffer": input_buf.buffer,
                    "offset": 0,
                    "size": input_buf.size * 4,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": bias.buffer, "offset": 0, "size": bias.size * 4},
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": output.buffer,
                    "offset": 0,
                    "size": output.size * 4,
                },
            },
        ],
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups((total_size + 255) // 256, 1, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])


# ============================================================================
# MODEL CREATION
# ============================================================================


def create_gpu_layer_params(embedding_dim: int, device=None) -> GPULayerParams:
    """Initialize GPU layer parameters"""
    device = device or get_device()
    dim = embedding_dim

    return GPULayerParams(
        attn_wq=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        attn_wk=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        attn_wv=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        attn_wo=create_gpu_buffer(
            (dim, dim), np.random.normal(0, 0.02, (dim, dim)).astype(np.float32), device
        ),
        ff_w1=create_gpu_buffer(
            (dim, 4 * dim),
            np.random.normal(0, 0.02, (dim, 4 * dim)).astype(np.float32),
            device,
        ),
        ff_b1=create_gpu_buffer(
            (4 * dim,), np.zeros(4 * dim, dtype=np.float32), device
        ),
        ff_w2=create_gpu_buffer(
            (4 * dim, dim),
            np.random.normal(0, 0.02, (4 * dim, dim)).astype(np.float32),
            device,
        ),
        ff_b2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma1=create_gpu_buffer((dim,), np.ones(dim, dtype=np.float32), device),
        ln_beta1=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
        ln_gamma2=create_gpu_buffer((dim,), np.ones(dim, dtype=np.float32), device),
        ln_beta2=create_gpu_buffer((dim,), np.zeros(dim, dtype=np.float32), device),
    )


def create_gpu_model_params(
    vocab_size: int, embedding_dim: int, context_length: int, n_layers: int, device=None
) -> GPUModelParams:
    """Initialize complete GPU model"""
    device = device or get_device()

    embedding_data = np.random.normal(0, 0.02, (vocab_size, embedding_dim)).astype(
        np.float32
    )
    embedding = create_gpu_buffer((vocab_size, embedding_dim), embedding_data, device)

    pos_encoding_data = positional_encoding(context_length, embedding_dim)
    pos_encoding = create_gpu_buffer(
        (context_length, embedding_dim), pos_encoding_data, device
    )

    layers = [create_gpu_layer_params(embedding_dim, device) for _ in range(n_layers)]

    return GPUModelParams(embedding=embedding, pos_encoding=pos_encoding, layers=layers)


def create_optimizer_state(model_params: GPUModelParams) -> GPUOptimizerState:
    """Initialize optimizer state (zero moments)"""
    device = model_params.embedding.device

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

    m_layers = []
    v_layers = []
    for layer in model_params.layers:
        dim = layer.attn_wq.shape[0]

        m_layer = create_gpu_layer_params(dim, device)
        v_layer = create_gpu_layer_params(dim, device)

        # Zero initialize all buffers
        for attr in [
            "attn_wq",
            "attn_wk",
            "attn_wv",
            "attn_wo",
            "ff_w1",
            "ff_b1",
            "ff_w2",
            "ff_b2",
            "ln_gamma1",
            "ln_beta1",
            "ln_gamma2",
            "ln_beta2",
        ]:
            buf = getattr(m_layer, attr)
            device.queue.write_buffer(
                buf.buffer, 0, np.zeros(buf.size, dtype=np.float32)
            )
            buf = getattr(v_layer, attr)
            device.queue.write_buffer(
                buf.buffer, 0, np.zeros(buf.size, dtype=np.float32)
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


# ============================================================================
# SERIALIZATION
# ============================================================================


def gpu_layer_to_dict(layer: GPULayerParams) -> dict:
    """Convert GPU layer to dict"""
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


def dict_to_gpu_layer(data: dict, embedding_dim: int, device=None) -> GPULayerParams:
    """Create GPU layer from dict"""
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
# TEST FUNCTION
# ============================================================================


def test_fixed_kernels():
    """Test the fixed implementation"""
    if not WGPU_AVAILABLE:
        print("⚠️  wgpu not available, skipping tests")
        return

    device = get_device()
    if device is None:
        print("⚠️  Could not initialize device")
        return

    print("\n🧪 Testing Fixed Kernels\n")

    # Test matrix multiplication
    print("Testing tiled matmul...")
    M, K, N = 64, 128, 64
    A_data = np.random.randn(M, K).astype(np.float32)
    B_data = np.random.randn(K, N).astype(np.float32)
    C_expected = A_data @ B_data

    A_gpu = create_gpu_buffer((M, K), A_data, device)
    B_gpu = create_gpu_buffer((K, N), B_data, device)
    C_gpu = create_gpu_buffer((M, N), device=device)

    run_matmul(A_gpu, B_gpu, C_gpu, device)
    C_result = gpu_to_numpy(C_gpu)

    error = np.abs(C_result - C_expected).max()
    print(f"  Max error: {error:.6f}")
    print("  ✅ PASS" if error < 1e-3 else "  ❌ FAIL")

    # Test layer normalization
    print("\nTesting layer normalization...")
    batch, dim = 32, 128
    x_data = np.random.randn(batch, dim).astype(np.float32)
    gamma_data = np.ones(dim, dtype=np.float32)
    beta_data = np.zeros(dim, dtype=np.float32)

    # CPU reference
    mean = x_data.mean(axis=1, keepdims=True)
    var = x_data.var(axis=1, keepdims=True)
    x_norm_expected = (x_data - mean) / np.sqrt(var + 1e-5)

    x_gpu = create_gpu_buffer((batch, dim), x_data, device)
    gamma_gpu = create_gpu_buffer((dim,), gamma_data, device)
    beta_gpu = create_gpu_buffer((dim,), beta_data, device)
    out_gpu = create_gpu_buffer((batch, dim), device=device)

    run_layernorm(x_gpu, gamma_gpu, beta_gpu, out_gpu, device)
    out_result = gpu_to_numpy(out_gpu)

    error = np.abs(out_result - x_norm_expected).max()
    print(f"  Max error: {error:.6f}")
    print("  ✅ PASS" if error < 1e-3 else "  ❌ FAIL")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_fixed_kernels()
