"""WGSL kernels for optimizer operations"""

# ============================================================================
# OPTIMIZER KERNELS
# ============================================================================

ADAMW_OPTIMIZER_KERNEL = """
// Fused AdamW optimizer update with decoupled weight decay

struct OptimizerParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    eps: f32,
    step: f32,
    size: u32,
}

@group(0) @binding(0) var<uniform> params: OptimizerParams;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let grad = gradients[idx];
    let weight = weights[idx];

    // Update biased first moment estimate
    let m_new = params.beta1 * m[idx] + (1.0 - params.beta1) * grad;
    m[idx] = m_new;

    // Update biased second raw moment estimate
    let v_new = params.beta2 * v[idx] + (1.0 - params.beta2) * grad * grad;
    v[idx] = v_new;

    // Compute bias-corrected first moment estimate
    let m_hat = m_new / (1.0 - pow(params.beta1, params.step));

    // Compute bias-corrected second raw moment estimate
    let v_hat = v_new / (1.0 - pow(params.beta2, params.step));

    // Update weights with AdamW (decoupled weight decay)
    let update = m_hat / (sqrt(v_hat) + params.eps);
    weights[idx] = weight - params.lr * (update + params.weight_decay * weight);
}
"""
