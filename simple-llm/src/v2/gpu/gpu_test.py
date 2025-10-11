"""Test GPU operations"""

import numpy as np
from gpu_buffer import create_gpu_buffer, gpu_to_numpy
from gpu_device import WGPU_AVAILABLE, create_device, create_pipeline_cache
from gpu_ops import run_layernorm, run_matmul

# ============================================================================
# TEST FUNCTION
# ============================================================================


def test_fixed_kernels() -> None:
    """Test the fixed implementation"""
    if not WGPU_AVAILABLE:
        print("⚠️  wgpu not available, skipping tests")
        return

    device = create_device()
    if device is None:
        print("⚠️  Could not initialize device")
        return

    pipeline_cache = create_pipeline_cache(device)

    print("\n🧪 Testing Fixed Kernels\n")

    # Test matrix multiplication
    print("Testing tiled matmul...")
    M, K, N = 64, 128, 64
    A_data = np.random.randn(M, K).astype(np.float32)
    B_data = np.random.randn(K, N).astype(np.float32)
    C_expected = A_data @ B_data

    A_gpu = create_gpu_buffer(device, (M, K), A_data)
    B_gpu = create_gpu_buffer(device, (K, N), B_data)
    C_gpu = create_gpu_buffer(device, (M, N))

    run_matmul(pipeline_cache, A_gpu, B_gpu, C_gpu)
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

    x_gpu = create_gpu_buffer(device, (batch, dim), x_data)
    gamma_gpu = create_gpu_buffer(device, (dim,), gamma_data)
    beta_gpu = create_gpu_buffer(device, (dim,), beta_data)
    out_gpu = create_gpu_buffer(device, (batch, dim))

    run_layernorm(pipeline_cache, x_gpu, gamma_gpu, beta_gpu, out_gpu)
    out_result = gpu_to_numpy(out_gpu)

    error = np.abs(out_result - x_norm_expected).max()
    print(f"  Max error: {error:.6f}")
    print("  ✅ PASS" if error < 1e-3 else "  ❌ FAIL")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_fixed_kernels()
