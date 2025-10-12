"""Test GPU operations"""

import numpy as np

from .gpu_buffer import create_gpu_buffer_1d, create_gpu_buffer_2d, gpu_to_numpy
from .gpu_device import create_device, create_pipeline_cache
from .gpu_operations import run_layernorm, run_matmul

# ============================================================================
# TEST FUNCTIONS
# ============================================================================


def test_matmul(device: GPUDevice) -> bool:
    """Test matrix multiplication"""
    print("Testing tiled matmul...")
    M, K, N = 64, 128, 64
    A_data = np.random.randn(M, K).astype(np.float32)
    B_data = np.random.randn(K, N).astype(np.float32)
    C_expected = A_data @ B_data

    pipeline_cache = create_pipeline_cache(device)

    A_gpu = create_gpu_buffer_2d(device, M, K, A_data)
    B_gpu = create_gpu_buffer_2d(device, K, N, B_data)
    C_gpu = create_gpu_buffer_2d(device, M, N)

    run_matmul(pipeline_cache, A_gpu, B_gpu, C_gpu)
    C_result = gpu_to_numpy(C_gpu)

    error = np.abs(C_result - C_expected).max()
    print(f"  Max error: {error:.6f}")

    if error < 1e-3:
        print("  ✅ PASS")
        return True
    else:
        print("  ❌ FAIL")
        return False


def test_layernorm(device: GPUDevice) -> bool:
    """Test layer normalization"""
    print("\nTesting layer normalization...")
    batch, dim = 32, 128
    x_data = np.random.randn(batch, dim).astype(np.float32)
    gamma_data = np.ones(dim, dtype=np.float32)
    beta_data = np.zeros(dim, dtype=np.float32)

    # CPU reference
    mean = x_data.mean(axis=1, keepdims=True)
    var = x_data.var(axis=1, keepdims=True)
    x_norm_expected = (x_data - mean) / np.sqrt(var + 1e-5)

    pipeline_cache = create_pipeline_cache(device)

    x_gpu = create_gpu_buffer_2d(device, batch, dim, x_data)
    gamma_gpu = create_gpu_buffer_1d(device, dim, gamma_data)
    beta_gpu = create_gpu_buffer_1d(device, dim, beta_data)
    out_gpu = create_gpu_buffer_2d(device, batch, dim)

    run_layernorm(pipeline_cache, x_gpu, gamma_gpu, beta_gpu, out_gpu)
    out_result = gpu_to_numpy(out_gpu)

    error = np.abs(out_result - x_norm_expected).max()
    print(f"  Max error: {error:.6f}")

    if error < 1e-3:
        print("  ✅ PASS")
        return True
    else:
        print("  ❌ FAIL")
        return False


def run_all_tests() -> None:
    """Run all GPU operation tests"""
    device = create_device()

    print("\n🧪 Testing GPU Operations\n")

    results = []
    results.append(test_matmul(device))
    results.append(test_layernorm(device))

    print(f"\n{'=' * 50}")
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests completed successfully!")
    else:
        print(f"❌ {total - passed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
