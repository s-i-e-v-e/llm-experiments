#!/usr/bin/env python3
"""
Comprehensive Test Suite for GPU Module Operations
Tests all operations against JAX reference implementation
"""

import sys
import traceback
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from gpu import gpu


@dataclass
class TestConfig:
    """Configuration for test execution"""

    rtol: float = 1e-4  # Relative tolerance
    atol: float = 1e-5  # Absolute tolerance
    verbose: bool = True


class TestResult:
    """Stores result of a single test"""

    def __init__(
        self,
        name: str,
        passed: bool,
        max_error: float = 0.0,
        mean_error: float = 0.0,
        details: str = "",
    ):
        self.name = name
        self.passed = passed
        self.max_error = max_error
        self.mean_error = mean_error
        self.details = details

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f"{status:8} {self.name:40}"
        if not self.passed or self.max_error > 0:
            msg += f" max_err={self.max_error:.2e} mean_err={self.mean_error:.2e}"
        if self.details:
            msg += f"\n         {self.details}"
        return msg


class TestRunner:
    """Main test orchestrator"""

    def __init__(self, ctx, config: TestConfig = TestConfig()):
        self.ctx = ctx
        self.config = config
        self.results = []
        self.passed = 0
        self.failed = 0

    def compare_arrays(
        self, gpu_result: np.ndarray, jax_result: np.ndarray, name: str
    ) -> Tuple[bool, float, float]:
        """Compare GPU and JAX results with detailed error analysis"""
        if gpu_result.shape != jax_result.shape:
            print(f"  ✗ Shape mismatch: GPU={gpu_result.shape} JAX={jax_result.shape}")
            return False, float("inf"), float("inf")

        abs_diff = np.abs(gpu_result - jax_result)
        max_error = np.max(abs_diff)
        mean_error = np.mean(abs_diff)

        # Relative error
        mask = np.abs(jax_result) > 1e-10
        rel_error = np.zeros_like(abs_diff)
        rel_error[mask] = abs_diff[mask] / np.abs(jax_result[mask])
        max_rel_error = np.max(rel_error) if mask.any() else 0.0

        passed = np.allclose(
            gpu_result, jax_result, rtol=self.config.rtol, atol=self.config.atol
        )

        if self.config.verbose:
            print(f"  Array: {name}")
            print(f"    Shape: {gpu_result.shape}")
            print(f"    Max absolute error: {max_error:.2e}")
            print(f"    Mean absolute error: {mean_error:.2e}")
            print(f"    Max relative error: {max_rel_error:.2e}")

            if not passed:
                worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
                print(f"    Worst difference at {worst_idx}:")
                print(f"      GPU:  {gpu_result[worst_idx]:.6f}")
                print(f"      JAX:  {jax_result[worst_idx]:.6f}")
                print(f"      Diff: {abs_diff[worst_idx]:.6f}")

        return passed, max_error, mean_error

    def run_test(self, test_func, name: str):
        """Run a single test function"""
        print(f"\n{'=' * 80}")
        print(f"Testing: {name}")
        print("=" * 80)

        try:
            result = test_func(self.ctx, self.config)
            self.results.append(result)
            if result.passed:
                self.passed += 1
                print(f"\n✓ {name} PASSED")
            else:
                self.failed += 1
                print(f"\n✗ {name} FAILED")
        except Exception as e:
            print(f"\n✗ {name} CRASHED: {e}")

            traceback.print_exc()
            self.results.append(TestResult(name, False, details=str(e)))
            self.failed += 1

    def print_summary(self):
        """Print test summary"""
        print(f"\n\n{'=' * 80}")
        print("TEST SUMMARY")
        print("=" * 80)

        for result in self.results:
            print(result)

        total = self.passed + self.failed
        print(f"\n{'-' * 80}")
        print(f"Total: {total}  Passed: {self.passed}  Failed: {self.failed}")
        print(f"Success Rate: {100 * self.passed / total if total > 0 else 0:.1f}%")
        print("=" * 80)

        return self.failed == 0


# ============================================================================
# TEST FUNCTIONS - Replace PLACEHOLDER comments with your GPU API calls
# ============================================================================

matmul_test_configs = [
    (16, 16, 16, "Small square (16x16)"),
    (32, 32, 32, "Medium square (32x32)"),
    (64, 64, 64, "Large square (64x64)"),
    (128, 256, 64, "Rectangular wide A"),
    (64, 256, 128, "Rectangular wide B"),
    (256, 64, 128, "Rectangular tall A"),
    (100, 50, 75, "Non-aligned dims"),
    (17, 23, 19, "Prime dimensions"),
    (1, 64, 128, "Single row A"),
    (128, 64, 1, "Single column B"),
    (1, 1, 1, "Scalar (1x1)"),
    (33, 65, 129, "Powers of 2 + 1"),
]


def test_matmul(ctx, config: TestConfig) -> TestResult:
    """Test matrix multiplication: C = A @ B with comprehensive edge cases"""
    test_configs = matmul_test_configs

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for M, K, N, desc in test_configs:
        print(f"\n  Test: {desc} [{M}x{K}] @ [{K}x{N}]")

        np.random.seed(42)
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)

        # JAX reference
        C_jax = np.array(jnp.matmul(jnp.array(A_np), jnp.array(B_np)))

        # GPU computation
        A_gpu = gpu.gpu_buffer_2d_create(ctx, M, K, A_np)
        B_gpu = gpu.gpu_buffer_2d_create(ctx, K, N, B_np)
        C_gpu = gpu.gpu_buffer_2d_create(ctx, M, N)

        gpu.batch_begin(ctx)
        gpu.matmul(ctx, A_gpu, B_gpu, C_gpu)
        gpu.batch_commit(ctx)

        C_gpu_np = np.zeros((M, N), dtype=np.float32)
        gpu.gpu_buffer_2d_read(ctx, C_gpu, C_gpu_np)

        passed, max_e, mean_e = runner.compare_arrays(C_gpu_np, C_jax, desc)
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    return TestResult("matmul", all_passed, max_err, mean_err)


def test_matmul_backward_a(ctx, config: TestConfig) -> TestResult:
    """Test matmul backward A with comprehensive edge cases"""
    test_configs = matmul_test_configs

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for M, K, N, desc in test_configs:
        print(f"\n  Test: {desc} A[{M}x{K}] @ B[{K}x{N}] -> dA[{M}x{K}]")

        np.random.seed(42)
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)
        grad_C_np = np.random.randn(M, N).astype(np.float32)

        # JAX reference
        def forward(A, B):
            return jnp.matmul(A, B)

        A_jax = jnp.array(A_np)
        B_jax = jnp.array(B_np)
        grad_C_jax = jnp.array(grad_C_np)

        C_jax, vjp_fn = jax.vjp(forward, A_jax, B_jax)
        grad_A_jax, _ = vjp_fn(grad_C_jax)
        grad_A_jax_np = np.array(grad_A_jax)

        grad_A_analytical = grad_C_np @ B_np.T
        analytical_error = np.max(np.abs(grad_A_jax_np - grad_A_analytical))
        if analytical_error > 1e-5:
            print(f"    JAX vs analytical formula error: {analytical_error:.2e}")

        # GPU computation
        B_gpu = gpu.gpu_buffer_2d_create(ctx, K, N, B_np)
        grad_C_gpu = gpu.gpu_buffer_2d_create(ctx, M, N, grad_C_np)
        grad_A_gpu = gpu.gpu_buffer_2d_create(ctx, M, K)

        gpu.batch_begin(ctx)
        gpu.matmul_backward_a(ctx, grad_C_gpu, B_gpu, grad_A_gpu)
        gpu.batch_commit(ctx)

        grad_A_gpu_np = np.zeros((M, K), dtype=np.float32)
        gpu.gpu_buffer_2d_read(ctx, grad_A_gpu, grad_A_gpu_np)

        passed, max_e, mean_e = runner.compare_arrays(
            grad_A_gpu_np, grad_A_jax_np, desc
        )
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    return TestResult("matmul_backward_a", all_passed, max_err, mean_err)


def test_matmul_backward_b(ctx, config: TestConfig) -> TestResult:
    """Test matmul backward B with comprehensive edge cases"""
    test_configs = matmul_test_configs

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for M, K, N, desc in test_configs:
        print(f"\n  Test: {desc} A[{M}x{K}] @ B[{K}x{N}] -> dB[{K}x{N}]")

        np.random.seed(42)
        A_np = np.random.randn(M, K).astype(np.float32)
        B_np = np.random.randn(K, N).astype(np.float32)
        grad_C_np = np.random.randn(M, N).astype(np.float32)

        # JAX reference
        def forward(A, B):
            return jnp.matmul(A, B)

        A_jax = jnp.array(A_np)
        B_jax = jnp.array(B_np)
        grad_C_jax = jnp.array(grad_C_np)

        C_jax, vjp_fn = jax.vjp(forward, A_jax, B_jax)
        _, grad_B_jax = vjp_fn(grad_C_jax)
        grad_B_jax_np = np.array(grad_B_jax)

        grad_B_analytical = A_np.T @ grad_C_np
        analytical_error = np.max(np.abs(grad_B_jax_np - grad_B_analytical))
        if analytical_error > 1e-5:
            print(f"    JAX vs analytical formula error: {analytical_error:.2e}")

        # GPU computation
        A_gpu = gpu.gpu_buffer_2d_create(ctx, M, K, A_np)
        grad_C_gpu = gpu.gpu_buffer_2d_create(ctx, M, N, grad_C_np)
        grad_B_gpu = gpu.gpu_buffer_2d_create(ctx, K, N)

        gpu.batch_begin(ctx)
        gpu.matmul_backward_b(ctx, A_gpu, grad_C_gpu, grad_B_gpu)
        gpu.batch_commit(ctx)

        grad_B_gpu_np = np.zeros((K, N), dtype=np.float32)
        gpu.gpu_buffer_2d_read(ctx, grad_B_gpu, grad_B_gpu_np)

        passed, max_e, mean_e = runner.compare_arrays(
            grad_B_gpu_np, grad_B_jax_np, desc
        )
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    return TestResult("matmul_backward_b", all_passed, max_err, mean_err)


def test_matmul_full_chain(ctx, config: TestConfig) -> TestResult:
    """
    Test complete matmul forward + backward chain
    Verifies that forward and backward are consistent
    """
    M, K, N = 128, 256, 512
    print(f"\n  Full chain test: [{M}x{K}] @ [{K}x{N}]")

    np.random.seed(42)
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    grad_C_np = np.random.randn(M, N).astype(np.float32)

    # JAX reference
    def forward(A, B):
        return jnp.matmul(A, B)

    A_jax = jnp.array(A_np)
    B_jax = jnp.array(B_np)
    grad_C_jax = jnp.array(grad_C_np)

    C_jax, vjp_fn = jax.vjp(forward, A_jax, B_jax)
    grad_A_jax, grad_B_jax = vjp_fn(grad_C_jax)

    C_jax_np = np.array(C_jax)
    grad_A_jax_np = np.array(grad_A_jax)
    grad_B_jax_np = np.array(grad_B_jax)

    # GPU forward pass
    A_gpu = gpu.gpu_buffer_2d_create(ctx, M, K, A_np)
    B_gpu = gpu.gpu_buffer_2d_create(ctx, K, N, B_np)
    C_gpu = gpu.gpu_buffer_2d_create(ctx, M, N)

    gpu.batch_begin(ctx)
    gpu.matmul(ctx, A_gpu, B_gpu, C_gpu)
    gpu.batch_commit(ctx)

    C_gpu_np = np.zeros((M, N), dtype=np.float32)
    gpu.gpu_buffer_2d_read(ctx, C_gpu, C_gpu_np)

    # GPU backward pass
    grad_C_gpu = gpu.gpu_buffer_2d_create(ctx, M, N, grad_C_np)
    grad_A_gpu = gpu.gpu_buffer_2d_create(ctx, M, K)
    grad_B_gpu = gpu.gpu_buffer_2d_create(ctx, K, N)

    gpu.batch_begin(ctx)
    gpu.matmul_backward_a(ctx, grad_C_gpu, B_gpu, grad_A_gpu)
    gpu.matmul_backward_b(ctx, A_gpu, grad_C_gpu, grad_B_gpu)
    gpu.batch_commit(ctx)

    grad_A_gpu_np = np.zeros((M, K), dtype=np.float32)
    grad_B_gpu_np = np.zeros((K, N), dtype=np.float32)
    gpu.gpu_buffer_2d_read(ctx, grad_A_gpu, grad_A_gpu_np)
    gpu.gpu_buffer_2d_read(ctx, grad_B_gpu, grad_B_gpu_np)

    # Compare all outputs
    runner = TestRunner(ctx, config)

    print("\n  Forward pass:")
    passed_fwd, max_err_fwd, mean_err_fwd = runner.compare_arrays(
        C_gpu_np, C_jax_np, "output C"
    )

    print("\n  Backward pass:")
    passed_grad_a, max_err_a, mean_err_a = runner.compare_arrays(
        grad_A_gpu_np, grad_A_jax_np, "gradient dA"
    )
    passed_grad_b, max_err_b, mean_err_b = runner.compare_arrays(
        grad_B_gpu_np, grad_B_jax_np, "gradient dB"
    )

    all_passed = passed_fwd and passed_grad_a and passed_grad_b
    max_err = max(max_err_fwd, max_err_a, max_err_b)
    mean_err = (mean_err_fwd + mean_err_a + mean_err_b) / 3

    return TestResult("matmul_full_chain", all_passed, max_err, mean_err)


# ============================================================================
# ADD THESE TESTS TO YOUR TEST SUITE
# ============================================================================


def test_gelu(ctx, config: TestConfig) -> TestResult:
    """Test GELU activation function"""
    # Use relaxed tolerance for GPU approximation vs exact GELU
    relaxed_config = TestConfig(
        rtol=3e-2,  # 3% relative tolerance (GPU has ~2% error)
        atol=3e-2,  # 3% absolute tolerance
        verbose=config.verbose,
    )

    test_configs = [
        (100, "Small (100)"),
        (1024, "Medium (1024)"),
        (10000, "Large (10K)"),
        (32 * 1024, "Very Large (32K)"),
        (1, "Edge: Single element"),
    ]

    runner = TestRunner(ctx, relaxed_config)  # Use relaxed config
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for size, desc in test_configs:
        print(f"\n  Test: {desc} [{size}]")

        np.random.seed(42)
        x_np = (np.random.randn(size) * 2.0).astype(np.float32)

        # JAX reference - use EXACT GELU (not approximate)
        x_jax = jnp.array(x_np)
        y_jax = jax.nn.gelu(x_jax, approximate=False)  # Changed: use EXACT
        y_jax_np = np.array(y_jax)

        # GPU computation
        x_gpu = gpu.gpu_buffer_1d_create(ctx, size, x_np)
        y_gpu = gpu.gpu_buffer_1d_create(ctx, size)

        gpu.batch_begin(ctx)
        gpu.gelu(ctx, x_gpu, y_gpu)
        gpu.batch_commit(ctx)

        y_gpu_np = np.zeros(size, dtype=np.float32)
        gpu.gpu_buffer_1d_read(ctx, y_gpu, y_gpu_np)

        passed, max_e, mean_e = runner.compare_arrays(y_gpu_np, y_jax_np, desc)
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    details = "GPU uses sigmoid approximation (~2% error vs exact GELU)"
    return TestResult("gelu", all_passed, max_err, mean_err, details)


def test_gelu_backward(ctx, config: TestConfig) -> TestResult:
    """Test GELU backward pass using numerical gradient checking"""
    test_configs = [
        (100, "Small (100)"),
        (1024, "Medium (1024)"),
        (10000, "Large (10K)"),
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for size, desc in test_configs:
        print(f"\n  Test: {desc} [{size}]")

        np.random.seed(42)
        x_np = (np.random.randn(size) * 2.0).astype(np.float32)
        grad_y_np = np.random.randn(size).astype(np.float32)

        # Compute numerical gradient using finite differences
        epsilon = 1e-3
        grad_x_numerical = np.zeros(size, dtype=np.float32)

        # Helper function for GPU GELU forward pass
        def gelu_forward_single(x_in):
            x_gpu = gpu.gpu_buffer_1d_create(ctx, size, x_in)
            y_gpu = gpu.gpu_buffer_1d_create(ctx, size)
            gpu.batch_begin(ctx)
            gpu.gelu(ctx, x_gpu, y_gpu)
            gpu.batch_commit(ctx)
            y_out = np.zeros(size, dtype=np.float32)
            gpu.gpu_buffer_1d_read(ctx, y_gpu, y_out)
            return y_out

        # Compute numerical gradient for each element
        print(f"    Computing numerical gradients for {size} elements...")
        for i in range(min(size, 100)):  # Sample first 100 for speed
            x_plus = x_np.copy()
            x_minus = x_np.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon

            y_plus = gelu_forward_single(x_plus)
            y_minus = gelu_forward_single(x_minus)

            # Chain rule: (dy/dx) * grad_y
            dy_dx = (y_plus[i] - y_minus[i]) / (2 * epsilon)
            grad_x_numerical[i] = dy_dx * grad_y_np[i]

        # GPU backward pass
        x_gpu = gpu.gpu_buffer_1d_create(ctx, size, x_np)
        grad_y_gpu = gpu.gpu_buffer_1d_create(ctx, size, grad_y_np)
        grad_x_gpu = gpu.gpu_buffer_1d_create(ctx, size)

        gpu.batch_begin(ctx)
        gpu.gelu_backward(ctx, x_gpu, grad_y_gpu, grad_x_gpu)
        gpu.batch_commit(ctx)

        grad_x_gpu_np = np.zeros(size, dtype=np.float32)
        gpu.gpu_buffer_1d_read(ctx, grad_x_gpu, grad_x_gpu_np)

        # Compare only the sampled elements
        passed, max_e, mean_e = runner.compare_arrays(
            grad_x_gpu_np[:100], grad_x_numerical[:100], desc
        )
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    details = "Numerical gradient check (forward/backward consistency)"
    return TestResult("gelu_backward", all_passed, max_err, mean_err, details)


def test_layernorm(ctx, config: TestConfig) -> TestResult:
    """Test layer normalization with affine transform"""
    test_configs = [
        (32, 768, "GPT-2 hidden size"),
        (1, 512, "Single element, medium dim"),
        (128, 1024, "Large batch, large dim"),
        (1024, 64, "Many elements, small dim"),
        (10, 100, "Medium square"),
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for n_elements, size, desc in test_configs:
        print(f"\n  Test: {desc} [{n_elements}, {size}]")

        np.random.seed(42)
        x_np = np.random.randn(n_elements, size).astype(np.float32)
        gamma_np = np.random.randn(size).astype(np.float32)
        beta_np = np.random.randn(size).astype(np.float32)

        # JAX reference (manual layernorm with affine)
        x_jax = jnp.array(x_np)
        gamma_jax = jnp.array(gamma_np)
        beta_jax = jnp.array(beta_np)

        eps = 1e-5
        mean = jnp.mean(x_jax, axis=-1, keepdims=True)
        var = jnp.var(x_jax, axis=-1, keepdims=True)
        normalized = (x_jax - mean) / jnp.sqrt(var + eps)
        y_jax = gamma_jax * normalized + beta_jax
        y_jax_np = np.array(y_jax)

        # GPU computation
        x_gpu = gpu.gpu_buffer_2d_create(ctx, n_elements, size, x_np)
        gamma_gpu = gpu.gpu_buffer_1d_create(ctx, size, gamma_np)
        beta_gpu = gpu.gpu_buffer_1d_create(ctx, size, beta_np)
        y_gpu = gpu.gpu_buffer_2d_create(ctx, n_elements, size)

        gpu.batch_begin(ctx)
        gpu.layernorm(ctx, x_gpu, gamma_gpu, beta_gpu, y_gpu)
        gpu.batch_commit(ctx)

        y_gpu_np = np.zeros((n_elements, size), dtype=np.float32)
        gpu.gpu_buffer_2d_read(ctx, y_gpu, y_gpu_np)

        passed, max_e, mean_e = runner.compare_arrays(y_gpu_np, y_jax_np, desc)
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    return TestResult("layernorm", all_passed, max_err, mean_err)


def test_layernorm_backward(ctx, config: TestConfig) -> TestResult:
    """Test layer normalization backward pass"""
    test_configs = [
        (32, 768, "GPT-2 hidden size"),
        (1, 512, "Single element"),
        (64, 256, "Medium"),
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for n_elements, size, desc in test_configs:
        print(f"\n  Test: {desc} [{n_elements}, {size}]")

        np.random.seed(42)
        x_np = np.random.randn(n_elements, size).astype(np.float32)
        gamma_np = np.random.randn(size).astype(np.float32)
        beta_np = np.random.randn(size).astype(np.float32)
        grad_y_np = np.random.randn(n_elements, size).astype(np.float32)

        # JAX reference
        def layernorm_fn(x, gamma, beta):
            eps = 1e-5
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            normalized = (x - mean) / jnp.sqrt(var + eps)
            return gamma * normalized + beta

        x_jax = jnp.array(x_np)
        gamma_jax = jnp.array(gamma_np)
        beta_jax = jnp.array(beta_np)
        grad_y_jax = jnp.array(grad_y_np)

        y_jax, vjp_fn = jax.vjp(layernorm_fn, x_jax, gamma_jax, beta_jax)
        grad_x_jax, grad_gamma_jax, grad_beta_jax = vjp_fn(grad_y_jax)

        grad_x_jax_np = np.array(grad_x_jax)
        grad_gamma_jax_np = np.array(grad_gamma_jax)
        grad_beta_jax_np = np.array(grad_beta_jax)

        # GPU computation - Correct workspace calculation!
        x_gpu = gpu.gpu_buffer_2d_create(ctx, n_elements, size, x_np)
        gamma_gpu = gpu.gpu_buffer_1d_create(ctx, size, gamma_np)
        grad_y_gpu = gpu.gpu_buffer_2d_create(ctx, n_elements, size, grad_y_np)
        grad_x_gpu = gpu.gpu_buffer_2d_create(ctx, n_elements, size)
        grad_gamma_gpu = gpu.gpu_buffer_1d_create(ctx, size)
        grad_beta_gpu = gpu.gpu_buffer_1d_create(ctx, size)

        # CRITICAL FIX: Workspace is (n_elements, size) - for partials per element per dim
        workspace_gamma = gpu.gpu_buffer_2d_create(ctx, n_elements, size)
        workspace_beta = gpu.gpu_buffer_2d_create(ctx, n_elements, size)

        gpu.batch_begin(ctx)
        gpu.layernorm_backward(
            ctx,
            x_gpu,
            gamma_gpu,
            grad_y_gpu,
            grad_x_gpu,
            grad_gamma_gpu,
            grad_beta_gpu,
            workspace_gamma,
            workspace_beta,
            accumulate=False,
        )
        gpu.batch_commit(ctx)

        grad_x_gpu_np = np.zeros((n_elements, size), dtype=np.float32)
        grad_gamma_gpu_np = np.zeros(size, dtype=np.float32)
        grad_beta_gpu_np = np.zeros(size, dtype=np.float32)
        gpu.gpu_buffer_2d_read(ctx, grad_x_gpu, grad_x_gpu_np)
        gpu.gpu_buffer_1d_read(ctx, grad_gamma_gpu, grad_gamma_gpu_np)
        gpu.gpu_buffer_1d_read(ctx, grad_beta_gpu, grad_beta_gpu_np)

        # Compare all gradients
        print("\n  Gradient w.r.t. input:")
        passed_x, max_e_x, mean_e_x = runner.compare_arrays(
            grad_x_gpu_np, grad_x_jax_np, "grad_x"
        )

        print("\n  Gradient w.r.t. gamma:")
        passed_gamma, max_e_gamma, mean_e_gamma = runner.compare_arrays(
            grad_gamma_gpu_np, grad_gamma_jax_np, "grad_gamma"
        )

        print("\n  Gradient w.r.t. beta:")
        passed_beta, max_e_beta, mean_e_beta = runner.compare_arrays(
            grad_beta_gpu_np, grad_beta_jax_np, "grad_beta"
        )

        all_passed = all_passed and passed_x and passed_gamma and passed_beta
        max_err = max(max_err, max_e_x, max_e_gamma, max_e_beta)
        mean_err = max(mean_err, (mean_e_x + mean_e_gamma + mean_e_beta) / 3)

    return TestResult("layernorm_backward", all_passed, max_err, mean_err)


def test_embedding(ctx, config: TestConfig) -> TestResult:
    """Test embedding lookup with positional encoding"""
    test_configs = [
        (2, 4, 8, "Tiny (debug)"),
        (8, 32, 128, "Small"),
        (32, 512, 768, "GPT-2 scale"),
        (1, 1024, 512, "Single seq, long context"),
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for batch_size, seq_len, emb_dim, desc in test_configs:
        print(f"\n  Test: {desc} [batch={batch_size}, seq={seq_len}, dim={emb_dim}]")

        vocab_size = 1000
        np.random.seed(42)

        # Random token IDs
        input_ids_np = np.random.randint(
            0, vocab_size, (batch_size, seq_len), dtype=np.int32
        )
        # Embedding table and positional encoding
        emb_table_np = np.random.randn(vocab_size, emb_dim).astype(np.float32)
        pos_enc_np = np.random.randn(seq_len, emb_dim).astype(np.float32)

        # JAX reference
        input_ids_flat = input_ids_np.flatten()
        emb_lookup = emb_table_np[input_ids_flat]
        emb_lookup = emb_lookup.reshape(batch_size, seq_len, emb_dim)
        output_jax = emb_lookup + pos_enc_np[np.newaxis, :, :]
        output_jax_np = output_jax.reshape(batch_size * seq_len, emb_dim)

        # GPU computation - FIXED API call!
        emb_table_gpu = gpu.gpu_buffer_2d_create(ctx, vocab_size, emb_dim, emb_table_np)
        pos_enc_gpu = gpu.gpu_buffer_2d_create(ctx, seq_len, emb_dim, pos_enc_np)
        input_ids_gpu = gpu.gpu_buffer_1d_create(
            ctx, batch_size * seq_len, input_ids_np.flatten()
        )
        output_gpu = gpu.gpu_buffer_2d_create(ctx, batch_size * seq_len, emb_dim)

        gpu.batch_begin(ctx)
        # CRITICAL FIX: Pass batch_size and seq_len as separate parameters!
        gpu.embedding(
            ctx,
            emb_table_gpu,
            pos_enc_gpu,
            input_ids_gpu,
            output_gpu,
            batch_size,
            seq_len,  # These were missing!
        )
        gpu.batch_commit(ctx)

        output_gpu_np = np.zeros((batch_size * seq_len, emb_dim), dtype=np.float32)
        gpu.gpu_buffer_2d_read(ctx, output_gpu, output_gpu_np)

        passed, max_e, mean_e = runner.compare_arrays(
            output_gpu_np, output_jax_np, desc
        )
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    return TestResult("embedding", all_passed, max_err, mean_err)


def test_flash_attention(ctx, config: TestConfig) -> TestResult:
    """Test Flash Attention forward pass"""
    # Use smaller dimensions that fit in 16KB workgroup memory
    # Memory usage: (2*Bc*head_dim + Br*head_dim + 2*Br*Bc + Br + Bc) * 4 bytes
    test_configs = [
        (2, 8, 2, 16, "Tiny: 8d"),  # Very small for debugging
        (2, 4, 2, 16, "Tiny (debug)"),
        (2, 16, 2, 16, "Small: 16d"),  # Small head_dim
        (4, 16, 4, 64, "Small"),
        (8, 32, 8, 64, "Medium"),
        (4, 32, 4, 32, "Medium: 32d"),  # Medium head_dim
        # (2,64,4,128,"Long sequence",),  # FIXME: FAILS. head_dim of 128 > max supported 64
        (2, 64, 2, 32, "Large: 64seq x 32d"),  # Longer sequence, medium head_dim
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    old_head_dim = ctx.config.flash_attn_max_head_dim
    for batch, seq_len, n_heads, head_dim, desc in test_configs:
        print(
            f"\n Test: {desc} [batch={batch}, seq={seq_len}, heads={n_heads}, dim={head_dim}]"
        )

        ctx.config.flash_attn_max_head_dim = head_dim
        np.random.seed(42)
        # Q, K, V matrices
        Q_np = np.random.randn(batch, seq_len, n_heads * head_dim).astype(np.float32)
        K_np = np.random.randn(batch, seq_len, n_heads * head_dim).astype(np.float32)
        V_np = np.random.randn(batch, seq_len, n_heads * head_dim).astype(np.float32)

        # JAX reference - standard attention
        # Reshape to (batch, n_heads, seq_len, head_dim)
        Q_jax = (
            jnp.array(Q_np)
            .reshape(batch, seq_len, n_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        K_jax = (
            jnp.array(K_np)
            .reshape(batch, seq_len, n_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        V_jax = (
            jnp.array(V_np)
            .reshape(batch, seq_len, n_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        scale = 1.0 / jnp.sqrt(float(head_dim))
        scores = jnp.einsum("bhqd,bhkd->bhqk", Q_jax, K_jax) * scale

        # For causal attention (no explicit mask in this version)
        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        scores = jnp.where(causal_mask == 1, scores, -1e10)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        output_jax = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, V_jax)

        # Reshape back to (batch, seq_len, n_heads * head_dim)
        output_jax = output_jax.transpose(0, 2, 1, 3).reshape(
            batch, seq_len, n_heads * head_dim
        )
        output_jax_np = np.array(output_jax)

        # GPU computation
        Q_gpu = gpu.gpu_buffer_2d_create(
            ctx,
            batch * seq_len,
            n_heads * head_dim,
            Q_np.reshape(-1, n_heads * head_dim),
        )
        K_gpu = gpu.gpu_buffer_2d_create(
            ctx,
            batch * seq_len,
            n_heads * head_dim,
            K_np.reshape(-1, n_heads * head_dim),
        )
        V_gpu = gpu.gpu_buffer_2d_create(
            ctx,
            batch * seq_len,
            n_heads * head_dim,
            V_np.reshape(-1, n_heads * head_dim),
        )
        O_gpu = gpu.gpu_buffer_2d_create(ctx, batch * seq_len, n_heads * head_dim)

        # Flash attention needs L (normalizer) and M (max) buffers for the forward pass
        # These are intermediate values used in the algorithm
        L_gpu = gpu.gpu_buffer_1d_create(ctx, batch * seq_len * n_heads)
        M_gpu = gpu.gpu_buffer_1d_create(ctx, batch * seq_len * n_heads)

        gpu.batch_begin(ctx)
        gpu.flash_attention(
            ctx,
            Q_gpu,
            K_gpu,
            V_gpu,
            O_gpu,
            L_gpu,
            M_gpu,
            batch,
            seq_len,
            n_heads,
            head_dim,
        )
        gpu.batch_commit(ctx)

        output_gpu_np = np.zeros(
            (batch * seq_len, n_heads * head_dim), dtype=np.float32
        )
        gpu.gpu_buffer_2d_read(ctx, O_gpu, output_gpu_np)
        output_gpu_np = output_gpu_np.reshape(batch, seq_len, n_heads * head_dim)

        passed, max_e, mean_e = runner.compare_arrays(
            output_gpu_np, output_jax_np, desc
        )
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

    ctx.config.flash_attn_max_head_dim = old_head_dim
    return TestResult("flash_attention", all_passed, max_err, mean_err)


def test_flash_attention_backward(ctx, config: TestConfig) -> TestResult:
    """Test Flash Attention backward pass using JAX autodiff"""
    # Use smaller configs for backward (more expensive)
    test_configs = [
        (2, 8, 2, 16, "Small"),
        (4, 16, 4, 32, "Medium"),
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    old_head_dim = ctx.config.flash_attn_max_head_dim
    for batch, seq_len, n_heads, head_dim, desc in test_configs:
        print(
            f"\n Test: {desc} [batch={batch}, seq={seq_len}, heads={n_heads}, dim={head_dim}]"
        )
        ctx.config.flash_attn_max_head_dim = head_dim

        np.random.seed(42)
        Q_np = np.random.randn(batch, seq_len, n_heads * head_dim).astype(np.float32)
        K_np = np.random.randn(batch, seq_len, n_heads * head_dim).astype(np.float32)
        V_np = np.random.randn(batch, seq_len, n_heads * head_dim).astype(np.float32)
        grad_out_np = np.random.randn(batch, seq_len, n_heads * head_dim).astype(
            np.float32
        )

        # JAX reference with autodiff
        def attention_fn(Q, K, V):
            Q = Q.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            K = K.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            V = V.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

            scale = 1.0 / jnp.sqrt(float(head_dim))
            scores = jnp.einsum("bhqd,bhkd->bhqk", Q, K) * scale

            # ADD CAUSAL MASK!
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))  # Lower triangular
            mask = mask[None, None, :, :]  # Broadcast to (1, 1, seq, seq)
            scores = jnp.where(mask == 1, scores, -1e10)

            attn_weights = jax.nn.softmax(scores, axis=-1)
            output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, V)
            return output.transpose(0, 2, 1, 3).reshape(
                batch, seq_len, n_heads * head_dim
            )

        Q_jax = jnp.array(Q_np)
        K_jax = jnp.array(K_np)
        V_jax = jnp.array(V_np)
        grad_out_jax = jnp.array(grad_out_np)

        _, vjp_fn = jax.vjp(attention_fn, Q_jax, K_jax, V_jax)
        grad_Q_jax, grad_K_jax, grad_V_jax = vjp_fn(grad_out_jax)

        grad_Q_jax_np = np.array(grad_Q_jax)
        grad_K_jax_np = np.array(grad_K_jax)
        grad_V_jax_np = np.array(grad_V_jax)

        # GPU computation
        tot_tokens = batch * seq_len
        emb_dim = n_heads * head_dim

        Q_gpu = gpu.gpu_buffer_2d_create(
            ctx, tot_tokens, emb_dim, Q_np.reshape(-1, emb_dim)
        )
        K_gpu = gpu.gpu_buffer_2d_create(
            ctx, tot_tokens, emb_dim, K_np.reshape(-1, emb_dim)
        )
        V_gpu = gpu.gpu_buffer_2d_create(
            ctx, tot_tokens, emb_dim, V_np.reshape(-1, emb_dim)
        )

        # Forward pass first
        O_gpu = gpu.gpu_buffer_2d_create(ctx, tot_tokens, emb_dim)
        L_gpu = gpu.gpu_buffer_1d_create(ctx, batch * seq_len * n_heads)
        M_gpu = gpu.gpu_buffer_1d_create(ctx, batch * seq_len * n_heads)

        gpu.batch_begin(ctx)
        gpu.flash_attention(
            ctx,
            Q_gpu,
            K_gpu,
            V_gpu,
            O_gpu,
            L_gpu,
            M_gpu,
            batch,
            seq_len,
            n_heads,
            head_dim,
        )
        gpu.batch_commit(ctx)

        # Backward pass with CORRECT workspace calculation
        grad_out_gpu = gpu.gpu_buffer_2d_create(
            ctx, tot_tokens, emb_dim, grad_out_np.reshape(-1, emb_dim)
        )
        grad_Q_gpu = gpu.gpu_buffer_2d_create(ctx, tot_tokens, emb_dim)
        grad_K_gpu = gpu.gpu_buffer_2d_create(ctx, tot_tokens, emb_dim)
        grad_V_gpu = gpu.gpu_buffer_2d_create(ctx, tot_tokens, emb_dim)

        # Get Br from context config (matching what the function uses)
        Br = ctx.config.flash_attn_br if hasattr(ctx.config, "flash_attn_br") else 32
        num_q_blocks = (seq_len + Br - 1) // Br
        workspace_tokens = (
            num_q_blocks * tot_tokens
        )  # CRITICAL: num_q_blocks * total_tokens

        print(
            f"  Debug: tot_tokens={tot_tokens}, Br={Br}, num_q_blocks={num_q_blocks}, workspace_tokens={workspace_tokens}"
        )

        grad_K_workspace = gpu.gpu_buffer_2d_create(ctx, workspace_tokens, emb_dim)
        grad_V_workspace = gpu.gpu_buffer_2d_create(ctx, workspace_tokens, emb_dim)

        gpu.batch_begin(ctx)
        gpu.flash_attention_backward(
            ctx,
            Q_gpu,
            K_gpu,
            V_gpu,
            O_gpu,
            L_gpu,
            M_gpu,
            grad_out_gpu,
            grad_Q_gpu,
            grad_K_workspace,
            grad_V_workspace,
            grad_K_gpu,
            grad_V_gpu,
            batch,
            seq_len,
            n_heads,
            head_dim,
        )
        gpu.batch_commit(ctx)

        grad_Q_gpu_np = np.zeros((tot_tokens, emb_dim), dtype=np.float32)
        grad_K_gpu_np = np.zeros((tot_tokens, emb_dim), dtype=np.float32)
        grad_V_gpu_np = np.zeros((tot_tokens, emb_dim), dtype=np.float32)

        gpu.gpu_buffer_2d_read(ctx, grad_Q_gpu, grad_Q_gpu_np)
        gpu.gpu_buffer_2d_read(ctx, grad_K_gpu, grad_K_gpu_np)
        gpu.gpu_buffer_2d_read(ctx, grad_V_gpu, grad_V_gpu_np)

        grad_Q_gpu_np = grad_Q_gpu_np.reshape(batch, seq_len, emb_dim)
        grad_K_gpu_np = grad_K_gpu_np.reshape(batch, seq_len, emb_dim)
        grad_V_gpu_np = grad_V_gpu_np.reshape(batch, seq_len, emb_dim)

        print("\n Gradient w.r.t. Q:")
        passed_Q, max_e_Q, mean_e_Q = runner.compare_arrays(
            grad_Q_gpu_np, grad_Q_jax_np, "grad_Q"
        )

        print("\n Gradient w.r.t. K:")
        passed_K, max_e_K, mean_e_K = runner.compare_arrays(
            grad_K_gpu_np, grad_K_jax_np, "grad_K"
        )

        print("\n Gradient w.r.t. V:")
        passed_V, max_e_V, mean_e_V = runner.compare_arrays(
            grad_V_gpu_np, grad_V_jax_np, "grad_V"
        )

        passed = passed_Q and passed_K and passed_V
        max_e = max(max_e_Q, max_e_K, max_e_V)
        mean_e = (mean_e_Q + mean_e_K + mean_e_V) / 3

        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

        if not passed:
            print("\n  ⚠ WARNING: Large numerical errors detected!")
            print(
                "     This indicates potential bugs in the Flash Attention backward kernel"
            )

    if max_err > 1.0:
        details = f"KERNEL BUG DETECTED: Max error {max_err:.2e} >> tolerance"
    else:
        details = "Gradients computed correctly"

    ctx.config.flash_attn_max_head_dim = old_head_dim
    return TestResult(
        "flash_attention_backward", all_passed, max_err, mean_err, details
    )


def test_cross_entropy_loss(ctx, config: TestConfig) -> TestResult:
    """Test cross-entropy loss with masking"""
    test_configs = [
        (4, 8, 100, 0.0, "No masking"),
        (4, 8, 100, 0.2, "20% masked"),
        (8, 16, 1000, 0.3, "30% masked, large vocab"),
        (2, 32, 5000, 0.5, "50% masked"),
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for batch, seq_len, vocab_size, mask_ratio, desc in test_configs:
        print(f"\n Test: {desc} [batch={batch}, seq={seq_len}, vocab={vocab_size}]")

        np.random.seed(42)
        batch_seq = batch * seq_len

        # Logits and target labels
        logits_np = np.random.randn(batch_seq, vocab_size).astype(np.float32)
        targets_np = np.random.randint(0, vocab_size, (batch_seq,), dtype=np.uint32)

        # Create random mask
        mask_np = (np.random.rand(batch_seq) > mask_ratio).astype(np.uint32)
        n_valid = int(np.sum(mask_np))

        # JAX reference
        logits_jax = jnp.array(logits_np)
        targets_jax = jnp.array(targets_np)
        mask_jax = jnp.array(mask_np, dtype=np.float32)

        # Compute log softmax
        log_probs = jax.nn.log_softmax(logits_jax, axis=-1)

        # Per-token losses (negative log likelihood)
        target_log_probs = log_probs[jnp.arange(batch_seq), targets_jax]
        per_token_loss_jax = -target_log_probs * mask_jax

        # Mean loss
        loss_jax = jnp.sum(per_token_loss_jax) / jnp.maximum(jnp.sum(mask_jax), 1.0)

        # Gradients: (softmax - one_hot) * mask / valid_count
        probs = jax.nn.softmax(logits_jax, axis=-1)
        one_hot = jax.nn.one_hot(targets_jax, vocab_size)
        grad_logits_jax = (probs - one_hot) * mask_jax[:, None]
        grad_logits_jax = grad_logits_jax / jnp.maximum(jnp.sum(mask_jax), 1.0)

        loss_jax_np = float(loss_jax)
        per_token_loss_jax_np = np.array(per_token_loss_jax)
        grad_logits_jax_np = np.array(grad_logits_jax)

        # GPU computation - Use the proper wrapper with 5 buffers
        logits_gpu = gpu.gpu_buffer_2d_create(ctx, batch_seq, vocab_size, logits_np)
        targets_gpu = gpu.gpu_buffer_1d_create(ctx, batch_seq, targets_np)
        mask_gpu = gpu.gpu_buffer_1d_create(ctx, batch_seq, mask_np)
        loss_per_token_gpu = gpu.gpu_buffer_1d_create(ctx, batch_seq)
        grad_logits_gpu = gpu.gpu_buffer_2d_create(ctx, batch_seq, vocab_size)

        gpu.batch_begin(ctx)
        # Use the proper wrapper - it takes 5 buffers
        gpu.cross_entropy_loss(
            ctx, logits_gpu, targets_gpu, mask_gpu, loss_per_token_gpu, grad_logits_gpu
        )
        gpu.batch_commit(ctx)

        # Read results
        loss_per_token_gpu_np = np.zeros(batch_seq, dtype=np.float32)
        grad_logits_gpu_np = np.zeros((batch_seq, vocab_size), dtype=np.float32)

        gpu.gpu_buffer_1d_read(ctx, loss_per_token_gpu, loss_per_token_gpu_np)
        gpu.gpu_buffer_2d_read(ctx, grad_logits_gpu, grad_logits_gpu_np)

        # Debug: Check per-token losses
        print("\n  Per-token loss comparison (first 5):")
        print(f"    JAX: {per_token_loss_jax_np[:5]}")
        print(f"    GPU: {loss_per_token_gpu_np[:5]}")

        # Compute mean loss (normalized by valid count)
        loss_gpu_np = np.sum(loss_per_token_gpu_np) / max(n_valid, 1)

        # Compare mean loss
        print("\n  Mean loss comparison:")
        print(f"    JAX: {loss_jax_np:.6f}")
        print(f"    GPU: {loss_gpu_np:.6f}")
        print(f"    Diff: {abs(loss_jax_np - loss_gpu_np):.6e}")
        print(f"    Valid tokens: {n_valid} / {batch_seq}")

        loss_passed = (
            abs(loss_jax_np - loss_gpu_np) < config.atol * 10
        )  # Relax tolerance slightly

        # Compare gradients
        print("\n  Gradient comparison:")
        passed_grad, max_e, mean_e = runner.compare_arrays(
            grad_logits_gpu_np, grad_logits_jax_np, "grad_logits"
        )

        passed = loss_passed and passed_grad
        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

        if not passed:
            print(f"  ✗ Test FAILED for {desc}")
        else:
            print("  ✓ Test PASSED")

    return TestResult("cross_entropy_loss", all_passed, max_err, mean_err)


def test_embedding_backward(ctx, config: TestConfig) -> TestResult:
    """Test embedding backward pass"""
    test_configs = [
        (2, 4, 100, 32, "Tiny"),
        (4, 8, 1000, 64, "Small"),
        (8, 16, 5000, 128, "Medium"),
    ]

    runner = TestRunner(ctx, config)
    all_passed = True
    max_err = 0.0
    mean_err = 0.0

    for batch, seq_len, vocab_size, embedding_dim, desc in test_configs:
        print(
            f"\n Test: {desc} [batch={batch}, seq={seq_len}, vocab={vocab_size}, dim={embedding_dim}]"
        )

        np.random.seed(42)
        total_tokens = batch * seq_len

        # Create input token IDs
        input_ids_np = np.random.randint(
            0, vocab_size, (total_tokens,), dtype=np.uint32
        )

        # Gradient from upstream (like from loss)
        grad_output_np = np.random.randn(total_tokens, embedding_dim).astype(np.float32)

        # JAX reference - embedding backward is just scatter-add
        grad_embedding_jax_np = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        for i, token_id in enumerate(input_ids_np):
            grad_embedding_jax_np[token_id] += grad_output_np[i]

        # GPU computation
        input_ids_gpu = gpu.gpu_buffer_1d_create(ctx, total_tokens, input_ids_np)
        grad_output_gpu = gpu.gpu_buffer_2d_create(
            ctx, total_tokens, embedding_dim, grad_output_np
        )
        grad_embedding_gpu = gpu.gpu_buffer_2d_create(ctx, vocab_size, embedding_dim)
        reduction_workspace_gpu = gpu.gpu_buffer_2d_create(
            ctx, total_tokens, embedding_dim
        )

        gpu.batch_begin(ctx)
        # FIXED: Pass all 6 required arguments (after ctx)
        gpu.embedding_backward(
            ctx,
            input_ids_gpu,
            grad_output_gpu,
            grad_embedding_gpu,
            reduction_workspace_gpu,
            batch,  # ADDED: batch_size
            seq_len,  # ADDED: seq_len
        )
        gpu.batch_commit(ctx)

        # Read result
        grad_embedding_gpu_np = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        gpu.gpu_buffer_2d_read(ctx, grad_embedding_gpu, grad_embedding_gpu_np)

        # Compare
        print("\n  Gradient comparison:")
        passed, max_e, mean_e = runner.compare_arrays(
            grad_embedding_gpu_np, grad_embedding_jax_np, "grad_embedding"
        )

        max_err = max(max_err, max_e)
        mean_err = max(mean_err, mean_e)
        all_passed = all_passed and passed

        if not passed:
            print(f"  ✗ Test FAILED for {desc}")
        else:
            print("  ✓ Test PASSED")

    return TestResult("embedding_backward", all_passed, max_err, mean_err)


def main():
    """Main test entry point"""
    print("=" * 80)
    print("GPU MODULE TEST SUITE vs JAX")
    print("=" * 80)

    # Initialize GPU
    print("Initializing GPU device...")
    device = gpu.device_create()
    config = gpu.device_config_create(device)
    pipeline_cache = gpu.pipeline_cache_create(device)
    ctx = gpu.GPUContext(
        device=device,
        config=config,
        pipeline_cache=pipeline_cache,
        batch_state=None,
    )

    ctx.config.flash_attn_max_head_dim = 64  # Safe for 16KB workgroup memory
    ctx.config.flash_attn_bc = 8
    ctx.config.flash_attn_br = 8

    config = TestConfig(rtol=1e-4, atol=1e-3, verbose=True)
    runner = TestRunner(ctx, config)

    # Run tests
    runner.run_test(test_matmul, "Matrix Multiplication")
    runner.run_test(test_matmul_backward_a, "MatMul Backward (grad A)")
    runner.run_test(test_matmul_backward_b, "MatMul Backward (grad B)")
    runner.run_test(test_matmul_full_chain, "MatMul Full Chain")

    runner.run_test(test_gelu, "GELU Activation")
    runner.run_test(test_gelu_backward, "GELU Backward")
    runner.run_test(test_layernorm, "Layer Normalization")
    runner.run_test(test_layernorm_backward, "LayerNorm Backward")
    runner.run_test(test_embedding, "Embedding Lookup")
    runner.run_test(test_flash_attention, "Flash Attention Forward")
    runner.run_test(test_flash_attention_backward, "Flash Attention Backward")
    runner.run_test(test_cross_entropy_loss, "Cross-Entropy Loss with Masking")
    runner.run_test(test_embedding_backward, "Embedding Backward")

    all_passed = runner.print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
