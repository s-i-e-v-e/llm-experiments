"""Performance monitoring functions"""

from gpu_types import KernelTimeStats, PerfMonitor, PerfStats


def create_perf_monitor() -> PerfMonitor:
    """Create performance monitor state"""
    return PerfMonitor()


def record_kernel_time(
    monitor: PerfMonitor, kernel_name: str, duration_ms: float
) -> PerfMonitor:
    """Record kernel execution time"""
    if kernel_name not in monitor.kernel_times:
        monitor.kernel_times[kernel_name] = []
    monitor.kernel_times[kernel_name].append(duration_ms)
    return monitor


def record_submission(monitor: PerfMonitor) -> PerfMonitor:
    """Increment submission counter"""
    monitor.submission_count += 1
    return monitor


def get_perf_stats(monitor: PerfMonitor) -> PerfStats:
    """Get performance statistics"""
    kernel_stats = {}

    for kernel_name, times in monitor.kernel_times.items():
        kernel_stats[kernel_name] = KernelTimeStats(
            count=len(times),
            total_ms=sum(times),
            avg_ms=sum(times) / len(times) if times else 0,
            min_ms=min(times) if times else 0,
            max_ms=max(times) if times else 0,
        )

    return PerfStats(
        total_submissions=monitor.submission_count, kernel_times=kernel_stats
    )


def reset_perf_monitor(monitor: PerfMonitor) -> PerfMonitor:
    """Reset all counters"""
    monitor.kernel_times.clear()
    monitor.memory_usage.clear()
    monitor.submission_count = 0
    return monitor
