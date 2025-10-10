"""Performance monitoring functions"""

# Module-level state
_perf_monitor = None


def create_perf_monitor():
    """Create performance monitor state"""
    return {"kernel_times": {}, "memory_usage": {}, "submission_count": 0}


def record_kernel_time(monitor, kernel_name, duration_ms):
    """Record kernel execution time"""
    if kernel_name not in monitor["kernel_times"]:
        monitor["kernel_times"][kernel_name] = []
    monitor["kernel_times"][kernel_name].append(duration_ms)
    return monitor


def record_submission(monitor):
    """Increment submission counter"""
    monitor["submission_count"] += 1
    return monitor


def get_perf_stats(monitor):
    """Get performance statistics"""
    stats = {"total_submissions": monitor["submission_count"], "kernel_times": {}}

    for kernel_name, times in monitor["kernel_times"].items():
        stats["kernel_times"][kernel_name] = {
            "count": len(times),
            "total_ms": sum(times),
            "avg_ms": sum(times) / len(times) if times else 0,
            "min_ms": min(times) if times else 0,
            "max_ms": max(times) if times else 0,
        }

    return stats


def reset_perf_monitor(monitor):
    """Reset all counters"""
    monitor["kernel_times"].clear()
    monitor["memory_usage"].clear()
    monitor["submission_count"] = 0
    return monitor


def get_performance_monitor():
    """Get global performance monitor"""
    global _perf_monitor
    if _perf_monitor is None:
        _perf_monitor = create_perf_monitor()
    return _perf_monitor
