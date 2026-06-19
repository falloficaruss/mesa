"""Runner for global performance benchmarks."""

from __future__ import annotations

import gc
import os
import pickle
import sys
import time
import tracemalloc

_BENCHMARK_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_BENCHMARK_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:  # pragma: no cover - import style depends on execution mode.
    from .configurations import configurations
except ImportError:  # pragma: no cover - script execution path.
    from configurations import configurations


def run_model(model_class, steps, scenario):
    """Run one model instance and capture timing plus peak allocations."""
    gc.collect()
    gc.disable()
    tracemalloc.start()

    model = None
    start_init = time.perf_counter()

    try:
        model = model_class(scenario=scenario)
        end_init = time.perf_counter()
        init_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()

        model.run_for(steps)
        end_run = time.perf_counter()
        run_peak = tracemalloc.get_traced_memory()[1]

        return {
            "init_time_s": end_init - start_init,
            "run_time_s": end_run - end_init,
            "peak_init_bytes": init_peak,
            "peak_run_bytes": run_peak,
        }
    finally:
        try:
            if model is not None:
                model.remove_all_agents()
                model = None
        finally:
            tracemalloc.stop()
            gc.enable()
            gc.collect()


def run_experiments(model_class, config):
    """Run performance benchmarks for one model configuration."""
    init_times = []
    run_times = []
    peak_init_bytes = []
    peak_run_bytes = []

    steps = config["steps"]
    for scenario in config["scenario"].spawn_replications(config["replications"]):
        fastest_init = float("inf")
        fastest_run = float("inf")
        lowest_peak_init = float("inf")
        lowest_peak_run = float("inf")

        for _ in range(3):
            run_model(model_class, steps, scenario)

        for _ in range(config["iterations"]):
            metrics = run_model(model_class, steps, scenario)
            if metrics["init_time_s"] < fastest_init:
                fastest_init = metrics["init_time_s"]
                lowest_peak_init = metrics["peak_init_bytes"]
            if metrics["run_time_s"] < fastest_run:
                fastest_run = metrics["run_time_s"]
                lowest_peak_run = metrics["peak_run_bytes"]

        init_times.append(fastest_init)
        run_times.append(fastest_run)
        peak_init_bytes.append(lowest_peak_init)
        peak_run_bytes.append(lowest_peak_run)

    return {
        "init_time_s": init_times,
        "run_time_s": run_times,
        "peak_init_bytes": peak_init_bytes,
        "peak_run_bytes": peak_run_bytes,
    }


def _mean(values):
    return sum(values) / len(values)


def _bytes_to_mib(value):
    return value / (1024 * 1024)


def main():
    """Run the benchmark suite and persist the results."""
    print(f"{time.strftime('%H:%M:%S', time.localtime())} starting benchmarks.")
    results_dict = {}
    for model, model_config in configurations.items():
        for size, config in model_config.items():
            results = run_experiments(model, config)

            mean_init = _mean(results["init_time_s"])
            mean_run = _mean(results["run_time_s"])
            mean_init_peak = _bytes_to_mib(_mean(results["peak_init_bytes"]))
            mean_run_peak = _bytes_to_mib(_mean(results["peak_run_bytes"]))

            print(
                f"{time.strftime('%H:%M:%S', time.localtime())} "
                f"{model.__name__:<22} ({size}) timings: "
                f"Init {mean_init:.5f} s; Run {mean_run:.4f} s; "
                f"Peak init {mean_init_peak:.2f} MiB; Peak run {mean_run_peak:.2f} MiB"
            )

            results_dict[model, size] = results

    save_name = "timings"
    i = 1
    while os.path.exists(f"{save_name}_{i}.pickle"):
        i += 1

    with open(f"{save_name}_{i}.pickle", "wb") as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done benchmarking. Saved results to {save_name}_{i}.pickle.")


if __name__ == "__main__":
    main()
