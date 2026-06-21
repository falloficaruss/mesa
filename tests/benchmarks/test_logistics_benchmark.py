"""Smoke tests for the synthetic logistics benchmark."""

from __future__ import annotations

import pickle

from benchmarks.configurations import configurations
from benchmarks.global_benchmark import _log_logistics_deltas, run_experiments, run_model
from benchmarks.logistics_benchmark import (
    LogisticsHubBenchmark,
    LogisticsHubBenchmarkNoEntityIndex,
    LogisticsScenario,
)


def test_logistics_benchmark_smoke_and_serialization():
    """The logistics benchmark should exercise entity indexing and serialize cleanly."""
    scenario = LogisticsScenario(
        rng=42,
        hubs=4,
        crews=12,
        parcels=48,
        reassignments_per_step=12,
        refresh_batch=6,
    )

    model = LogisticsHubBenchmark(scenario=scenario)
    model.run_for(2)

    model.entity_index.assert_invariants()
    model.membership_backend.assert_invariants()

    model.remove_all_agents()
    assert model.membership_backend.as_triplets() == set()
    model.entity_index.assert_invariants()

    results = run_experiments(
        LogisticsHubBenchmark,
        {
            "steps": 1,
            "iterations": 1,
            "replications": 1,
            "scenario": scenario,
        },
    )

    assert set(results) == {
        "init_time_s",
        "run_time_s",
        "peak_init_bytes",
        "peak_run_bytes",
    }
    assert results["init_time_s"] and results["run_time_s"]
    assert results["peak_init_bytes"] and results["peak_run_bytes"]

    payload = pickle.dumps({(LogisticsHubBenchmark, "smoke"): results})
    assert payload


def test_logistics_benchmark_without_entity_index_smoke_and_serialization():
    """The no-index variant should run the same workload and report metrics."""
    scenario = LogisticsScenario(
        rng=42,
        hubs=4,
        crews=12,
        parcels=48,
        reassignments_per_step=12,
        refresh_batch=6,
    )

    model = LogisticsHubBenchmarkNoEntityIndex(scenario=scenario)
    model.run_for(2)

    model.entity_index.assert_invariants()
    model.membership_backend.assert_invariants()

    model.remove_all_agents()
    assert model.membership_backend.as_triplets() == set()
    model.entity_index.assert_invariants()

    results = run_experiments(
        LogisticsHubBenchmarkNoEntityIndex,
        {
            "steps": 1,
            "iterations": 1,
            "replications": 1,
            "scenario": scenario,
        },
    )

    assert set(results) == {
        "init_time_s",
        "run_time_s",
        "peak_init_bytes",
        "peak_run_bytes",
    }
    assert results["init_time_s"] and results["run_time_s"]
    assert results["peak_init_bytes"] and results["peak_run_bytes"]

    payload = pickle.dumps({(LogisticsHubBenchmarkNoEntityIndex, "smoke"): results})
    assert payload


def test_logistics_benchmark_runner_reports_memory_metrics():
    """The benchmark runner should return timing and tracemalloc metrics."""
    scenario = LogisticsScenario(
        rng=7,
        hubs=3,
        crews=8,
        parcels=32,
        reassignments_per_step=8,
        refresh_batch=4,
    )

    result = run_model(LogisticsHubBenchmark, 1, scenario)

    assert result["init_time_s"] >= 0
    assert result["run_time_s"] >= 0
    assert result["peak_init_bytes"] > 0
    assert result["peak_run_bytes"] > 0


def test_logistics_benchmark_config_covers_small_and_huge_modes():
    """Both benchmark variants should be exposed in the shared benchmark matrix."""
    for benchmark_cls in (LogisticsHubBenchmark, LogisticsHubBenchmarkNoEntityIndex):
        assert benchmark_cls in configurations
        assert set(configurations[benchmark_cls]) == {"small", "huge"}


def test_logistics_benchmark_delta_summary_is_printed(capsys):
    """The benchmark runner should print indexed-vs-no-index deltas."""
    indexed_results = {
        "init_time_s": [2.0, 2.0],
        "run_time_s": [3.0, 3.0],
        "peak_init_bytes": [4 * 1024 * 1024, 4 * 1024 * 1024],
        "peak_run_bytes": [6 * 1024 * 1024, 6 * 1024 * 1024],
    }
    no_index_results = {
        "init_time_s": [1.0, 1.0],
        "run_time_s": [2.0, 2.0],
        "peak_init_bytes": [2 * 1024 * 1024, 2 * 1024 * 1024],
        "peak_run_bytes": [5 * 1024 * 1024, 5 * 1024 * 1024],
    }

    _log_logistics_deltas(
        {
            (LogisticsHubBenchmark, "small"): indexed_results,
            (LogisticsHubBenchmarkNoEntityIndex, "small"): no_index_results,
        }
    )

    output = capsys.readouterr().out
    assert "Logistics delta (small, indexed - no-index)" in output
    assert "Init +1.00000 s (+100.0%)" in output
    assert "Peak init +2.00 MiB (+100.0%)" in output
