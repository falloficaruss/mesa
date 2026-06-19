"""Smoke tests for the synthetic logistics benchmark."""

from __future__ import annotations

import pickle

from benchmarks.global_benchmark import run_experiments, run_model
from benchmarks.logistics_benchmark import LogisticsHubBenchmark, LogisticsScenario


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
