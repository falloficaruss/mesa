"""compare timings across 2 benchmarks."""

import pickle

import numpy as np
import pandas as pd

filename1 = "timings_1"
filename2 = "timings_2"

with open(f"{filename1}.pickle", "rb") as handle:
    timings_1 = pickle.load(handle)  # noqa: S301

with open(f"{filename2}.pickle", "rb") as handle:
    timings_2 = pickle.load(handle)  # noqa: S301


def bootstrap_percentage_change_confidence_interval(data1, data2, n=1000):
    """Calculate the percentage change and bootstrap a confidence interval.

    Args:
        data1: benchmark dataset 1
        data2: benchmark dataset 2
        n: bootstrap sample size

    Returns:
        float, median, and lower and upper bound of confidence interval.
    """
    change_samples = []
    for _ in range(n):
        sampled_indices = np.random.choice(
            range(len(data1)), size=len(data1), replace=True
        )
        sampled_data1 = np.array(data1)[sampled_indices]
        sampled_data2 = np.array(data2)[sampled_indices]
        change = 100 * (sampled_data2 - sampled_data1) / sampled_data1
        change_samples.append(np.median(change))
    lower, upper = np.percentile(change_samples, [2.5, 97.5])
    return np.median(change_samples), lower, upper


# DataFrame to store the results
results_df = pd.DataFrame()


def performance_emoji(lower, upper):
    """Function to determine the emoji based on change and confidence interval."""
    if upper < -3:
        return "🟢"  # Emoji for faster performance
    elif lower > 3:
        return "🔴"  # Emoji for slower performance
    else:
        return "🔵"  # Emoji for insignificant change


def extract_series(results, metric, fallback_index=None):
    """Return a metric series from either the new dict format or the legacy tuple."""
    if isinstance(results, dict):
        return results.get(metric)
    if fallback_index is None:
        return None
    return results[fallback_index]


def summary_or_na(series_1, series_2):
    """Return a formatted comparison summary or N/A when the metric is absent."""
    if series_1 is None or series_2 is None:
        return "N/A"

    change, lower, upper = bootstrap_percentage_change_confidence_interval(
        series_1, series_2
    )
    emoji = performance_emoji(lower, upper)
    return f"{emoji} {change:+.1f}% [{lower:+.1f}%, {upper:+.1f}%]"


# Iterate over the models and sizes, perform analysis, and populate the DataFrame
for model, size in timings_1:
    model_name = model.__name__

    init_summary = summary_or_na(
        extract_series(timings_1[(model, size)], "init_time_s", 0),
        extract_series(timings_2[(model, size)], "init_time_s", 0),
    )
    run_summary = summary_or_na(
        extract_series(timings_1[(model, size)], "run_time_s", 1),
        extract_series(timings_2[(model, size)], "run_time_s", 1),
    )
    peak_init_summary = summary_or_na(
        extract_series(timings_1[(model, size)], "peak_init_bytes", None),
        extract_series(timings_2[(model, size)], "peak_init_bytes", None),
    )
    peak_run_summary = summary_or_na(
        extract_series(timings_1[(model, size)], "peak_run_bytes", None),
        extract_series(timings_2[(model, size)], "peak_run_bytes", None),
    )
    # Append results to DataFrame
    row = pd.DataFrame(
        {
            "Model": [model_name],
            "Size": [size],
            "Init time [95% CI]": [init_summary],
            "Run time [95% CI]": [run_summary],
            "Peak init memory [95% CI]": [peak_init_summary],
            "Peak run memory [95% CI]": [peak_run_summary],
        }
    )

    results_df = pd.concat([results_df, row], ignore_index=True)

# Convert DataFrame to markdown with specified alignments
markdown_representation = results_df.to_markdown(index=False, tablefmt="github")

# Display the markdown representation
print(markdown_representation)
