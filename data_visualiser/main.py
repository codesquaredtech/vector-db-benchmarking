from app.logger import get_logger
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_FOLDER = "results"
PLOTS_FOLDER = "plots"


def retrieve_benchmarker_files(type: str):
    file_paths = [
        os.path.join(RESULTS_FOLDER, f)
        for f in os.listdir(RESULTS_FOLDER)
        if type in f and f.endswith(".csv")
    ]
    file_paths.sort()
    return file_paths


def plot_metrics_scatterplot(
    search_file_paths,
    metric="search_time",
    title=None,
    ylabel=None,
    file_name=None,
):
    """
    Create a grid of scatterplots with a specified metric for all databases.
    The grid will have 3 rows and 2 columns, and the Y-axis will be the same for all plots.

    Parameters:
    - search_file_paths: List of paths to CSV files.
    - metric: The metric to plot (e.g., 'search_time', 'rps', etc.). Default is 'search_time'.
    - title: Title of the plot (optional). If not provided, a default title will be generated.
    - ylabel: Label for the y-axis (optional). If not provided, the metric name will be used.
    - file_name: Custom file name for saving the plot (optional). If not provided, a default name will be generated.
    """

    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharey=True)
    axes = axes.flatten()

    colors = sns.color_palette("husl", n_colors=len(search_file_paths))

    for i, file_path in enumerate(search_file_paths):
        logger.info(f"Processing {file_path}...")
        df = pd.read_csv(file_path)

        df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
        df_valid = df.dropna(subset=["iteration"])

        db_name = os.path.basename(file_path).split("_database_")[-1].split("_")[0]

        ax = axes[i]
        sns.scatterplot(
            x=df_valid["iteration"],
            y=df_valid[metric],
            color=colors[i],
            alpha=0.7,
            ax=ax,
        )

        ax.plot(
            df_valid["iteration"],
            df_valid[metric],
            color=colors[i],
            linewidth=4,
            alpha=0.8,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel if ylabel else metric)
        ax.set_title(f"{db_name} - {metric.capitalize()}")
        ax.grid(True)
        ax.yaxis.set_visible(True)

    fig.suptitle(
        title
        if title
        else f"{metric.capitalize()} Across Iterations for All Databases",
        fontsize=16,
    )

    plt.tight_layout()

    if file_name is None:
        file_name = f"{metric}_combined_scatterplot_grid.jpg"
    plot_path = os.path.join(PLOTS_FOLDER, file_name)

    fig.savefig(plot_path, dpi=300)

    logger.info(f"Saved combined scatter plot grid: {plot_path}")


def plot_metrics_barplot(
    search_file_paths,
    metrics,
    column_name="search_time",
    colors=None,
    title="Metrics by Database",
    ylabel="Value",
    rotation=45,
    file_name=None,
):
    """
    Generalised function to plot bar plots for multiple metrics for each database.

    Parameters:
    - search_file_paths: List of paths to CSV files.
    - metrics: List of metrics to plot (e.g., ['mean', 'std'] or ['p90', 'p95', 'p99']).
    - column_name: Name of the column in the CSV file containing the metric values (default: 'search_time').
    - colors: List of colors for each metric (optional). If not provided, default colors will be used.
    - title: Title of the plot (optional).
    - ylabel: Label for the y-axis (optional).
    - rotation: Rotation angle for x-axis labels (optional).
    - file_name: Custom file name for saving the plot (optional). If not provided, a default name will be generated.
    """

    fig, ax = plt.subplots(figsize=(15, 8))

    metric_values = {metric: [] for metric in metrics}
    db_names = []

    for file_path in search_file_paths:
        logger.info(f"Processing {file_path}...")
        df = pd.read_csv(file_path)

        df_first_file = pd.read_csv(search_file_paths[0])

        numeric_requests = pd.to_numeric(df_first_file["iteration"], errors="coerce")
        number_of_requests = numeric_requests.max()
        title_with_requests = f"{title} ({int(number_of_requests)} requests)"

        db_name = os.path.basename(file_path).split("_database_")[-1].split("_")[0]
        db_names.append(db_name)

        for metric in metrics:
            logger.info(
                f"full metric value {df.loc[df['iteration'] == metric, column_name].values[0]}"
            )
            metric_value = df.loc[df["iteration"] == metric, column_name].values[0]
            metric_values[metric].append(metric_value)

    x = range(len(db_names))
    width = 0.8 / len(metrics)

    if colors is None:
        colors = plt.cm.tab20.colors[: len(metrics)]

    for i, metric in enumerate(metrics):
        bars = ax.bar(
            [pos + i * width for pos in x],
            metric_values[metric],
            width,
            label=metric,
            color=colors[i],
            alpha=0.7,
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Database")
    ax.set_ylabel(ylabel)
    ax.set_title(title_with_requests)
    ax.set_xticks([pos + (len(metrics) - 1) * width / 2 for pos in x])
    ax.set_xticklabels(db_names, rotation=rotation)
    ax.legend()

    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if file_name is None:
        file_name = f"{'_'.join(metrics)}_barplot.jpg"
    plot_path = os.path.join(PLOTS_FOLDER, file_name)

    fig.savefig(plot_path, dpi=300)

    logger.info(f"Saved bar plot: {plot_path}")


if __name__ == "__main__":
    # TODO: Present precision, recall, f1 values.
    logger = get_logger()

    search_file_paths = retrieve_benchmarker_files("search")
    store_delete_file_paths = retrieve_benchmarker_files("storing_and_deletion")

    # Scatterplot for search times
    plot_metrics_scatterplot(
        search_file_paths,
        metric="search_time",
        title="Search Time Across Requests",
        ylabel="Search Time (s)",
        file_name="search_time_combined_scatter_plot.jpg",
    )

    # Scatterplot for initialisation times
    plot_metrics_scatterplot(
        store_delete_file_paths,
        metric="initialisation_time",
        title="Initialisation Time Across Requests",
        ylabel="Initialisation Time (s)",
        file_name="initialisation_time_combined_scatter_plot.jpg",
    )

    # Scatterplot for insertion times
    plot_metrics_scatterplot(
        store_delete_file_paths,
        metric="insertion_time",
        title="Insertion Time Across Requests",
        ylabel="Insertion Time (s)",
        file_name="insertion_time_combined_scatter_plot.jpg",
    )

    # Scatterplot for deletion times
    plot_metrics_scatterplot(
        store_delete_file_paths,
        metric="deletion_time",
        title="Deletion Time Across Requests",
        ylabel="Deletion Time (s)",
        file_name="deletion_time_combined_scatter_plot.jpg",
    )

    # Scatterplot for memory usage when initialising database
    plot_metrics_scatterplot(
        store_delete_file_paths,
        metric="memory_usage_initialisation",
        title="Memory Usage for Database Initialisation Across Requests",
        ylabel="Memory Usage (MB)",
        file_name="memory_usage_initialisation_combined_scatter_plot.jpg",
    )

    # Scatterplot for memory usage when inserting into database
    plot_metrics_scatterplot(
        store_delete_file_paths,
        metric="memory_usage_insertion",
        title="Memory Usage for Database Insertion Across Requests",
        ylabel="Memory Usage (MB)",
        file_name="memory_usage_insertion_combined_scatter_plot.jpg",
    )

    # Barplot for mean and std search times
    plot_metrics_barplot(
        search_file_paths,
        metrics=["search_time_mean", "search_time_std"],
        column_name="search_time",
        title="Search Mean and Standard Deviation by Database",
        ylabel="Search Time (s)",
        file_name="search_mean_std_barplot.jpg",
    )

    # Barplot for p90, p95, and p99 search times
    plot_metrics_barplot(
        search_file_paths,
        metrics=["search_time_p90", "search_time_p95", "search_time_p99"],
        column_name="search_time",
        title="Search Time Percentiles by Database",
        ylabel="Search Time (s)",
        file_name="search_percentiles_barplot.jpg",
    )

    # Barplot for RPS and successful requests when searching
    plot_metrics_barplot(
        search_file_paths,
        metrics=["rps", "successful_requests"],
        column_name="search_time",
        title="RPS and Successful Requests by Database",
        ylabel="Value",
        file_name="search_rps_successful_requests_barplot.jpg",
    )

    # Barplot for mean and std initialisation times
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["initialisation_mean", "initialisation_std"],
        column_name="initialisation_time",
        title="Insertion Mean and Standard Deviation by Database",
        ylabel="Initialisation Time (s)",
        file_name="initialisation_mean_std_barplot.jpg",
    )

    # Barplot for mean and std insertion times
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["insertion_mean", "insertion_std"],
        column_name="initialisation_time",
        title="Insertion Mean and Standard Deviation by Database",
        ylabel="Insertion Time (s)",
        file_name="insertion_mean_std_barplot.jpg",
    )

    # Barplot for mean and std deletion times
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["deletion_mean", "deletion_std"],
        column_name="initialisation_time",
        title="Deletion Mean and Standard Deviation by Database",
        ylabel="Deletion Time (s)",
        file_name="deletion_mean_std_barplot.jpg",
    )

    # Barplot for p90, p95, and p99 initialisation times
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["initialisation_p90", "initialisation_p95", "initialisation_p99"],
        column_name="initialisation_time",
        title="Initialisation Time Percentiles by Database",
        ylabel="Initialisation Time (s)",
        file_name="initialisation_percentiles_barplot.jpg",
    )

    # Barplot for p90, p95, and p99 insertion times
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["insertion_p90", "insertion_p95", "insertion_p99"],
        column_name="initialisation_time",
        title="Insertion Time Percentiles by Database",
        ylabel="Insertion Time (s)",
        file_name="insertion_percentiles_barplot.jpg",
    )

    # Barplot for p90, p95, and p99 deletion times
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["deletion_p90", "deletion_p95", "deletion_p99"],
        column_name="initialisation_time",
        title="Deletion Time Percentiles by Database",
        ylabel="Deletion Time (s)",
        file_name="deletion_percentiles_barplot.jpg",
    )

    # Barplot for mean and std memory occupation values when initialising databases
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["memory_usage_init_mean", "memory_usage_init_std"],
        column_name="initialisation_time",
        title="Memory Occupation (MB) Mean and Standard Deviation by Database Initialisation",
        ylabel="Memory Occupation (MB)",
        file_name="memory_occupation_initialisation_mean_std_barplot.jpg",
    )

    # Barplot for mean and std memory occupation values when inserting into databases
    plot_metrics_barplot(
        store_delete_file_paths,
        metrics=["memory_usage_insert_mean", "memory_usage_insert_std"],
        column_name="initialisation_time",
        title="Memory Occupation (MB) Mean and Standard Deviation by Database Insertion",
        ylabel="Memory Occupation (MB)",
        file_name="memory_occupation_insertion_mean_std_barplot.jpg",
    )
