import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, gaussian_kde
from sklearn.linear_model import LinearRegression


# =============================================================================
# Section: Error Metrics Calculation
# =============================================================================

def calculate_errors_with_pdf(y_true, y_pred, feature_names=None, kde_points=200):
    """
    Calculate per-sample, per-feature error metrics and estimate the PDF of normalized errors.

    Parameters:
    - y_true: ndarray of shape (n_samples, n_timesteps, n_features)
    - y_pred: ndarray of shape (n_samples, n_timesteps, n_features)
    - feature_names: optional list of feature labels
    - kde_points: number of points to evaluate the KDE

    Returns:
    - metrics_df: DataFrame with mean and std for each metric by feature
    - error_samples: ndarray with shape (n_samples, n_features, n_metrics, 2)
    - error_pdfs: dict mapping feature name to {'x': array, 'pdf': array}
    """
    # Define metric names
    metric_names = ['MSE', 'R2', 'MAE', 'Pearson_r', 'NormError']

    n_samples, n_timesteps, n_features = y_true.shape
    # Initialize array to store [mean, std] for each sample, feature, metric
    error_samples = np.zeros((n_samples, n_features, len(metric_names), 2))
    error_pdfs = {}

    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    # Loop over each sample
    for sample_idx in range(n_samples):
        yt = y_true[sample_idx]
        yp = y_pred[sample_idx]

        # Compute vectorized metrics per feature
        mse_vals = mean_squared_error(yt, yp, multioutput='raw_values')
        mae_vals = mean_absolute_error(yt, yp, multioutput='raw_values')
        r2_vals = r2_score(yt, yp, multioutput='raw_values')

        # Compute Pearson r and normalized error per feature
        for feat_idx in range(n_features):
            true_series = yt[:, feat_idx]
            pred_series = yp[:, feat_idx]
            error_series = true_series - pred_series

            # Pearson correlation coefficient
            pearson_r, _ = pearsonr(true_series, pred_series)

            # Normalized error: divide by max absolute true value
            denom = np.max(np.abs(true_series))
            norm_err = error_series / denom if denom != 0 else error_series
            norm_mean = norm_err.mean()
            norm_std = norm_err.std()

            # Store mean and std for each metric
            # MSE: mean of squared error and std of squared error
            error_samples[sample_idx, feat_idx, 0, 0] = mse_vals[feat_idx]
            error_samples[sample_idx, feat_idx, 0, 1] = np.std(error_series ** 2)

            # R2: mean R2 and no std
            error_samples[sample_idx, feat_idx, 1, 0] = r2_vals[feat_idx]
            error_samples[sample_idx, feat_idx, 1, 1] = 0.0

            # MAE: mean absolute error and std
            error_samples[sample_idx, feat_idx, 2, 0] = mae_vals[feat_idx]
            error_samples[sample_idx, feat_idx, 2, 1] = np.std(np.abs(error_series))

            # Pearson r: value and no std
            error_samples[sample_idx, feat_idx, 3, 0] = pearson_r
            error_samples[sample_idx, feat_idx, 3, 1] = 0.0

            # Normalized error: mean and std
            error_samples[sample_idx, feat_idx, 4, 0] = norm_mean
            error_samples[sample_idx, feat_idx, 4, 1] = norm_std

    # Estimate PDFs of normalized errors across all samples per feature
    for feat_idx in range(n_features):
        aggregated_norm_err = []
        for sample_idx in range(n_samples):
            true_series = y_true[sample_idx, :, feat_idx]
            pred_series = y_pred[sample_idx, :, feat_idx]
            error_series = true_series - pred_series
            denom = np.max(np.abs(true_series))
            norm_err = error_series / denom if denom != 0 else error_series
            aggregated_norm_err.extend(norm_err)

        kde = gaussian_kde(aggregated_norm_err)
        x_vals = np.linspace(-1, 1, kde_points)
        error_pdfs[feature_names[feat_idx]] = {'x': x_vals, 'pdf': kde(x_vals)}

    # Compute mean and std across samples for each metric and feature
    mean_vals = error_samples[..., 0].mean(axis=0)
    std_vals = error_samples[..., 0].std(axis=0)

    # Build a DataFrame with interleaved columns: metric_mean, metric_std
    columns = []
    data_cols = []
    for idx, metric in enumerate(metric_names):
        columns += [f'{metric}_mean', f'{metric}_std']
        data_cols.append(mean_vals[:, idx])
        data_cols.append(std_vals[:, idx])

    metrics_df = pd.DataFrame(np.column_stack(data_cols), columns=columns)
    metrics_df.insert(0, 'Feature', feature_names)

    return metrics_df, error_samples, error_pdfs


# =============================================================================
# Section: Table Formatting
# =============================================================================

def format_metrics_df(df, metric_names=None):
    """
    Combine mean and std columns into formatted strings in a DataFrame.

    Parameters:
    - df: DataFrame with columns '<metric>_mean' and '<metric>_std'
    - metric_names: list of metric base names to format

    Returns:
    - formatted_df: DataFrame with one column per metric, values as 'mean ± std'
    """
    # Copy feature column to result
    result_df = pd.DataFrame({'Feature': df['Feature']})

    # Determine metrics if not given
    if metric_names is None:
        metric_names = [col.replace('_mean', '') for col in df.columns if col.endswith('_mean')]

    for metric in metric_names:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col not in df or std_col not in df:
            raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")

        # Format each row as 'mean ± std'
        result_df[metric] = df.apply(
            lambda row: f"{row[mean_col]:.6f} ± {row[std_col]:.6f}", axis=1
        )

    return result_df


# =============================================================================
# Section: Visualization Functions
# =============================================================================

def plot_error_pdfs(error_pdfs, ci_threshold=0.10, xlim_percent=(-100, 100)):
    """
    Plot PDFs of normalized errors for each feature, highlighting the
    area within ±ci_threshold.
    """
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'magenta', 'green', 'orange', 'purple', 'cyan']
    linestyles = ['-', '--', ':', '-.', '-', '--', ':']

    for idx, (feat_name, data) in enumerate(error_pdfs.items()):
        x = data['x'] * 100  # convert to percent
        pdf = data['pdf']

        # Compute coverage in ±threshold
        mask = (data['x'] >= -ci_threshold) & (data['x'] <= ci_threshold)
        ci_area = np.trapz(pdf[mask], data['x'][mask])
        ci_pct = int(round(ci_area * 100))

        plt.plot(x, pdf,
                 linestyles[idx % len(linestyles)],
                 color=colors[idx % len(colors)],
                 label=f"{feat_name} (CI ≈ {ci_pct}%)")

    plt.axvline(-ci_threshold * 100, color='k', linestyle='-.')
    plt.axvline(ci_threshold * 100, color='k', linestyle='-.')

    plt.title("Normalized Error Distribution by Feature")
    plt.xlabel("Normalized Error (%)")
    plt.ylabel("Probability Density Function")
    plt.xlim(xlim_percent)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_regression_coeff_histogram(y_ref, y_pred, feature_indices=[0],
                                    feature_names=None, bins=20,
                                    invert_x=True, xlim=None):
    """
    Plot histograms of linear regression coefficients between y_ref and y_pred
    for each specified feature across all samples.
    """
    n_feats = len(feature_indices)
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5), sharey=True)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for ax, feat_idx in zip(axes, feature_indices):
        coefs = []
        # Fit linear model per sample
        for sample in range(y_ref.shape[0]):
            X = y_ref[sample, :, feat_idx].reshape(-1, 1)
            y = y_pred[sample, :, feat_idx].reshape(-1, 1)
            coef = LinearRegression().fit(X, y).coef_[0, 0]
            coefs.append(coef)

        name = feature_names[feat_idx] if feature_names else f"Feature_{feat_idx}"
        ax.hist(coefs, bins=bins, density=True, alpha=0.6, edgecolor='black')
        ax.set_title(f"Regression Coefficients\n{name}")
        ax.set_xlabel("Coefficient Value")
        ax.set_ylabel("Density")
        if xlim is not None:
            ax.set_xlim(xlim)
        if invert_x:
            ax.invert_xaxis()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_correlation_histogram(error_samples, feature_indices=[0],
                               feature_names=None, bins=20):
    """
    Plot histograms of Pearson correlation coefficients for specified features
    across all samples.
    """
    n_feats = len(feature_indices)
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5), sharey=True)
    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for ax, feat_idx in zip(axes, feature_indices):
        # Extract Pearson r values (metric index 3)
        r_vals = error_samples[:, feat_idx, 3, 0]
        name = feature_names[feat_idx] if feature_names else f"Feature_{feat_idx}"
        ax.hist(r_vals, bins=bins, density=True, alpha=0.6, edgecolor='black')
        ax.set_title(f"Correlation Coefficients\n{name}")
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Section: Utility Functions for Tabular Output
# =============================================================================

def create_sample_error_table(error_sample_1d, feature_names=None, metric_names=None):
    """
    Convert a single-sample error array into a DataFrame with metrics by feature.

    Parameters:
    - error_sample_1d: ndarray shape (n_features, n_metrics[, 2])
    - feature_names: optional list of feature labels
    - metric_names: optional list of metric names

    Returns:
    - DataFrame with one row per feature, columns for mean and std (if available)
    """
    n_features, n_metrics = error_sample_1d.shape[:2]
    has_std = error_sample_1d.ndim == 3 and error_sample_1d.shape[2] == 2

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    if metric_names is None:
        metric_names = ['MSE', 'R2', 'MAE', 'Pearson_r', 'NormError']

    rows = []
    for i in range(n_features):
        row = {'Feature': feature_names[i]}
        for m in range(n_metrics):
            if has_std:
                row[f'{metric_names[m]}_mean'] = error_sample_1d[i, m, 0]
                row[f'{metric_names[m]}_std'] = error_sample_1d[i, m, 1]
            else:
                row[metric_names[m]] = error_sample_1d[i, m]
        rows.append(row)

    return pd.DataFrame(rows)

# =============================================================================
# Section: Selection of Best/Worst Series
# =============================================================================

def select_best_worst_series(error_samples, metric, metric_names=None, top_k=5, reverse=None, feature_names=None):
    """
    Select top-k best and worst samples per feature according to a specified metric.

    Parameters:
    - error_samples: ndarray of shape (n_samples, n_features, n_metrics, 2)
    - metric: str, the name of the metric to sort by (e.g., 'MAE', 'R2')
    - metric_names: optional list of metric names in order
    - top_k: number of best/worst to return
    - reverse: optional dict {metric: bool} indicating sort order (True for descending)
    - feature_names: optional list of feature labels

    Returns:
    - results: dict mapping feature name to {
          'best': list of (sample_index, metric_value),
          'worst': list of (sample_index, metric_value)
      }
    """
    n_samples, n_features, n_metrics, _ = error_samples.shape

    # Default metric names if not provided
    if metric_names is None:
        metric_names = ['MSE', 'R2', 'MAE', 'Pearson_r', 'NormError']

    if metric not in metric_names:
        raise ValueError(f"Metric '{metric}' not found in {metric_names}")
    m_idx = metric_names.index(metric)

    # Default sort directions (False for ascending meaning lower is better unless specified)
    default_reverse = {'MSE': False, 'MAE': False, 'NormError': False, 'R2': True, 'Pearson_r': True}
    if reverse is None:
        reverse = default_reverse

    # Generate feature names if missing
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    results = {}

    for f_idx in range(n_features):
        # Extract mean metric values for this feature across all samples
        values = [(s, error_samples[s, f_idx, m_idx, 0]) for s in range(n_samples)]
        # Sort according to specified direction
        sorted_vals = sorted(values, key=lambda x: x[1], reverse=reverse.get(metric, False))
        results[feature_names[f_idx]] = {
            'best': sorted_vals[:top_k],
            'worst': sorted_vals[-top_k:]
        }

    return results

# =============================================================================
# Section: Prediction Plotting
# =============================================================================

def plot_predictions(y_true, y_pred, title='Model Predictions', dt=1.0,
                     sample_indices=None, feature_idx=0, max_plots=None,
                     feature_label='Displacement [m]', window_stride=None,
                     show_original=True):
    """
    Plot time-series predictions against true values for selected samples and feature.

    Parameters:
    - y_true, y_pred: arrays of shape (n_samples, timesteps, n_features) or (n_samples, timesteps)
    - title: base title for each subplot
    - dt: time step between measurements
    - sample_indices: list of sample indices to plot (defaults to all)
    - feature_idx: which feature column to visualize
    - max_plots: limit on number of plots
    - feature_label: label for y-axis
    - window_stride: if predictions use a stride, this rescales the time axis
    - show_original: if True, overlay full true series in gray
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Ensure 3D shape: (n_samples, timesteps, features)
    if y_true.ndim == 2:
        y_true = np.expand_dims(y_true, axis=-1)
    if y_pred.ndim == 2:
        y_pred = np.expand_dims(y_pred, axis=-1)

    # Determine if shapes match exactly
    same_shape = y_true.shape == y_pred.shape
    if not same_shape and window_stride is None:
        raise ValueError("window_stride must be specified when shapes differ.")

    # Prepare sample indices
    n_samples = y_true.shape[0]
    if sample_indices is None:
        sample_indices = list(range(n_samples))
    else:
        # filter out-of-range indices
        sample_indices = [i for i in sample_indices if 0 <= i < n_samples]

    if max_plots is not None:
        sample_indices = sample_indices[:max_plots]

    # Loop and plot each sample
    for i in sample_indices:
        # Full true series for the given feature
        full_series = y_true[i, :, feature_idx]
        time_full = np.arange(len(full_series)) * dt

        if same_shape:
            # When y_pred aligns with y_true
            y_plot = y_pred[i, :, feature_idx]
            y_ref = full_series
            time_axis = time_full
        else:
            # When y_pred is downsampled by window_stride
            y_ref = full_series[window_stride-1::window_stride]
            y_plot = y_pred[i, :, feature_idx]
            time_axis = np.arange(len(y_plot)) * (dt * window_stride)
            # Align lengths
            min_len = min(len(time_axis), len(y_ref), len(y_plot))
            y_ref = y_ref[:min_len]
            y_plot = y_plot[:min_len]
            time_axis = time_axis[:min_len]

        # Plot configuration
        plt.figure(figsize=(10, 4))
        if show_original:
            plt.plot(time_full, full_series, color='gray', alpha=0.3, label='Original')

        # Plot true vs predicted
        plt.plot(time_axis, y_ref, label='True', color='black')
        plt.plot(time_axis, y_plot, linestyle='--', label='Predicted', color='red')

        # Labels and title
        plt.title(f"{title} Sample {i}")
        plt.xlabel("Time [s]")
        plt.ylabel(feature_label)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
