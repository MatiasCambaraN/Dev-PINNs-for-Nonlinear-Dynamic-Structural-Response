import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, gaussian_kde
from sklearn.linear_model import LinearRegression
from matplotlib import gridspec


# =============================================================================
# Section: Error Metrics Calculation
# =============================================================================

# def calculate_errors_with_pdf(y_true, y_pred, feature_names=None, kde_points=200):
#     """
#     Calculate per-sample, per-feature error metrics and estimate the PDF of normalized errors.

#     Parameters:
#     - y_true: ndarray of shape (n_samples, n_timesteps, n_features)
#     - y_pred: ndarray of shape (n_samples, n_timesteps, n_features)
#     - feature_names: optional list of feature labels
#     - kde_points: number of points to evaluate the KDE

#     Returns:
#     - metrics_df: DataFrame with mean and std for each metric by feature
#     - error_samples: ndarray with shape (n_samples, n_features, n_metrics, 2)
#     - error_pdfs: dict mapping feature name to {'x': array, 'pdf': array}
#     """
#     # Define metric names
#     metric_names = ['MSE', 'R2', 'MAE', 'Pearson_r', 'NormError']

#     n_samples, n_timesteps, n_features = y_true.shape
#     # Initialize array to store [mean, std] for each sample, feature, metric
#     error_samples = np.zeros((n_samples, n_features, len(metric_names), 2))
#     error_pdfs = {}

#     # Default feature names if not provided
#     if feature_names is None:
#         feature_names = [f'Feature_{i}' for i in range(n_features)]

#     # Loop over each sample
#     for sample_idx in range(n_samples):
#         yt = y_true[sample_idx]
#         yp = y_pred[sample_idx]

#         # Compute vectorized metrics per feature
#         mse_vals = mean_squared_error(yt, yp, multioutput='raw_values')
#         mae_vals = mean_absolute_error(yt, yp, multioutput='raw_values')
#         r2_vals = r2_score(yt, yp, multioutput='raw_values')

#         # Compute Pearson r and normalized error per feature
#         for feat_idx in range(n_features):
#             true_series = yt[:, feat_idx]
#             pred_series = yp[:, feat_idx]
#             error_series = true_series - pred_series

#             # Pearson correlation coefficient
#             pearson_r, _ = pearsonr(true_series, pred_series)

#             # Normalized error: divide by max absolute true value
#             denom = np.max(np.abs(true_series))
#             norm_err = error_series / denom if denom != 0 else error_series
#             norm_mean = norm_err.mean()
#             norm_std = norm_err.std()

#             # Store mean and std for each metric
#             # MSE: mean of squared error and std of squared error
#             error_samples[sample_idx, feat_idx, 0, 0] = mse_vals[feat_idx]
#             error_samples[sample_idx, feat_idx, 0, 1] = np.std(error_series ** 2)

#             # R2: mean R2 and no std
#             error_samples[sample_idx, feat_idx, 1, 0] = r2_vals[feat_idx]
#             error_samples[sample_idx, feat_idx, 1, 1] = 0.0

#             # MAE: mean absolute error and std
#             error_samples[sample_idx, feat_idx, 2, 0] = mae_vals[feat_idx]
#             error_samples[sample_idx, feat_idx, 2, 1] = np.std(np.abs(error_series))

#             # Pearson r: value and no std
#             error_samples[sample_idx, feat_idx, 3, 0] = pearson_r
#             error_samples[sample_idx, feat_idx, 3, 1] = 0.0

#             # Normalized error: mean and std
#             error_samples[sample_idx, feat_idx, 4, 0] = norm_mean
#             error_samples[sample_idx, feat_idx, 4, 1] = norm_std

#     # Estimate PDFs of normalized errors across all samples per feature
#     for feat_idx in range(n_features):
#         aggregated_norm_err = []
#         for sample_idx in range(n_samples):
#             true_series = y_true[sample_idx, :, feat_idx]
#             pred_series = y_pred[sample_idx, :, feat_idx]
#             error_series = true_series - pred_series
#             denom = np.max(np.abs(true_series))
#             norm_err = error_series / denom if denom != 0 else error_series
#             aggregated_norm_err.extend(norm_err)

#         kde = gaussian_kde(aggregated_norm_err)
#         x_vals = np.linspace(-1, 1, kde_points)
#         error_pdfs[feature_names[feat_idx]] = {'x': x_vals, 'pdf': kde(x_vals)}

#     # Compute mean and std across samples for each metric and feature
#     mean_vals = error_samples[..., 0].mean(axis=0)
#     std_vals = error_samples[..., 0].std(axis=0)

#     # Build a DataFrame with interleaved columns: metric_mean, metric_std
#     columns = []
#     data_cols = []
#     for idx, metric in enumerate(metric_names):
#         columns += [f'{metric}_mean', f'{metric}_std']
#         data_cols.append(mean_vals[:, idx])
#         data_cols.append(std_vals[:, idx])

#     metrics_df = pd.DataFrame(np.column_stack(data_cols), columns=columns)
#     metrics_df.insert(0, 'Feature', feature_names)

#     return metrics_df, error_samples, error_pdfs


def calculate_errors_with_pdf(y_true,
                              y_pred,
                              feature_names = None,
                              kde_points = 200, 
                              hysteresis_errors = False,
                              desp_force_index = None):
    """
    Calculate per-sample, per-feature error metrics and estimate the PDF of normalized errors.

    Metrics computed per sample & feature:
      - RMSE          : √(mean((y_true - y_pred)^2))
      - NRMSE         : RMSE normalized by (max(y_true) - min(y_true))
      - MAE           : mean(|y_true - y_pred|)
      - L2RE          : ||y_true - y_pred||₂ / ||y_true||₂
      - NEE           : L2RE²
      - R2            : coefficient of determination
      - Pearson_r     : Pearson correlation coefficient
      - NormError     : (y_true - y_pred) / max(|y_true|) → mean & std

    Parameters:
    - y_true       : ndarray of shape (n_samples, n_timesteps, n_features)
    - y_pred       : ndarray of shape (n_samples, n_timesteps, n_features)
    - feature_names: optional list of feature labels
    - kde_points   : number of points to evaluate the KDE of normalized errors
    - hysteresis_errors: if True, also compute hysteresis error metrics for displacement-force pairs
    - desp_force_index: if provided, indices for displacement and force features (e.g., [0, 1] for [displacement, force])

    Returns:
    - metrics_df   : DataFrame with mean and std for each metric by feature
    - metrics_by_feature: dict with structure {feature_name: {metric_name: array(n_samples, 2)}}
                     where array[:, 0] = mean values, array[:, 1] = std values
    - error_pdfs   : dict mapping feature name → {'x': array, 'pdf': array}
    """
    # Define metric names in desired order
    metric_names = [ 'L2RE', 'RMSE', 'NRMSE', 'MAE', 'NEE', 'R2', 'Pearson_r', 'NormError']

    n_samples, n_timesteps, n_features = y_true.shape
    n_metrics = len(metric_names)

    # Initialize dictionary: {feature_name: {metric_name: array(n_samples, 2)}}
    metrics_by_feature = {}
    error_pdfs = {}

    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Initialize the nested dictionary structure
    for feat_name in feature_names:
        metrics_by_feature[feat_name] = {}
        for metric_name in metric_names:
            metrics_by_feature[feat_name][metric_name] = np.zeros((n_samples, 2))

    # Loop over each sample
    for i in range(n_samples):
        yt = y_true[i]
        yp = y_pred[i]
        errors = yt - yp                    # shape (n_timesteps, n_features)
        sq_errors = errors**2

        # Precompute global norms and ranges per feature
        true_norms = np.linalg.norm(yt, axis=0)                  # L2 of true series
        error_norms = np.linalg.norm(errors, axis=0)             # L2 of error series
        true_min = yt.min(axis=0)
        true_max = yt.max(axis=0)
        true_range = true_max - true_min
        denom_abs = np.maximum(np.abs(yt).max(axis=0), 1e-12)     # avoid div by zero

        # Vectorized: MSE, MAE, R2
        mse_vals = mean_squared_error(yt, yp, multioutput='raw_values')
        mae_vals = mean_absolute_error(yt, yp, multioutput='raw_values')
        r2_vals  = r2_score(yt, yp, multioutput='raw_values')

        for j in range(n_features):
            err_j       = errors[:, j]
            sq_err_j    = sq_errors[:, j]
            abs_err_j   = np.abs(err_j)

            # RMSE and its std
            rmse_mean = np.sqrt(mse_vals[j])
            rmse_std  = np.sqrt(np.std(sq_err_j))

            # NRMSE normalized by (max - min)
            if true_range[j] > 0:
                nrmse_mean = rmse_mean / true_range[j]
                nrmse_std  = rmse_std / true_range[j]
            else:
                nrmse_mean = np.nan
                nrmse_std  = np.nan

            # MAE mean & std
            mae_mean = mae_vals[j]
            mae_std  = abs_err_j.std()

            # L2 relative error (L2RE) and NEE = L2RE^2
            l2re = error_norms[j] / (true_norms[j] + 1e-12)
            nee  = l2re**2

            # R2: mean & zero std
            r2_mean, r2_std = r2_vals[j], 0.0

            # Pearson r: one value, zero std
            pearson_r, _ = pearsonr(yt[:, j], yp[:, j])
            pr_mean, pr_std = pearson_r, 0.0

            # Normalized error series & stats
            norm_err_series = err_j / denom_abs[j]
            ne_mean = norm_err_series.mean()
            ne_std  = norm_err_series.std()

            # Store metrics in the dictionary structure
            vals = [
                (rmse_mean,    rmse_std),
                (nrmse_mean,   nrmse_std),
                (mae_mean,     mae_std),
                (l2re,         0.0),
                (nee,          0.0),
                (r2_mean,      r2_std),
                (pr_mean,      pr_std),
                (ne_mean,      ne_std),
            ]
            feat_name = feature_names[j]
            for k, (m_mean, m_std) in enumerate(vals):
                metric_name = metric_names[k]
                metrics_by_feature[feat_name][metric_name][i, 0] = m_mean
                metrics_by_feature[feat_name][metric_name][i, 1] = m_std

    # Estimate PDFs of normalized errors across all samples per feature
    for j in range(n_features):
        all_norm_errs = []
        for i in range(n_samples):
            err_j = y_true[i, :, j] - y_pred[i, :, j]
            denom = np.max(np.abs(y_true[i, :, j]))
            denom = denom if denom != 0 else 1.0
            all_norm_errs.extend((err_j / denom).tolist())
        kde = gaussian_kde(all_norm_errs)
        x_vals = np.linspace(-1, 1, kde_points)
        error_pdfs[feature_names[j]] = {
            'x': x_vals,
            'pdf': kde(x_vals)
        }

    # Aggregate across samples: mean & std of the metric means
    metric_means = np.zeros((n_features, n_metrics))
    metric_stds = np.zeros((n_features, n_metrics))
    
    for j, feat_name in enumerate(feature_names):
        for k, metric_name in enumerate(metric_names):
            # Extract mean values across samples for this feature-metric combination
            sample_means = metrics_by_feature[feat_name][metric_name][:, 0]
            metric_means[j, k] = sample_means.mean()
            metric_stds[j, k] = sample_means.std()

    # Build DataFrame with interleaved mean & std columns
    cols, data = [], []
    for name in metric_names:
        cols += [f'{name}_mean', f'{name}_std']
    for j in range(n_metrics):
        data.append(metric_means[:, j])
        data.append(metric_stds[:, j])
    metrics_df = pd.DataFrame(
        np.column_stack(data),
        columns=cols,
        index=feature_names
    ).reset_index().rename(columns={'index': 'Feature'})
    
    if hysteresis_errors:
        # Calculate hysteresis errors if requested
        y_true_hysteresis = y_true[:, :, desp_force_index]  # Assuming desp_force_index is [displacement, force]
        y_pred_hysteresis = y_pred[:, :, desp_force_index]
        hysteresis_metrics_df, hysteresis_metrics_by_sample = calculate_hysteresis_errors(y_true_hysteresis, y_pred_hysteresis)
        metrics_df = pd.merge(metrics_df,
                              hysteresis_metrics_df,
                              on="Feature",
                              how="outer" ) # or "left" if you want to keep only the original features
        metrics_by_feature['Hysteresis'] = hysteresis_metrics_by_sample
    
    return metrics_df, metrics_by_feature, error_pdfs



# =============================================================================

def normalize_by_true_max_abs(y_true, y_pred, axis=1):
    """
    Normalize both y_true and y_pred by the max absolute value of y_true along the given axis.

    Parameters:
    - y_true: ndarray (n_samples, n_timesteps) or (n_samples, n_timesteps, n_features)
    - y_pred: same shape as y_true
    - axis: axis along which to normalize (default: 1 = time)

    Returns:
    - y_true_norm, y_pred_norm: normalized arrays
    """
    max_abs = np.max(np.abs(y_true), axis=axis, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-12)  # to avoid division by zero

    y_true_norm = y_true / max_abs
    y_pred_norm = y_pred / max_abs

    return y_true_norm, y_pred_norm



def calculate_hysteresis_errors(y_true, y_pred):
    """
    Computes hysteresis error metrics for a batch of 2D signals (displacement-force pairs).

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_timesteps, 2)
        Real signals [displacement, force] for each sample.

    y_pred : ndarray of shape (n_samples, n_timesteps, 2)
        Predicted signals [displacement, force] for each sample.

    Returns
    -------
    metrics_by_sample : dict
        Dictionary with keys as metric names and values as (n_samples, 2) arrays:
        [:, 0] = mean, [:, 1] = std for each sample.

    metrics_df : pandas.DataFrame
        DataFrame with global mean and std for each metric across all samples.
    """
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    assert y_true.shape[2] == 2, "Last dimension must contain [displacement, force]"

    n_samples, n_timesteps, _ = y_true.shape

    # Initialize dict to collect per-sample metrics
    metric_names = ['L2RE_2D', 'RMSE_2D', 'NRMSE_2D', 'MAE_2D', 'NMAE_2D', 'Area_Error']
    metrics_by_sample = {name: np.zeros((n_samples, 2)) for name in metric_names}

    for i in range(n_samples):
        # Normalize signals by their max absolute value because
        # we are dealing with 2D hysteresis curves
        yd, yd_hat = normalize_by_true_max_abs(y_true[i, :, 0], y_pred[i, :, 0], axis=0)     # Normalized displacement (true and predicted)
        yf, yf_hat = normalize_by_true_max_abs(y_true[i, :, 1], y_pred[i, :, 1], axis=0)     # Normalized force (true and predicted)

        # Pointwise distances
        distances = np.sqrt((yd - yd_hat) ** 2 + (yf - yf_hat) ** 2)
        mae_2d = np.mean(distances)
        rmse_2d = np.sqrt(np.mean((yd - yd_hat) ** 2 + (yf - yf_hat) ** 2))

        # Rango 2D
        dy = np.max(yd) - np.min(yd)
        df = np.max(yf) - np.min(yf)
        rango_2d = np.sqrt(dy**2 + df**2) if (dy != 0 or df != 0) else 1.0

        # Normalized errors
        nmae_2d = mae_2d / rango_2d
        nrmse_2d = rmse_2d / rango_2d

        # L2RE_2D
        l2_error = np.linalg.norm(np.stack([yd - yd_hat, yf - yf_hat], axis=1))
        l2_real = np.linalg.norm(np.stack([yd, yf], axis=1))
        l2re_2d = l2_error / l2_real if l2_real != 0 else 0.0

        # Area error (trapezoidal integral)
        area_real = np.sum(0.5 * (yf[1:] + yf[:-1]) * (yd[1:] - yd[:-1]))
        area_pred = np.sum(0.5 * (yf_hat[1:] + yf_hat[:-1]) * (yd_hat[1:] - yd_hat[:-1]))
        area_error = abs((area_pred - area_real) / area_real) if area_real != 0 else 0.0

        # Store [mean, std] for each sample (std of time series errors where applicable)
        metrics_by_sample['MAE_2D'][i] = [mae_2d, distances.std()]
        metrics_by_sample['RMSE_2D'][i] = [rmse_2d, np.std((yd - yd_hat) ** 2 + (yf - yf_hat) ** 2)]
        metrics_by_sample['NMAE_2D'][i] = [nmae_2d, 0.0]
        metrics_by_sample['NRMSE_2D'][i] = [nrmse_2d, 0.0]
        metrics_by_sample['L2RE_2D'][i] = [l2re_2d, 0.0]
        metrics_by_sample['Area_Error'][i] = [area_error, 0.0]

    # Create global summary DataFrame with one row (hysteresis) and one column per metric
    summary_row = {"Feature": "Hysteresis"}
    for metric, values in metrics_by_sample.items():
        mean_over_samples = values[:, 0].mean()
        std_over_samples = values[:, 0].std()
        summary_row[f"{metric}_mean"] = mean_over_samples
        summary_row[f"{metric}_std"] = std_over_samples

    metrics_df = pd.DataFrame([summary_row])
    
    return metrics_df, metrics_by_sample



# =============================================================================
# Section: Table Formatting
# =============================================================================

def format_metrics_df(df, metric_names=None):
    """
    Combine mean and std columns into formatted strings in a DataFrame.
    If multiple metric variants exist (e.g., 'L2RE', 'L2RE_2D'), only one is kept,
    and a warning is shown if others are discarded.

    Parameters:
    - df: DataFrame with columns '<metric>_mean' and '<metric>_std'
    - metric_names: list of metric base names to format

    Returns:
    - formatted_df: DataFrame with one column per metric, values as 'mean ± std'
    """
    # Copy feature column to result
    result_df = pd.DataFrame({'Feature': df['Feature']})
    
    # Identify all <metric>_mean columns
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    std_cols = [col for col in df.columns if col.endswith('_std')]

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

    result_df.replace("nan ± nan", "N/A", inplace=True)

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
        metric_names = ['L2RE', 'RMSE', 'NRMSE', 'MAE', 'NEE', 'R2', 'Pearson_r', 'NormError']

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

def select_best_worst_series(metrics_by_sample, metric, metric_names=None, top_k=5, reverse=None, feature_names=None):
    """
    Select top-k best and worst samples per feature according to a specified metric.

    Parameters:
    - metrics_by_sample: dict with structure {feature_name: {metric_name: array(n_samples, 2)}}
                         where array[:, 0] = mean values, array[:, 1] = std values
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

    # Default metric names if not provided
    if metric_names is None:
        metric_names = ['L2RE', 'RMSE', 'NRMSE', 'MAE', 'NEE', 'R2', 'Pearson_r', 'NormError',
                        'L2RE_2D', 'RMSE_2D', 'NRMSE_2D', 'MAE_2D', 'NMAE_2D', 'Area_Error']

    if metric not in metric_names:
        raise ValueError(f"Metric '{metric}' not found in {metric_names}")

    # Default sort directions (False for ascending meaning lower is better unless specified)
    default_reverse = {'L2RE': False, 'RMSE': False, 'NRMSE': False, 'MAE': False, 'NEE': False, 
                      'R2': True, 'Pearson_r': True, 'NormError': False, 
                      'L2RE_2D': False, 'RMSE_2D': False, 'NRMSE_2D': False, 
                      'MAE_2D': False, 'NMAE_2D': False, 'Area_Error': False}
    
    if reverse is None:
        reverse = default_reverse
        
    # Validate that the metric exists
    feature_names = list(metrics_by_sample.keys())
    if not feature_names:
        raise ValueError("metrics_by_sample dictionary is empty")

    results = {}

    for feat_name in feature_names:
        
        available_metrics = metrics_by_sample[feat_name].keys()

        # Try exact match first
        matched_metric = metric if metric in available_metrics else None

        # If not found, find metric that starts with base metric (e.g., 'L2RE' in 'L2RE_2D')
        if not matched_metric:
            candidates = [m for m in available_metrics if m.startswith(metric)]
            if not candidates:
                raise ValueError(f"Metric '{metric}' not found for feature '{feat_name}'. Available metrics:\n \t{list(available_metrics)}")
            # Take first match (you can customize this)
            matched_metric = candidates[0]
            print(f"Using metric '{matched_metric}' for feature '{feat_name}' based on prefix match with '{metric}'")
            
        # Extract mean metric values for this feature across all samples
        metric_values = metrics_by_sample[feat_name][matched_metric][:, 0]  # Take mean values (column 0)
        n_samples = len(metric_values)
        
        # Create list of (sample_index, metric_value) tuples
        values = [(s, metric_values[s]) for s in range(n_samples)]
        
        # Determine sorting direction
        sort_desc = reverse.get(matched_metric, False)
        
        # Sort according to specified direction
        sorted_vals = sorted(values, key=lambda x: x[1], reverse=sort_desc)

        results[feat_name] = {
            'best': sorted_vals[:top_k],
            'worst': sorted_vals[-top_k:]
        }

    return results

# =============================================================================
# Section: Prediction Plotting
# =============================================================================

def plot_predictions(y_true, y_pred, dt=1.0, title='Model Predictions',
                     sample_indices=None, feature_idx=0, max_plots=None,
                     feature_label='Displacement [m]', window_stride=0,
                     show_original=True):
    """
    Plot time-series predictions against true values for selected samples and feature.

    Parameters:
    - y_true, y_pred: arrays of shape (n_samples, timesteps, n_features) or (n_samples, timesteps)
    - dt: time step between measurements
    - title: base title for each subplot
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
    if not same_shape and window_stride == 0:
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


# =============================================================================
# Section: Prediction Histeresys Plotting
# =============================================================================

def plot_hysteresis_prediction(y_true,
                               y_pred,
                               dt = 1.0,
                               title = 'Hysteresis: True vs. Predicted',
                               sample_indices = None,
                               max_plots = None,
                               figsize = (16, 6),
                               disp_label = 'Displacement [m]',
                               force_label = 'Normalized Force [m/s²]'
                               ):
    """
    Plot true vs. predicted hysteresis loops (left) and, on the right,
    time histories for displacement and normalized force.

    The two time-history plots on the right are stacked so that their
    combined height matches the height of the hysteresis loop on the left.

    Parameters:
    ----------
    y_true : ndarray of shape (n_samples, n_timesteps, 2)
        Real signals [displacement, force] for each sample.

    y_pred : ndarray of shape (n_samples, n_timesteps, 2)
        Predicted signals [displacement, force] for each sample.
        
    dt : float, default=1.0
        Time step between samples.
    title : str
        Base title for each sample.
    sample_indices : list of int, optional
        Which samples to plot. Defaults to all.
    max_plots : int, optional
        Max number of samples to display.
    figsize : tuple, default=(12, 4)
        Size of the figure.
    * _label : str
        Axis labels for true/predicted curves.
    """
    # Convert inputs
    disp_true = np.asarray(y_true[:, :, 0])  # Displacement 
    force_norm_true = np.asarray(y_true[:, :, 1])  # Normalized Force
    disp_pred = np.asarray(y_pred[:, :, 0])  # Displacement (predicted)
    force_norm_pred = np.asarray(y_pred[:, :, 1])  # Normalized Force (predicted)

    # Validate shapes
    if not (disp_true.shape == force_norm_true.shape == disp_pred.shape == force_norm_pred.shape):
        raise ValueError("All input arrays must have the same shape (n_samples, timesteps).")

    n_samples, n_steps = disp_true.shape
    time = np.arange(n_steps) * dt

    # Determine which samples to plot
    if sample_indices is None:
        sample_indices = list(range(n_samples))
    else:
        sample_indices = [i for i in sample_indices if 0 <= i < n_samples]

    if max_plots is not None:
        sample_indices = sample_indices[:max_plots]

    # Loop over selected samples
    for i in sample_indices:
        x_true = disp_true[i]
        y_true = force_norm_true[i]
        x_pred = disp_pred[i]
        y_pred = force_norm_pred[i]

        # Create figure with GridSpec: 2 rows, 2 cols
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            nrows=2, ncols=2,
            width_ratios=[2, 3],    # 2/5 for hysteresis, 3/5 for time plots
            height_ratios=[1, 1],
            wspace=0.3, hspace=0.4
        )

        # Left: hysteresis loop spanning both rows
        ax_loop = fig.add_subplot(gs[:, 0])
        ax_loop.plot(x_true, y_true, '-', label='True', color='black')
        ax_loop.plot(x_pred, y_pred, '--', label='Predicted', color='red')
        ax_loop.set_xlabel('Displacement [m]')
        ax_loop.set_ylabel('Normalized Force [m/s²]')
        ax_loop.set_title(f"{title} — Sample {i}")
        ax_loop.grid(True)
        ax_loop.legend()

        # Top-right: displacement vs time
        ax_disp = fig.add_subplot(gs[0, 1])
        ax_disp.plot(time, x_true, '-', label='True Disp.', color='black')
        ax_disp.plot(time, x_pred, '--', label='Pred. Disp.', color='red')
        ax_disp.set_xlabel('Time [s]')
        ax_disp.set_ylabel( 'Displacement [m]')
        ax_disp.set_title(disp_label.split('[')[0].strip() + ' vs. Time')
        ax_disp.grid(True)
        ax_disp.legend()

        # Bottom-right: force vs time
        ax_force = fig.add_subplot(gs[1, 1])
        ax_force.plot(time, y_true, '-', label='True Force', color='black')
        ax_force.plot(time, y_pred, '--', label='Pred. Force', color='red')
        ax_force.set_xlabel('Time [s]')
        ax_force.set_ylabel('Normalized Force [m/s²]')
        ax_force.set_title(force_label.split('[')[0].strip() + ' vs. Time')
        ax_force.grid(True)
        ax_force.legend()

        plt.show()
