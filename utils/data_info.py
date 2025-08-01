import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np


# =========================================================================================
# Function to plot seismic dataset records with comparison
# =========================================================================================

def plot_record_comparison(
    main_id,
    X_data,
    y_data,
    yt_data,
    ytt_data,
    dt,
    dataset_name="Dataset",
    show_dof=None
):
    """
    Visualize seismic dataset responses by plotting all records in gray and 
    highlighting one selected record in black. The function supports both
    single-degree-of-freedom (SDOF) and multi-degree-of-freedom (MDOF) data formats.

    Parameters
    ----------
    main_id : int
        Index of the record to highlight in black.

    X_data : ndarray
        Ground motion input data (shape: n_records, time_steps, 1).

    y_data : ndarray
        Relative displacement time history.
        Shape:
            - (n_records, time_steps, 1) for SDOF data
            - (n_records, time_steps, 1, n_dof) for MDOF data

    yt_data : ndarray
        Relative velocity time history. Same shape as `y_data`.

    ytt_data : ndarray
        Normalized inertial force time history. Same shape as `y_data`.

    dt : float
        Time step size in seconds.

    dataset_name : str, default="Dataset"
        Name of the dataset for plot title.

    show_dof : int or None, default=None
        Degree of freedom (1-based index) to plot when using MDOF data.
        If None and MDOF data is provided, defaults to DOF=1.
        Ignored for SDOF data.

    Notes
    -----
    - All records are plotted in light gray for context.
    - The selected record (main_id) is plotted in black for emphasis.
    - A custom legend is added showing:
        * All Records (gray line)
        * Selected Record (black line)
    - Displacement is shown in centimeters, acceleration in g units, 
      velocity and force in SI units.
    """

    # Create time vector
    time = np.arange(y_data.shape[1]) * dt

    # Set up the figure with 4 subplots (Acceleration, Displacement, Velocity, Force)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Data Visualization\n{dataset_name}", fontsize=16)

    # Detect if dataset has multiple DOFs (4 dimensions)
    multi_dof = (y_data.ndim == 4)

    # Determine which DOF to plot (defaults to 1 if multi-DOF)
    dof_idx = (show_dof - 1) if (multi_dof and show_dof is not None) else 0

    # -------------------------------------------------
    # 1) Plot all records in light gray for context
    # -------------------------------------------------
    for idx in np.arange(y_data.shape[0]):
        axs[0].plot(time, X_data[idx,:,0]/9.81, color="#B7B7B7", linewidth=1)  # Convert to g
        axs[1].plot(time, (y_data[idx,:,0,dof_idx] if multi_dof else y_data[idx,:,0])*100, color="#B7B7B7", linewidth=1)  # Convert to cm
        axs[2].plot(time, yt_data[idx,:,0,dof_idx] if multi_dof else yt_data[idx,:,0], color="#B7B7B7", linewidth=1)
        axs[3].plot(time, ytt_data[idx,:,0,dof_idx] if multi_dof else ytt_data[idx,:,0], color="#B7B7B7", linewidth=1)

    # -------------------------------------------------
    # 2) Plot the selected record in black for emphasis
    # -------------------------------------------------
    axs[0].plot(time, X_data[main_id,:,0]/9.81, linewidth=1, color="black")
    axs[1].plot(time, (y_data[main_id,:,0,dof_idx] if multi_dof else y_data[main_id,:,0])*100, linewidth=1, color="black")
    axs[2].plot(time, yt_data[main_id,:,0,dof_idx] if multi_dof else yt_data[main_id,:,0], linewidth=1, color="black")
    axs[3].plot(time, ytt_data[main_id,:,0,dof_idx] if multi_dof else ytt_data[main_id,:,0], linewidth=1, color="black")

    # -------------------------------------------------
    # 3) Set axis labels for each subplot
    # -------------------------------------------------
    axs[0].set_ylabel('Seismic Acceleration\n[g]')
    axs[1].set_ylabel('Relative Displacement\n[cm]')
    axs[2].set_ylabel('Relative Velocity\n[m/s]')
    axs[3].set_ylabel('Normalized Inertial Force\n[m/s² = N/kg]')
    axs[3].set_xlabel('Time [s]')

    # -------------------------------------------------
    # 4) Create custom legend with two entries
    # -------------------------------------------------
    all_records_line = mlines.Line2D([], [], color="#B7B7B7", linewidth=1.5, label="All Records")
    main_record_line = mlines.Line2D([], [], color="black", linewidth=1.5, label=f"Selected Record {main_id}")

    # Display legend on the first subplot
    axs[0].legend(handles=[all_records_line, main_record_line], loc='upper right')

    # Enable grid on all subplots
    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()



# =========================================================================================
# Function to plot dataset summary with subsets and residuals
# =========================================================================================

def plot_dataset_summary(
    X_data, y_data, yt_data, g_data,
    train_indices, val_indices, test_indices,
    residual_mask=None,
    sort_data=True,
    colors=None,
    markers=None,
    return_data=False
):
    """
    Plot dataset metrics highlighting data subsets and residual cases.

    This utility generates a multi-panel summary plot for a seismic dataset, where each 
    record is characterized by four metrics: residual position change, maximum velocity, 
    maximum inertial force, and peak ground acceleration (PGA). The function visually 
    distinguishes:
    
    - **Data splits:** Train, Validation, and Test sets are color-coded.
    - **Residual behavior:** Records with significant residual displacement are 
      displayed with a different marker.

    Parameters
    ----------
    X_data : ndarray of shape (n_records, n_timesteps, 1)
        Ground acceleration input time series (excitation) in [m/s²].

    y_data : ndarray of shape (n_records, n_timesteps, 1)
        Relative displacement response time series in [m].

    yt_data : ndarray of shape (n_records, n_timesteps, 1)
        Relative velocity response time series in [m/s].

    g_data : ndarray of shape (n_records, n_timesteps, 1)
        Normalized inertial force time series in [m/s²].

    train_indices : array-like
        Indices of records included in the training set.

    val_indices : array-like
        Indices of records included in the validation set.

    test_indices : array-like
        Indices of records included in the test set.

    residual_mask : array-like of bool, optional
        Boolean mask of shape (n_records,) indicating whether each record 
        presents significant residual displacement. Defaults to all False.

    sort_data : bool, default=True
        If True, records are sorted by residual position change (ascending) before plotting.

    colors : dict, optional
        Mapping of dataset subsets to colors. Example:
        `{"Train": "#ebc725", "Validation": "#1989da", "Test": "#0F9B0F"}`
        If None, defaults to this mapping.

    markers : dict, optional
        Mapping for residual vs non-residual markers. Example:
        `{"residual": "*", "non_residual": "o"}`
        If None, defaults to this mapping.

    return_data : bool, default=False
        If True, returns the pandas DataFrame with computed metrics.

    Returns
    -------
    df_metrics : pandas.DataFrame, optional
        DataFrame containing computed metrics per record, returned only if `return_data=True`.

    Notes
    -----
    - Metrics computed per record:
        * **Residual Drift (mm):** Absolute final displacement change in millimeters.
        * **Max Velocity (m/s):** Maximum absolute relative velocity.
        * **Max Inertial Force (m/s²=N/kg):** Maximum absolute normalized inertial force.
        * **PGA (g):** Peak ground acceleration.
    - Each metric is plotted in a separate subplot.
    - Markers:
        * **Residual:** Star symbol (or specified via `markers["residual"]`).
        * **Non-residual:** Circle symbol (or specified via `markers["non_residual"]`).
    - Points are **colored** according to the data split:
        **Default color coding:**
            * Yellow = Train
            * Blue = Validation
            * Green = Test
    - Record IDs are annotated (except in the first subplot to avoid clutter).
    

    Example
    -------
    >>> df_summary = plot_dataset_summary(
    ...     X_data, y_data, yt_data, g_data,
    ...     train_indices, val_indices, test_indices,
    ...     residual_mask=residual_mask,
    ...     sort_data=True,
    ...     return_data=True
    ... )
    """
    # Default parameters
    if colors is None:
        colors = {"Train": "#eec81e", "Validation": "#1b7eee", "Test": "#1C9B11"}
    if markers is None:
        markers = {"residual": "*", "non_residual": "o"}
    if residual_mask is None:
        residual_mask = np.zeros(len(y_data), dtype=bool)

    # Compute metrics
    position_change = np.abs(y_data[:, -1, 0] - y_data[:, 0, 0]) * 1000  # mm
    vel_max = np.max(np.abs(yt_data), axis=1).flatten()
    force_max = np.max(np.abs(g_data), axis=1).flatten()
    pga = (np.max(np.abs(X_data), axis=1).flatten()) / 9.81

    # Build dataframe
    df_metrics = pd.DataFrame({
        "Record": np.arange(1, len(y_data)+1),
        "Residual Drift (mm)": position_change,
        "Max Velocity (m/s)": vel_max,
        "Max Inertial Force (m/s²=N/kg)": force_max,
        "PGA (g)": pga,
        "Set": "Test",
        "Residual": residual_mask
    })

    # Assign data subsets
    df_metrics.loc[train_indices, "Set"] = "Train"
    df_metrics.loc[val_indices, "Set"] = "Validation"
    df_metrics.loc[test_indices, "Set"] = "Test"

    # Sort if requested
    if sort_data:
        title_suffix = " (Sorted by Residual Drift)"
        df_sorted = df_metrics.sort_values(by="Residual Drift (mm)", ascending=True).reset_index(drop=True)
    else:
        title_suffix = ""
        df_sorted = df_metrics.copy()

    # Metrics to plot
    metric_columns = ["Residual Drift (mm)", "Max Velocity (m/s)", "Max Inertial Force (m/s²=N/kg)", "PGA (g)"]
    fig, axs = plt.subplots(len(metric_columns), 1, figsize=(15, 4 * len(metric_columns)))

    if len(metric_columns) == 1:
        axs = [axs]

    for i, column in enumerate(metric_columns):
        for j in range(len(df_sorted)):
            value = df_sorted[column].iloc[j]
            record_id = j + 1
            set_label = df_sorted["Set"].iloc[j]
            is_residual = df_sorted["Residual"].iloc[j]

            axs[i].plot([record_id, record_id], [0, value], 'k-', linewidth=0.5, alpha=0.6)
            axs[i].scatter(
                record_id,
                value,
                color=colors.get(set_label, "gray"),
                s=60 if is_residual else 35,
                alpha=0.8,
                marker=markers["residual"] if is_residual else markers["non_residual"],
                label=f"{set_label}{' Residual' if is_residual else ''}"
            )

            # Annotate record IDs except in the first plot
            if i != 0:
                axs[i].text(record_id, value + (value * 0.02), f'{df_sorted["Record"].iloc[j]-1}',
                            ha='center', va='bottom', fontsize=8, color='gray')

        axs[i].set_title(f'{column} per Record{title_suffix}', fontsize=16)
        axs[i].set_ylabel(column, fontsize=14)
        axs[i].grid(True)
        axs[i].get_xaxis().set_visible(False)
        axs[i].set_xlim(0, len(df_sorted) + 1)

        # Custom legend
        handles, labels = axs[i].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axs[i].legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.show()

    if return_data:
        return df_metrics


# =========================================================================================
# Function to perform stratified sampling
# =========================================================================================

def stratified_sample(residual_idx, non_res_idx, total_size, ratio=0.5, random_state=None):
    """
    Perform stratified sampling to select a subset of indices while maintaining 
    a desired proportion of residual and non-residual records.

    Parameters
    ----------
    residual_idx : array-like
        Array of indices corresponding to records with residual displacement.

    non_res_idx : array-like
        Array of indices corresponding to records without residual displacement.

    total_size : int
        Total number of samples to select.

    ratio : float, default=0.5
        Desired fraction of residual records in the sampled subset (0 to 1).
        Example: ratio=0.5 → 50% residual, 50% non-residual.

    random_state : int or None, default=None
        Seed for reproducibility. If None, random sampling will vary between runs.

    Returns
    -------
    selected_indices : ndarray
        Array of selected indices of length `total_size`, containing a stratified
        combination of residual and non-residual indices.

    Notes
    -----
    - If there are insufficient residual or non-residual samples to meet the 
      requested ratio, the function fills the remaining quota with available 
      samples from the other group.
    - The resulting array is **not shuffled** internally. If you need a 
      randomized order, you can shuffle after selection.

    Example
    -------
    >>> residual_idx = [0, 2, 5]
    >>> non_res_idx = [1, 3, 4, 6, 7]
    >>> stratified_sample(residual_idx, non_res_idx, total_size=4, ratio=0.5, random_state=42)
    array([0, 2, 1, 4])
    """
    rng = np.random.default_rng(seed=random_state)

    # Calculate counts for each group
    n_res = min(int(total_size * ratio), len(residual_idx))
    n_non = min(total_size - n_res, len(non_res_idx))

    # Adjust if one group doesn't have enough samples
    if n_res < total_size * ratio:
        n_non = total_size - n_res
    elif n_non < total_size * (1 - ratio):
        n_res = total_size - n_non

    sampled_res = rng.choice(residual_idx, n_res, replace=False) if n_res > 0 else np.array([], dtype=int)
    sampled_non = rng.choice(non_res_idx, n_non, replace=False) if n_non > 0 else np.array([], dtype=int)

    selected_indices = np.concatenate([sampled_res, sampled_non])
    return np.sort(selected_indices)
