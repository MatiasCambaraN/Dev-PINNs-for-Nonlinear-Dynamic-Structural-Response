import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, gaussian_kde
from sklearn.linear_model import LinearRegression



##############################################################################

def calcular_errores_con_pdf(y_true, y_pred, feature_names=None, kde_points=200):
    """
    Calcula métricas por sample y feature usando sklearn y scipy.
    
    Retorna:
    - df_final: resumen promedio/std por feature
    - errores_sample: ndarray (samples, features, metrics, 2)
    - pdfs_error: dict de PDFs por feature
    """
    n_samples, n_timesteps, n_features = y_true.shape
    metric_names = ['MSE', 'R2', 'MAE', 'Pearson_r', 'NormError']
    errores_sample = np.zeros((n_samples, n_features, len(metric_names), 2))
    pdfs_error = {}

    if feature_names is None:
        feature_names = [f'Feature_{f}' for f in range(n_features)]

    for s in range(n_samples):
        yt = y_true[s]  # shape (T, F)
        yp = y_pred[s]  # shape (T, F)

        # Usamos sklearn con multioutput='raw_values' para obtener métricas por feature
        mse_vals = mean_squared_error(yt, yp, multioutput='raw_values') # shape (F,)
        mae_vals = mean_absolute_error(yt, yp, multioutput='raw_values') # shape (F,)
        r2_vals = r2_score(yt, yp, multioutput='raw_values') # shape (F,)

        # Pearson y NormError: columna por columna
        for f in range(n_features):
            yt_f = yt[:, f]
            yp_f = yp[:, f]
            error = yt_f - yp_f

            # Pearson
            pearson_r, _ = pearsonr(yt_f, yp_f)

            # Normalized Error
            denom = np.max(np.abs(yt_f))
            norm_error_series = error / denom if denom != 0 else error
            norm_error_mean = norm_error_series.mean()
            norm_error_std = norm_error_series.std()

            # Guardar todas las métricas
            errores_sample[s, f, 0, 0] = mse_vals[f]
            errores_sample[s, f, 0, 1] = np.std(error**2)

            errores_sample[s, f, 1, 0] = r2_vals[f]
            errores_sample[s, f, 1, 1] = 0  # no std

            errores_sample[s, f, 2, 0] = mae_vals[f]
            errores_sample[s, f, 2, 1] = np.std(np.abs(error))

            errores_sample[s, f, 3, 0] = pearson_r
            errores_sample[s, f, 3, 1] = 0  # no std

            errores_sample[s, f, 4, 0] = norm_error_mean
            errores_sample[s, f, 4, 1] = norm_error_std

    # PDFs de error normalizado por feature
    for f in range(n_features):
        all_errors = []
        for s in range(n_samples):
            yt_f = y_true[s, :, f]
            yp_f = y_pred[s, :, f]
            error = yt_f - yp_f
            denom = np.max(np.abs(yt_f))
            norm_err = error / denom if denom != 0 else error
            all_errors.extend(norm_err)

        kde = gaussian_kde(all_errors)
        x_vals = np.linspace(-1, 1, kde_points)
        pdfs_error[feature_names[f]] = {'x': x_vals, 'pdf': kde(x_vals)}

    # Resumen por feature
    mean_vals = errores_sample[:, :, :, 0].mean(axis=0)
    std_vals = errores_sample[:, :, :, 0].std(axis=0)

    # DataFrame con columnas intercaladas
    columnas, datos = [], []
    for s, m in enumerate(metric_names):
        columnas.extend([f'{m}_mean', f'{m}_std'])
        datos.append(mean_vals[:, s])
        datos.append(std_vals[:, s])
    df_metrica = pd.DataFrame(np.column_stack(datos), columns=columnas)
    df_final = pd.concat([pd.Series(feature_names, name='Feature'), df_metrica], axis=1)

    return df_final, errores_sample, pdfs_error

##############################################################################


def formatear_metricas_df(df, metric_names=None):
    import pandas as pd
    
    # Copiar la columna de la feature para mantenerla
    df_resultado = pd.DataFrame()
    df_resultado['Feature'] = df['Feature']
    
    if metric_names is None:
        # Buscar todas las métricas base (sin sufijo)
        metric_names = set(col.replace('_mean', '') for col in df.columns if col.endswith('_mean'))

    # Para cada métrica, combinar media y std en un string
    for metrica in metric_names:
        col_mean = f'{metrica}_mean'
        col_std = f'{metrica}_std'

        if col_mean in df.columns and col_std in df.columns:
            df_resultado[metrica] = df.apply(
                lambda row: f"{row[col_mean]:.6f} ± {row[col_std]:.6f}", axis=1
            )
        else:
            raise ValueError(f"La métrica '{metrica}' no se encuentra en las columnas del DataFrame.")

    return df_resultado


##############################################################################


def seleccionar_mejores_peores_series(errores_sample, metrica, metric_names=None, top_k=5, reverse=None, names_features=None):
    """
    Selecciona las top_k mejores y peores series por feature según una métrica dada.

    Parámetros:
    - errores_sample: array (samples, features, metrics)
    - metrica: str, nombre de la métrica (ej: "MAE", "R2")
    - metric_names: lista de nombres de métricas (si no se usa el orden por defecto)
    - top_k: cuántas mejores/peores devolver
    - reverse: dict opcional con {metrica: True/False} para ordenar (mayor/mejor o menor/mejor)

    Retorna:
    - resultados: dict con keys = feature index o name, y valores con:
        {
            'mejores': list of (sample, valor),
            'peores': list of (sample, valor)
        }
    """
    n_samples, n_features, n_metrics, mean_std = errores_sample.shape

    if metric_names is None:
        metric_names = ['MSE', 'R2', 'MAE', 'Pearson_r', 'NormError']

    if metrica not in metric_names:
        raise ValueError(f"Métrica '{metrica}' no encontrada en {metric_names}")

    m_idx = metric_names.index(metrica)

    if reverse is None:
        reverse = {
            'MSE': False,
            'MAE': False,
            'NormError': False,
            'R2': True,
            'Pearson_r': True
        }

    resultados = {}

    for f in range(n_features):
        valores = [(s, errores_sample[s, f, m_idx, 0]) for s in range(n_samples)]  # solo el mean
        ordenados = sorted(valores, key=lambda x: x[1], reverse=reverse.get(metrica, False))
        mejores = ordenados[:top_k]
        peores = ordenados[-top_k:]
        if names_features is not None:
            feature_name = names_features[f]
        else:
            feature_name = f'Feature_{f}'
        resultados[feature_name] = {
            'mejores': mejores,
            'peores': peores
        }

    return resultados


##############################################################################

def tabla_errores_sample(errores_sample_1d, feature_names=None, metric_names=None):
    """
    Convierte los errores de un sample en una tabla con métricas por feature.

    Parámetros:
    - errores_sample_1d: ndarray de shape (features, metrics, [2]) con o sin std
    - feature_names: lista opcional con nombres de features
    - metric_names: lista opcional con nombres de métricas

    Retorna:
    - DataFrame con una fila por feature y columnas con métricas (+std si existen)
    """
    n_features, n_metrics = errores_sample_1d.shape[:2]
    tiene_std = errores_sample_1d.ndim == 3 and errores_sample_1d.shape[2] == 2

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    if metric_names is None:
        metric_names = ['MSE', 'R2', 'MAE', 'Pearson_r', 'NormError']

    data = []

    for f in range(n_features):
        fila = {'Feature': feature_names[f]}
        for m in range(n_metrics):
            if tiene_std:
                mean_val = errores_sample_1d[f, m, 0]
                std_val = errores_sample_1d[f, m, 1]
                fila[f'{metric_names[m]}_mean'] = mean_val
                fila[f'{metric_names[m]}_std'] = std_val
            else:
                fila[metric_names[m]] = errores_sample_1d[f, m]
        data.append(fila)

    return pd.DataFrame(data)

    
##############################################################################

def graficar_pdf_errores(pdfs_error, ci_threshold=0.10, xlim=(-100, 100)):
    plt.figure(figsize=(10, 5))

    colores = ['blue', 'red', 'magenta', 'green', 'orange', 'purple', 'cyan']
    estilos = ['-', '--', ':', '-.', '-', '--', ':', '-.',]

    for idx, (feature, datos) in enumerate(pdfs_error.items()):
        x = datos['x']
        pdf = datos['pdf']

        # Área bajo la curva en ±10%
        mask = (x >= -ci_threshold) & (x <= ci_threshold)
        ci = np.trapz(pdf[mask], x[mask])
        ci_pct = round(ci * 100)

        # Dibujar la curva
        plt.plot(x * 100, pdf, estilos[idx % len(estilos)],
                 color=colores[idx % len(colores)],
                 label=f'{feature}  (CI ≈ {ci_pct}%)')

        # Agregar anotación
        # x_peak = x[np.argmax(pdf)]
        # y_peak = max(pdf)
        # plt.annotate(f'CI = {ci_pct}%',
        #              xy=(x_peak, y_peak),
        #              xytext=(x_peak + 15, y_peak),
        #              arrowprops=dict(arrowstyle='->'))

    # Líneas verticales en ±10%
    plt.axvline(x=-10, color='k', linestyle='-.')
    plt.axvline(x=10, color='k', linestyle='-.')

    plt.title("Error Distribution of Prediction")
    plt.xlabel("Normalized Error [%]")
    plt.xlim(xlim)
    plt.ylabel("PDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
##############################################################################  

def graficar_histograma_correlacion(errores_sample, feature_indices=[0], feature_names=None, bins=20):
    """
    Genera histogramas del coeficiente de correlación r para una o más características.

    Parámetros:
    - errores_sample: ndarray de shape (samples, features, metrics)
    - feature_indices: lista de índices de características a graficar
    - feature_names: lista opcional con nombres de características
    - bins: número de bins del histograma
    """
    n = len(feature_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)

    if n == 1:
        axes = [axes]  # Asegura que axes sea iterable

    for i, feature_idx in enumerate(feature_indices):
        r_vals = errores_sample[:, feature_idx, 3, 0]  # ← aquí el fix
        name = f"{feature_names[feature_idx]}" if feature_names else f"Feature {feature_idx}"

        axes[i].hist(r_vals, bins=bins, density=True, color='royalblue', edgecolor='black')
        axes[i].set_title(f"Regression Analysis\n{name}")
        axes[i].set_xlabel("Correlation Coefficient r")
        #axes[i].set_xlim(1, 0.4)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        #if i == 0:
        axes[i].set_ylabel("Probability")


    plt.tight_layout()
    plt.show()
    
    
##############################################################################

def graficar_histograma_coeficientes_regresion(
    y_pred_ref,
    y_pred,
    feature_indices=[0],
    feature_names=None,
    bins=20,
    invert_x=True,
    xlim=None
):
    """
    Genera histogramas de los coeficientes de regresión (pendientes) obtenidos al ajustar
    un modelo lineal entre y_pred_ref y y_pred para cada muestra y cada característica.

    Parámetros:
    - y_pred_ref: ndarray de shape (n_samples, n_timesteps, n_features)
    - y_pred:     ndarray de shape (n_samples, n_timesteps, n_features)
    - feature_indices: lista de índices de características a graficar
    - feature_names:    lista opcional con nombres de características
    - bins:             número de bins del histograma (o 'auto')
    - invert_x:         booleano, si True invierte el eje x 
    - xlim:             tupla (xmin, xmax) para limitar el eje x, None para no limitar
    """
    n_feats = len(feature_indices)
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5), sharey=True)
    if n_feats == 1:
        axes = [axes]

    for ax, feat_idx in zip(axes, feature_indices):
        coefs = []
        # Para cada muestra, ajustamos regresión lineal univariada
        for i in range(y_pred_ref.shape[0]):
            X = y_pred_ref[i, :, feat_idx:feat_idx+1]
            y = y_pred[i, :, feat_idx:feat_idx+1]
            reg = LinearRegression().fit(X, y)
            coefs.append(reg.coef_[0, 0])
        coefs = np.array(coefs)

        name = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
        ax.hist(coefs, bins=bins, density=True, facecolor='royalblue', edgecolor='black', alpha=0.5)
        ax.set_title(f"Histogram Regression\n{name}")
        ax.set_xlabel("Coeficiente de regresión")
        ax.set_ylabel("Densidad")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True, linestyle='--', alpha=0.5)
        if invert_x:
            ax.invert_xaxis()

    plt.tight_layout()
    plt.show()

    

##############################################################################

def plot_predictions(y_true, y_pred, title='LSTM\nPrediction', dt=1.0, 
                     sample_indices=None, feature_idx=0, max_plots=None, 
                     type_feature='Displacement [m]', window_stride=None,
                     show_original=True):
    """
    Grafica predicciones comparadas con los valores reales.

    Parámetros:
    - y_true: serie original completa (shape: [samples, timesteps, features] o [samples, timesteps])
    - y_pred: predicciones del modelo (remuestreadas si fueron por el modelo LSTM-s) (shape: [samples, timesteps, features] o [samples, timesteps])
    - title: título base para el gráfico
    - dt: paso temporal (correspondiente a la serie original) (default = 1.0)
    - sample_indices: lista de índices de muestras a graficar (por defecto todas)
    - feature_idx: índice de la característica (feature) a graficar (default = 0)
    - max_plots: número máximo de gráficos a mostrar (opcional)
    - type_feature: nombre descriptivo del eje Y
    - window_stride: si se usaron ventanas al predecir, define el salto real de los datos originales (cada cuántos pasos se remuestreó la serie original)
    - show_original: si True, grafica la señal completa como fondo
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 2:
        y_true = np.expand_dims(y_true, axis=-1)
    if y_pred.ndim == 2:
        y_pred = np.expand_dims(y_pred, axis=-1)

    same_shape = y_true.shape == y_pred.shape
    if not same_shape and window_stride is None:
        print("Debe especificarse 'window_stride' si las dimensiones de y_true y y_pred no coinciden.")
        return

    num_samples = y_true.shape[0]
    if sample_indices is None:
        sample_indices = list(range(num_samples))
    else:
        sample_indices = [i for i in sample_indices if i < num_samples]

    if max_plots is not None:
        sample_indices = sample_indices[:max_plots]

    for i in sample_indices:
        y_full = y_true[i][:, feature_idx]
        time_full = np.arange(len(y_full)) * dt

        if same_shape:
            y_plot = y_pred[i][:, feature_idx]
            y_ref = y_true[i][:, feature_idx]
            time_axis = time_full
        else:
            y_ref = y_full[window_stride-1::window_stride]
            y_plot = y_pred[i][:, feature_idx]
            time_axis = np.arange(len(y_plot)) * (dt * window_stride)

            # Truncar si hay diferencia mínima por padding, etc.
            min_len = min(len(y_plot), len(y_ref), len(time_axis))
            y_plot = y_plot[:min_len]
            y_ref = y_ref[:min_len]
            time_axis = time_axis[:min_len]

        plt.figure(figsize=(10, 4))
        if show_original:
            plt.plot(time_full, y_full, color='gray', alpha=0.3, label='Original')

        plt.plot(time_axis, y_ref, label='True', color='black')
        plt.plot(time_axis, y_plot, label='Predict', linestyle='--', color='red')

        plt.title(f'{title} Sample {i}')
        plt.xlabel('Time [s]')
        plt.ylabel(type_feature)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        