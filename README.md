# 🧠 Comparación de Modelos de Deep Learning para Respuesta Dinámica Estructural

Este proyecto tiene como objetivo comparar diversos modelos de deep learning propuestos por un autor para predecir la respuesta dinámica de estructuras ante cargas sísmicas, utilizando diferentes datasets sintéticos y reales. Se enfoca en una implementación modular, reutilizable y fácilmente escalable.

---

## 📁 Estructura del Proyecto

```plaintext
ComparacionModelos/
│
├── configs/                     # Configuraciones YAML por modelo
│   └── multiphylstm.yaml
│
├── data/                        # Datos estructurales
│   ├── raw/                    # Datasets originales
│   └── processed/              # Datasets preparados por modelo
│
├── logs/                        # Archivos de logs de entrenamiento
│
├── models/                      # Implementaciones de modelos
│   ├── deeplstm/
│   ├── phycnn/
│   └── phylstm/
│
├── notebooks/                   # Notebooks Jupyter de experimentación
│
├── results/                     # Resultados obtenidos
│   ├── figures/                # Gráficas comparativas y visualizaciones
│   ├── results_deeplstm/
│   ├── results_phycnn/
│   └── results_phylstm/
│
├── utils/                       # Funciones auxiliares (preprocesamiento, métricas, etc.)
│
└── README.md                    # Este archivo
```

---

## 📚 Modelos Comparados

* **DeepLSTM**

  * `LSTM-f`: Full sequence-to-sequence
  * `LSTM-s`: Sub-sequence windowing
* **PhyCNN**

  * `PhyCNN-$x,\dot{x},g$`
  * `PhyCNN-$\ddot{x}$`
* **PhyLSTM**

  * `PhyLSTM²`: arquitectura doble
  * `PhyLSTM³`: arquitectura triple

---

## 📊 Bases de Datos

* BoucWen-BLW
* Duffing
* BoucWen-5DOF
* MRFDBF-3DOF
* San Bernardino (acelerogramas reales)

---

## ⚙️ Configuración de Experimentos

Los archivos YAML dentro de `configs/` permiten especificar hiperparámetros, rutas, formato de datos, etc.
Ejemplo (`multiphylstm.yaml`):

```yaml
model:
  name: PhyLSTM3
  hidden_units: 128
  lstm_layers: 3
  fc_layers: 2
  dropout: 0.2

training:
  epochs: 100
  batch_size: 32
  optimizer: adam
  learning_rate: 0.001
```

---

## 🚀 Instrucciones de Uso

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/ComparacionModelos.git 
   cd ComparacionModelos
   ```

2. Crea y activa el entorno Conda según tu sistema operativo:

   ### 🔷 Linux

   ```bash
   conda env create -f configs/multiphylstm_linux.yaml
   conda activate multiphylstm
   ```

   Luego, instala manualmente TensorFlow optimizado para GPU:

   ```bash
   pip install nvidia-tensorflow[horovod]==1.15.5+nv20.12 --extra-index-url https://pypi.nvidia.com
   ```

   ### 🔶 Windows

   ```bash
   conda env create -f configs/multiphylstm.yaml
   conda activate multiphylstm
   ```

3. Coloca los datasets en el directorio:

   ```
   data/raw/
   ```

4. Preprocesa los datos usando las funciones en:

   ```
   utils/preprocessing.py
   ```

5. Ejecuta los notebooks o scripts para entrenamiento y evaluación desde:

   ```
   notebooks/
   ```

---

## 🧪 Resultados

Los resultados de entrenamiento y predicción se almacenan en `results/` separados por modelo. Las visualizaciones clave están en `results/figures/`.

---

## ✅ Estado del Proyecto

* [x] Estructura de carpetas
* [x] Separación modular de modelos
* [x] Scripts y notebooks de entrenamiento
* [ ] Validación cruzada automática
* [ ] Dashboard de comparación final

---

Claro, aquí tienes la sección modificada para reflejar que ahora utilizas dos archivos YAML (uno para Linux y otro para Windows), e incluyendo una nota clara sobre la instalación manual de TensorFlow en Linux:

---

## 🧰 Dependencias Principales

* **Python 3.6**
* **TensorFlow 1.15.5 (NVIDIA)** — soporte para GPU con CUDA 10.0 / cuDNN 7.6 *(se instala manualmente en Linux)*
* **Keras 2.3.1**
* **NumPy 1.19 / Pandas 1.1 / Matplotlib 3.3**
* **Scikit-learn 0.24**
* **Jupyter Notebook / JupyterLab**
* **Seaborn**
* **PyYAML / H5py / Protobuf**
* **absl-py / gast / six** — dependencias internas requeridas por TensorFlow

> 📦 Todas estas dependencias están incluidas en los entornos Conda definidos en:
>
> * [`configs/multiphylstm.yaml`](configs/multiphylstm.yaml) (para Windows)
> * [`configs/multiphylstm_linux.yaml`](configs/multiphylstm_linux.yaml) (para Linux)

> ⚠️ Nota: En Linux, TensorFlow debe instalarse manualmente con pip desde el repositorio oficial de NVIDIA.

---

Perfecto. Con esta información, aquí tienes el bloque formateado en Markdown para incluir directamente en tu `README.md` bajo una sección como:

---

## 💻 Hardware de Ejecución

### 🔷 Laptop (Windows)

* **Sistema operativo**: Windows 11 Home Single Language — v10.0.22631
* **CPU**: Intel(R) Core i7-9750H @ 2.60GHz — 6 núcleos / 12 hilos
* **RAM**: 16 GB (2x 8 GB)
* **GPU**:

  * NVIDIA GeForce GTX 1050 — 4 GB
  * Intel UHD Graphics 630 — 1 GB
* **CUDA**: 10.0 (v10.0.130)
* **cuDNN**: 7.4.1

### 🔶 PC de Escritorio (Linux)

* **Sistema operativo**: Ubuntu 24.04.2 LTS (Noble Numbat) — Kernel 6.8.0-60-generic
* **CPU**: Intel Core i5-12400 — 6 núcleos / 12 hilos
* **RAM**: 16 GB
* **GPU**: NVIDIA GeForce RTX 3060 — 12 GB
* **CUDA**: 12.0 (v12.0.140)
* **cuDNN**: 9.2.0

---

> ℹ️ Este entorno fue utilizado para entrenar y evaluar algunos modelos incluidos en el proyecto.

