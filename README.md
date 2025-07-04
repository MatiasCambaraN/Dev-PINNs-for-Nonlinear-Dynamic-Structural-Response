Aquí tienes la **traducción completa al inglés** manteniendo toda la estructura, formato y contenido original:

---

# Deep Learning Model Comparison for Structural Dynamic Response

This project aims to compare various deep learning models proposed by an author to predict the dynamic response of structures under seismic loads, using different synthetic and real datasets. It focuses on a modular, reusable, and easily scalable implementation.

---

## Project Structure

```plaintext
ComparacionModelos/
│
├── configs/                     # YAML configurations per model
│   └── multiphylstm.yaml
│
├── data/                        # Structural datasets
│   ├── raw/                     # Original datasets
│   └── processed/               # Model-specific prepared datasets
│
├── logs/                        # Training log files
│
├── models/                      # Model implementations
│   ├── deeplstm/
│   ├── phycnn/
│   └── phylstm/
│
├── notebooks/                   # Jupyter notebooks for experimentation
│
├── results/                     # Obtained results
│   ├── figures/                 # Comparative plots and visualizations
│   ├── results_deeplstm/
│   ├── results_phycnn/
│   └── results_phylstm/
│
├── utils/                       # Auxiliary functions (preprocessing, metrics, etc.)
│
└── README.md                    # This file
```

---

## Compared Models

* **DeepLSTM**

  * `LSTM-f`: Full sequence-to-sequence
  * `LSTM-s`: Sub-sequence windowing
* **PhyCNN**

  * `PhyCNN-$x,\dot{x},g$`
  * `PhyCNN-$\ddot{x}$`
* **PhyLSTM**

  * `PhyLSTM²`: double architecture
  * `PhyLSTM³`: triple architecture

---

## Datasets

* BoucWen-BLW
* Duffing
* BoucWen-5DOF
* MRFDBF-3DOF
* San Bernardino (real accelerograms)

---

---

## Experiment Configuration

The YAML files located in the `configs/` directory define the full Conda virtual environments required to run the project.

Each configuration file contains the necessary Python version, package dependencies, and environment variables to ensure consistent and reproducible execution across platforms.

Example (`multiphylstm.yaml`):

```yaml
name: multiphylstm              # Conda environment name
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.6
  - tensorflow=1.15.5
  - keras=2.3.1
  - numpy=1.19
  - pandas=1.1
  - matplotlib=3.3
  ...
```

> 🔧 For Linux users, TensorFlow with GPU support must be installed manually via pip using the NVIDIA repository.

You can create different YAML files to define environments optimized for various models, datasets, or operating systems. This structure allows seamless switching between configurations and ensures all code runs under the correct set of dependencies.

---

## Usage Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/ComparacionModelos.git 
   cd ComparacionModelos
   ```

2. Create and activate the Conda environment depending on your OS:

   ### 🔷 Linux

   ```bash
   conda env create -f configs/multiphylstm_linux.yaml
   conda activate multiphylstm
   ```

   Then manually install GPU-optimized TensorFlow:

   ```bash
   pip install nvidia-tensorflow[horovod]==1.15.5+nv20.12 --extra-index-url https://pypi.nvidia.com
   ```

   ### 🔶 Windows

   ```bash
   conda env create -f configs/multiphylstm.yaml
   conda activate multiphylstm
   ```

3. Place the datasets in the directory:

   ```
   data/raw/
   ```

4. Preprocess the data using the functions in:

   ```
   utils/preprocessing.py
   ```

5. Run the notebooks or training/evaluation scripts from:

   ```
   notebooks/
   ```

---

## Results

Training and prediction results are stored in `results/`, separated by model. Key visualizations are located in `results/figures/`.

---

## Project Status

* [x] Folder structure
* [x] Modular model separation
* [x] Training scripts and notebooks
* [ ] Automatic hyperparameter tuning for new datasets
* [ ] Final comparison dashboard

---

## Main Dependencies

* **Python 3.6**
* **TensorFlow 1.15.5 (NVIDIA)** — GPU support with CUDA 10.0 / cuDNN 7.6 *(manually installed on Linux)*
* **Keras 2.3.1**
* **NumPy 1.19 / Pandas 1.1 / Matplotlib 3.3**
* **Scikit-learn 0.24**
* **Jupyter Notebook / JupyterLab**
* **Seaborn**
* **PyYAML / H5py / Protobuf**
* **absl-py / gast / six** — internal dependencies required by TensorFlow

> 📦 All these dependencies are included in the Conda environments defined in:
>
> * [`configs/multiphylstm.yaml`](configs/multiphylstm.yaml) (for Windows)
> * [`configs/multiphylstm_linux.yaml`](configs/multiphylstm_linux.yaml) (for Linux)

> ⚠️ Note: On Linux, TensorFlow must be installed manually using pip from NVIDIA's official repository.

---

## Execution Hardware

### 🔷 Laptop (Windows)

* **Operating system**: Windows 11 Home Single Language — v10.0.22631
* **CPU**: Intel(R) Core i7-9750H @ 2.60GHz — 6 cores / 12 threads
* **RAM**: 16 GB (2x 8 GB)
* **GPU**:

  * NVIDIA GeForce GTX 1050 — 4 GB
  * Intel UHD Graphics 630 — 1 GB
* **CUDA**: 10.0 (v10.0.130)
* **cuDNN**: 7.4.1

### 🔶 Desktop PC (Linux)

* **Operating system**: Ubuntu 24.04.2 LTS (Noble Numbat) — Kernel 6.8.0-60-generic
* **CPU**: Intel Core i5-12400 — 6 cores / 12 threads
* **RAM**: 16 GB
* **GPU**: NVIDIA GeForce RTX 3060 — 12 GB
* **CUDA**: 12.0 (v12.0.140)
* **cuDNN**: 9.2.0

---

> ℹ️ This environment was used to train and evaluate some models included in the project.

---


