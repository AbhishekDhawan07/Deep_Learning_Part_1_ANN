<div align="center">

# 🧠 Deep Learning - Part 1: Artificial Neural Networks (ANN)

### Foundations of ANN - From Scratch to Iris Flower Classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A hands-on deep learning series covering ANN fundamentals - binary classification from scratch on a custom dataset, followed by multi-class Iris flower classification using a full Keras Sequential model.**

</div>

---

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 About the Project

**Deep Learning - Part 1** is the foundational module of a deep learning series focused on **Artificial Neural Networks (ANN)**. This part covers two progressive notebooks:

| Notebook | Focus |
|----------|-------|
| 📓 `Basics_of_ANN_Implementation.ipynb` | ANN fundamentals - binary classification on a custom soil/plant watering dataset using a 2-layer Keras model |
| 🌸 `Iris_Flower_Classification_using_ANN.ipynb` | Multi-class classification - Iris species prediction using a 3-layer ANN with Perceptron baseline comparison |

The series is designed for learners who want to **understand ANN architecture from the ground up** - covering data preprocessing, model building, training, evaluation, and visualization.

> 💡 Both notebooks are self-contained and progressively build upon each other - start with the Basics notebook before moving to the Iris classification.

---

## 🌐 Demo

### 📓 Notebook 1 — Basics of ANN Implementation

**Problem:** Predict whether a plant `needs_water` based on soil moisture, temperature, and sunlight hours.

**Architecture:**
```
Input (3 features) -> Dense(8, ReLU) -> Dense(1, Sigmoid) -> Binary Output
```

**Key Steps:**
- Custom tabular dataset with 16 samples
- Min-Max normalization
- Train/Test split with stratification
- Binary cross-entropy loss + Adam optimizer
- SGD with momentum exploration

---

### 🌸 Notebook 2 — Iris Flower Classification using ANN

**Problem:** Classify iris flowers into 3 species — *Iris-setosa*, *Iris-versicolor*, *Iris-virginica* — using 4 petal/sepal features.

**Architecture:**
```
Input (4 features) -> Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(3, Softmax) -> 3-class Output
```

**Key Steps:**
- EDA with `seaborn` pairplot
- Label encoding + Standard scaling
- Perceptron baseline -> ANN upgrade
- Categorical cross-entropy + Adam
- Training/Validation accuracy curve visualization

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10** | Core language |
| **TensorFlow / Keras** | ANN model building, training, and evaluation |
| **Scikit-Learn** | Train/test split, LabelEncoder, StandardScaler, Perceptron, metrics |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Array operations and normalization |
| **Matplotlib** | Training accuracy/loss curve plots |
| **Seaborn** | EDA pairplot visualization |

---

## ✨ Features

<details open>
<summary><b>📓 Basics of ANN Implementation</b></summary>
<br/>

- Custom plant-watering binary classification dataset (16 samples, 3 features)
- **Manual Min-Max normalization** without sklearn
- **Keras Sequential model** — Input → Dense(8, ReLU) → Dense(1, Sigmoid)
- Compiled with **Adam optimizer** + Binary Cross-Entropy loss
- Exploration of **SGD with momentum** (`learning_rate=0.01`, `momentum=0.9`)
- Stratified train/test split (75/25)

</details>

<details open>
<summary><b>🌸 Iris Flower Classification using ANN</b></summary>
<br/>

- Full EDA using **seaborn pairplot** with species hue
- **LabelEncoder** for target encoding + **StandardScaler** for feature scaling
- **Perceptron baseline** — classical linear classifier for comparison
- **3-layer ANN** — Dense(16) → Dense(8) → Dense(3, Softmax)
- **One-hot encoding** via `to_categorical` for multi-class targets
- Training with **validation split** (20%) across 100 epochs, batch size 8
- **Accuracy curves** plotted for train vs. validation performance

</details>

<details open>
<summary><b>📊 Evaluation & Metrics</b></summary>
<br/>

- `accuracy_score` and `classification_report` for Perceptron baseline
- `model.evaluate()` for ANN test accuracy
- Side-by-side training/validation accuracy plot
- `model.summary()` for architecture inspection

</details>

---

## 📊 Dataset

### Iris Dataset (`Iris.csv`)

| Property | Value |
|----------|-------|
| **Rows** | 150 |
| **Features** | 4 (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) |
| **Target** | Species (3 classes) |
| **Classes** | Iris-setosa, Iris-versicolor, Iris-virginica |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) |

```
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
```

### Custom Plant Dataset (Basics Notebook)

| Feature | Description |
|---------|-------------|
| `soil_moisture` | Moisture level (0.0 – 1.0) |
| `temperature_c` | Temperature in Celsius |
| `sunlight_hours` | Daily sunlight exposure |
| `needs_water` | Binary target (0 = No, 1 = Yes) |

---

## 🧠 How It Works

### Notebook 1 — ANN Binary Classification Pipeline

```
  Raw Tabular Data (16 samples)
        │
        ▼
  ┌──────────────────────────┐
  │  Min-Max Normalization   │  ← Manual scaling (no sklearn)
  └──────────────────────────┘
        │
        ▼
  ┌──────────────────────────┐
  │  Train / Test Split      │  ← 75/25, stratified
  └──────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Keras Sequential Model                  │
  │  Input(3) → Dense(8, ReLU)               │
  │           → Dense(1, Sigmoid)            │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Compile: Adam + Binary Cross-Entropy    │
  │  Fit: 100 epochs, full-batch             │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────┐
  │  SGD + Momentum Tuning   │  ← lr=0.01, momentum=0.9
  └──────────────────────────┘
```

---

### Notebook 2 — Iris ANN Classification Pipeline

```
  Iris.csv (150 samples, 4 features)
        │
        ▼
  ┌──────────────────────────┐
  │  EDA — Seaborn Pairplot  │
  └──────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Preprocessing                           │
  │  LabelEncoder (Species → 0,1,2)          │
  │  StandardScaler (X_train, X_test)        │
  │  to_categorical (y → one-hot)            │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Perceptron Baseline                     │  ← accuracy_score + classification_report
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Keras ANN                               │
  │  Dense(16, ReLU) → Dense(8, ReLU)        │
  │  → Dense(3, Softmax)                     │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  Compile: Adam + Categorical CE          │
  │  Fit: 100 epochs, batch=8, val_split=0.2 │
  └──────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────┐
  │  Accuracy Curve Plot     │  ← Train vs Validation Accuracy
  └──────────────────────────┘
```

---

## 📁 Project Structure

```
Deep_Learning_Part_1/
│
├── 📓 Basics_of_ANN_Implementation.ipynb                        # ANN binary classification from scratch
├── 🌸 Iris_Flower_Classification_using_Artificial_Neural_        # Multi-class ANN on Iris dataset
│      Network_ANN_.ipynb
│── 📊 Iris.csv                                                    # Iris flower dataset (150 samples)
|                                              
│
└── 📖 README.md                                                  # This file
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `Basics_of_ANN_Implementation.ipynb` | Introduces ANN fundamentals — custom dataset, manual normalization, binary classification with Keras, SGD vs Adam exploration |
| `Iris_Flower_Classification_using_ANN_.ipynb` | Full pipeline — EDA, preprocessing, Perceptron baseline, 3-layer ANN, evaluation, accuracy plots |
| `Iris.csv` | 150-sample Iris dataset with 4 features and 3 target species classes |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab
- pip

### 1 — Clone the Repository

```bash
git clone https://github.com/YourUsername/Deep_Learning_Part_1.git
cd Deep_Learning_Part_1
```

### 2 — Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
```

### 3 — Launch Jupyter

```bash
jupyter notebook
```

### 4 — Run the Notebooks (in order)

**Start here:**
```
Basics_of_ANN_Implementation.ipynb
```
**Then move to:**
```
Iris_Flower_Classification_using_Artificial_Neural_Network_ANN_.ipynb
```

> ⚠️ Make sure `Iris.csv` is in the same directory as the notebooks before running the Iris classification notebook.

---

## 🤝 Contributing

Contributions are welcome! If you have improvements, additional experiments, or new notebooks to add:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Made with ❤️ for Deep Learning enthusiasts and beginners**

⭐ If this helped your learning journey, please give it a star!

</div>
