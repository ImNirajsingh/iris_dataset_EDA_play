# 🌸 Iris Flower Classification: EDA & Machine Learning Pipeline

An interactive Python Jupyter Notebook dedicated to exploring, visualizing, and preparing the classic Iris dataset (`Iris.csv`) for machine learning and deep learning classification models. 

This project establishes a clean workflow-from initial data ingestion and exploratory data analysis (EDA) using **Seaborn** to setting up classification pipelines with **Scikit-Learn** and **TensorFlow/Keras**.

---

## 📌 Project Overview

The objective of this notebook is to investigate the morphological characteristics of Iris flowers and prepare neural network and linear classifiers to categorize samples into three biological species:
* ***Iris-setosa***
* ***Iris-versicolor***
* ***Iris-virginica***

---

## 🛠️ Technology Stack & Libraries

The notebook runs on a **Python 3** kernel and relies on the following core data science and machine learning packages:

* **Data Manipulation & Math:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib.pyplot`, `seaborn`
* **Machine Learning (Scikit-Learn):**
  * Model Selection & Preprocessing: `train_test_split`, `LabelEncoder`, `StandardScaler`
  * Linear Models: `Perceptron`
  * Evaluation Metrics: `accuracy_score`, `confusion_matrix`, `classification_report`
* **Deep Learning (TensorFlow / Keras):** `tensorflow`, `Sequential`, `Dense`, `to_categorical`

---

## 📊 Dataset Overview

The project loads data directly from `Iris.csv`, consisting of **150 total entries** across **6 columns** with zero null values. 

| Column Name | Data Type | Non-Null Count | Description |
| :--- | :--- | :--- | :--- |
| **Id** | `int64` | 150 | Unique sample identifier (`1` to `150`) |
| **SepalLengthCm** | `float64` | 150 | Sepal length measured in centimeters |
| **SepalWidthCm** | `float64` | 150 | Sepal width measured in centimeters |
| **PetalLengthCm** | `float64` | 150 | Petal length measured in centimeters |
| **PetalWidthCm** | `float64` | 150 | Petal width measured in centimeters |
| **Species** | `object` | 150 | Categorical target class |

### 🏷️ Class Balance
A frequency breakdown via `value_counts()` confirms that the dataset is perfectly balanced across all three target classes:
* **Iris-setosa:** 50 samples
* **Iris-versicolor:** 50 samples
* **Iris-virginica:** 50 samples

---

## 🚀 Notebook Workflow

### 1. Environment Setup & Preprocessing Imports
* Configures Python warning filters (`warnings.filterwarnings('ignore')`) for clean output display.
* Pre-loads Scikit-learn preprocessing utilities (`StandardScaler`, `LabelEncoder`) and TensorFlow Keras modules (`Sequential`, `Dense`) to facilitate multi-class classification.

### 2. Data Loading & Structural Inspection
* Ingests `Iris.csv` into a pandas DataFrame (`df`).
* Evaluates data integrity and memory usage (~7.2+ KB) using `df.info()`.
* Verifies class balance via `df['Species'].value_counts()`.

### 3. Exploratory Data Analysis (EDA)
* Generates a multi-variable scatter plot matrix via `sns.pairplot(df, hue='Species')`.
* Color-encodes data points by species to highlight linear cluster separability (particularly separating *Iris-setosa* from the other two species across petal dimensions).

---

## 💻 How to Run the Notebook Locally

1. **Clone or Download the Repository:** Ensure `Iris.csv` and the `.ipynb` notebook file are in the same working directory.
2. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
