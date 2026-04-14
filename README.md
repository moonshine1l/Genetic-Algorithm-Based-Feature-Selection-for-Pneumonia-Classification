
# Genetic Algorithm-Based Feature Selection for Pneumonia Classification

This project applies a Genetic Algorithm to select the most informative features extracted from chest X-ray images for pneumonia classification. Instead of using all 512 features produced by a pretrained ResNet18 model, the algorithm evolves optimal subsets of features that maximize classification accuracy while minimizing feature count. Using the Chest X-Ray Images (Pneumonia) dataset from Kaggle, we demonstrate that a compact, GA-selected feature set can achieve performance comparable to or better than using all features, while reducing computational cost and inference time.

## 📌 Project Overview

This project demonstrates how **Genetic Algorithms (GA)** can be used for **feature selection** to improve pneumonia classification from chest X-ray images.

Modern Convolutional Neural Networks extract hundreds of features from images, many of which are redundant or irrelevant. Instead of using all extracted features, we evolve an **optimal subset of features** using a Genetic Algorithm that maximizes classification performance while minimizing feature count.

The goal is to build a **faster**, **simpler**, and **equally (or more) accurate** model for pneumonia detection.

---

## 🧠 Key Idea

We combine:

* **Transfer Learning** using **ResNet18** as a feature extractor
* **Genetic Algorithm** for optimal feature subset selection
* **Logistic Regression** as a lightweight classifier
* Medical dataset: **Chest X-Ray Images (Pneumonia)** from **Kaggle**

---

## ⚙️ Pipeline

### 1. Feature Extraction

* Pretrained ResNet18 with classifier removed
* Each X-ray image → 512-dimensional feature vector

### 2. Genetic Algorithm

* Chromosome: binary vector of length 512
* Fitness: Logistic Regression accuracy − penalty for too many features
* Operators:

  * Tournament selection
  * Single-point crossover
  * Bit-flip mutation

### 3. Evaluation

We compare three models:

| Model                | Description                                |
| -------------------- | ------------------------------------------ |
| Full CNN             | End-to-end ResNet18 classifier             |
| All Features         | Logistic Regression on all 512 features    |
| GA Selected Features | Logistic Regression on GA-optimized subset |

Metrics:

* Accuracy
* F1-score
* Inference time
* Number of features used

---

## 🗂 Dataset

**Chest X-Ray Images (Pneumonia)**
Source: Kaggle

* 5,863 pediatric chest X-ray images
* Classes: NORMAL, PNEUMONIA
* Images resized to 224×224

Download: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🧬 Genetic Algorithm Details

| Component  | Description                           |
| ---------- | ------------------------------------- |
| Chromosome | 512-bit binary vector                 |
| Population | N chromosomes                         |
| Fitness    | Validation accuracy − α × (#features) |
| Selection  | Tournament                            |
| Crossover  | Single-point                          |
| Mutation   | Bit-flip                              |

---

## 📊 Expected Outcome

The GA should discover that only a **small subset of features** is required to classify pneumonia effectively, reducing computational cost and potentially improving generalization.

---

## 🛠 Tech Stack

* Python
* PyTorch
* Scikit-learn
* NumPy / Matplotlib

---

## 🚀 How to Run

```bash
# 1. Extract features
python extract_features.py

# 2. Run genetic algorithm
python ga_feature_selection.py

# 3. Train final classifier
python train_selected_features.py
```

---

## 👥 Authors

* Amaliya Kharisova
* Irina Napalkova
* Muhammadjon Aslonov

Nature Inspired Computing Project — 2026

---

## 📚 References

* ResNet paper (CVPR 2016)
* Kaggle Pneumonia Dataset
* Lecture notes on Nature Inspired Computing

---

Если хочешь, дальше можем:

* сделать **README под твой реальный код (по файлам репозитория)**,
* или написать ещё **Report / Introduction / Methodology** на основе этого.
