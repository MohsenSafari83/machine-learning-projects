# Machine Learning Projects

This repository is a **master reference** for my machine learning work.  
It organizes projects into **Foundations**, **Supervised**, and **Unsupervised** learning categories,  
linking to their respective repositories and sub-modules. 

---
##  Machine Learning Projects Portfolio Roadmap

This portfolio provides structured access to all implemented machine learning algorithms and real-world projects, categorized by learning type.

| Main Section | Learning Type | Algorithm / Topic | Project Link |
| :---: | :---: | :---: | :--- |
| **SUPERVISED LEARNING** | **Classification** | **Performance Metrics (Evaluation)** | [Evaluation Metrics Guide](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/Evaluation%20Metrics) |
|([Go to Supervised Learning Repo](https://github.com/MohsenSafari83/Supervised-Learning-)) | **Classification** | **k-Nearest Neighbors (kNN)** | [Breast Cancer Diagnosis ](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/KNN) |
| | Classification | Support Vector Machine (SVM) | [Link to SVM Project]() |
| | Classification | ensemble classifier | [ensemble classifier]() |
| | Classification | Logistic Regression | [Link to Logistic Regression Project](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/Logistic%20Regression) |
| | **Regression** | Linear Regression | [Link to Linear Regression Project](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/Linear%20Regression) |
| --- | --- | --- | --- |
| **UNSUPERVISED LEARNING** | **Clustering** | K-Means/ DBSCAN / Hierarchical | [Link to K-Means Project](https://github.com/MohsenSafari83/Unsupervised-Learning-/tree/main/project) |
| ([Go to Unsupervised Learning Repo](https://github.com/MohsenSafari83/Unsupervised-Learning-)) | **Dimensionality Reduction** | Principal Component Analysis (PCA) | [Link to PCA Project](https://github.com/MohsenSafari83/Unsupervised-Learning-/tree/main/docs) |
| | **Association Rules** | **Apriori Algorithm** | [Apriori Association Rules](https://github.com/MohsenSafari83/Apriori-Association-Rules) |

---

##  Foundations (Machine Learning Basics)

### 🔹 [ml-core-concepts](https://github.com/MohsenSafari83/ml-core-concepts)  
- **Goal:** Cover theoretical foundations of machine learning.  
- **Includes:**  
  - Overfitting vs Underfitting  
  - Bias-Variance Tradeoff  
  - Gradient Descent (Batch, Stochastic, Mini-Batch) & Learning Rate  
  - Loss Functions (MSE, MAE, Hinge Loss)  
  - Distance Measures (Euclidean, Minkowski, Cosine Similarity)  
  - Regularization techniques  
  - Softmax & Cost Functions  
  

### 🔹 [Data Preprocessing](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/Data%20Preprocessing)
- **Goal:** Essential techniques for preparing data before training ML models.  
- **Topics Covered:**
  - Handling missing values & categorical variables
  - Feature scaling (Normalization, Standardization)
  - Model evaluation with cross-validation
  - Feature engineering & selection methods
  - Dimensionality reduction (PCA)
  - Train-test split & avoiding data leakage
  - Pipelines in Scikit-learn for consistent preprocessing

### 🔹 [Python Cheat Sheet](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/Python_Cheat_Sheet)
- **Goal:** Quick reference for essential Python libraries in ML workflows.  
- **Topics Covered:**
  - **NumPy:** Core array manipulation & data management
  - **Matplotlib:** Data visualization (line, scatter, histograms, bar, box, subplots)
  - **Seaborn:** Advanced visualization (distributions, scatter with categories, box/violin plots, pairwise plots)

---

##  Supervised Learning

### 🔹 [Heart Failure Prediction](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/Heart%20Failure%20Prediction)
- **Goal:** Predict heart failure risk using ML models.  
- **Algorithms Used:** Logistic Regression, KNN, SVM, XGBoost, Random Forest  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC  

### 🔹 [Linear Regression on California Housing](https://github.com/your-username/supervised-learning/tree/main/california-housing)  
- **Goal:** Regression model to predict housing prices in California.  
- **Techniques:** Linear Regression, Feature Engineering, RMSE, R².

### 🔹 [kNN Classification Case Study](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/KNN) 
- **Goal:** Comprehensive demonstration of the **k-Nearest Neighbors (kNN)** algorithm, focusing on solutions to its core limitations.
- **Topics Covered:**
  - **Scale Sensitivity:** Demonstrating the necessity of scaling for distance-based algorithms.
  - **Robust Scaling:** Applying **RobustScaler** to effectively manage outliers and variance.
  - **Optimal K Search:** Using **Cross-Validation** and plotting **Error Rate vs. K** to manage the Bias-Variance Tradeoff.
  - **Evaluation Metrics:** In-depth analysis of Precision, **Recall (Sensitivity)**, and the Confusion Matrix in a critical medical context.
### 🔹 [Logistic Regression Case Study](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/Logistic%20Regression) 
- **Goal:** Demonstrating the use of Logistic Regression for **binary classification**, focusing on modeling probability and interpreting feature impact.
- **Topics Covered:**
  - **The Sigmoid Function:** Transforming linear input into a probability between 0 and 1.
  - **Log Loss (Cross-Entropy):** Understanding the cost function for probabilistic models.
  - **Optimization:** Implementing Gradient Descent for finding optimal model weights.
  - **Feature Interpretation:** Analyzing model coefficients to determine the impact and directionality of features (e.g., how cholesterol affects risk).
### 🔹 [Performance Metrics Guide (Evaluation)](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/Evaluation%20Metrics) 
- **Goal:** Comprehensive theoretical and practical guide on evaluating classification models, moving beyond simple accuracy. 
- **Topics Covered:**
  - **Accuracy & Limitations:** Understanding why it fails in imbalanced datasets.
  - **Confusion Matrix (TP, FP, FN, TN):** Understanding the four outcomes and Type I & II errors.
  - **Precision, Recall, F1-Score:** Understanding the trade-off between sensitivity and prediction correctness.
  - **AUC-ROC:** Evaluating the model’s discrimination ability at different thresholds.
  - **Averaging Methods:** Differences between Macro and Micro Averaging in multi-class evaluation.

---

## Unsupervised Learning

### 🔹 [Clustering & Dimensionality Reduction Case Study](https://github.com/MohsenSafari83/Unsupervised-Learning-/tree/main/project) 
- **Goal:** Comprehensive demonstration of core unsupervised learning techniques for discovering hidden patterns and efficient data representation.
- **Topics Covered:**
  - **K-Means Clustering:** Understanding the algorithm, initialization, and iterative process.
  - **Optimal K Search:** Using the **Elbow Method** and **Silhouette Score** to find the optimal number of clusters.
  - **Dimensionality Reduction:** Applying techniques like PCA to visualize high-dimensional data and improve model efficiency.
  - **Unlabeled Data Analysis:** Interpreting the results to derive business insights (e.g., customer segmentation).

### 🔹 [Clustering & Dimensional Reduction Guide](https://github.com/your-username/unsupervised-learning/tree/main/clustering-dim-reduction)  
- **Goal:** Walkthrough of common clustering and dimensionality reduction techniques.  
- **Techniques:** PCA, t-SNE, Hierarchical Clustering.  

### 🔹 [Dimensionality Reduction (PCA) Guide](https://github.com/MohsenSafari83/Unsupervised-Learning-/tree/main/docs) 
- **Goal:** Comprehensive documentation and guide on **Principal Component Analysis (PCA)** and its application in machine learning workflows.
- **Topics Covered:**
  - **The Curse of Dimensionality:** Understanding the challenges of high-dimensional data.
  - **PCA Mechanics:** Explaining eigenvalues, eigenvectors, and variance retention.
  - **Visualization:** Using PCA to project high-dimensional data onto 2D or 3D space.
  - **Application:** Using PCA for noise reduction and improving model efficiency.
### 🔹 [Association Rules Mining (Apriori)](https://github.com/MohsenSafari83/Apriori-Association-Rules) 
- **Goal:** Demonstrating **Apriori Algorithm** for discovering frequent itemsets and generating meaningful **association rules** from transaction data (e.g., market basket analysis).
- **Topics Covered:**
  - **Support, Confidence, and Lift:** Understanding the core metrics used to evaluate rule strength.
  - **Frequent Itemset Generation:** Implementing the Apriori principle to efficiently mine patterns.
  - **Rule Interpretation:** Analyzing rules to derive actionable insights (e.g., product placement strategies).

---

## 🌳 Project Tree Overview
```
machine-learning-projects
│
├── Foundations
│ ├── ml-core-concepts
│ ├── Data Preprocessing
│ └── Python Cheat Sheet
│
├── Supervised Learning
│ ├── Heart Failure Prediction
| ├── Logistic Regression Case Study
| ├── kNN Classification Case Study
| ├── Performance Metrics guide
│ └── Linear Regression on California Housing
│
└── Unsupervised Learning
├── Customer Segmentation
├── Clustering & Dimensional Reduction Guide
└── Apriori Association Rules
```

---

## Goal

This repository acts as a **map of my ML journey**,  
making it easy to navigate through foundational concepts, supervised learning, and unsupervised learning projects.

