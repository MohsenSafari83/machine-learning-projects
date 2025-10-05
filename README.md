
 

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
|([Go to Supervised Learning Repo](https://github.com/MohsenSafari83/Supervised-Learning-)) | Classification and Regression| **k-Nearest Neighbors (kNN)** | [Breast Cancer Diagnosis ](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/KNN) |
| | Classification and Regression | Support Vector Machine (SVM) | [Link to SVM Project](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/Support%20Vector%20Machines%20) |
| | Classification | ensemble classifier | [ensemble classifier]() |
| | Classification | Logistic Regression | [Link to Logistic Regression Project](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/Logistic%20Regression) |
| | **Regression** | Linear Regression | [Link to Linear Regression Project](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/Linear%20Regression) |
| --- | --- | --- | --- |
| **UNSUPERVISED LEARNING** | **Clustering** | K-Means/ DBSCAN / Hierarchical | [Link to K-Means Project](https://github.com/MohsenSafari83/Unsupervised-Learning-/tree/main/project) |
| ([Go to Unsupervised Learning Repo](https://github.com/MohsenSafari83/Unsupervised-Learning-)) | **Dimensionality Reduction** | Principal Component Analysis (PCA) | [Link to PCA Project](https://github.com/MohsenSafari83/Unsupervised-Learning-/tree/main/docs) |
| | **Association Rules** | **Apriori Algorithm** | [Apriori Association Rules](https://github.com/MohsenSafari83/Apriori-Association-Rules) |

---
# Quick Reference: Core ML Algorithms (Theoretical Overview)

This table provides a high-level summary of the **functionality**, **methodology**, and **common use cases** for foundational algorithms.

---

## 🔹 Supervised Learning

| Algorithm               | Type            | Purpose                                      | Methodology                                                                 | Common Use Cases                                                                 |
|--------------------------|-----------------|----------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Linear Regression**    | Regression      | Predict continuous output values             | Fits a linear equation minimizing sum of squared residuals                  | Predicting continuous values (e.g., housing prices, sales forecasting)            |
| **Logistic Regression**  | Classification | Predict binary output variable               | Logistic (sigmoid) function applied to linear relationship                  | Binary classification (e.g., heart failure risk, churn prediction)                |
| **Decision Trees**       | Both            | Model decisions and outcomes                 | Recursive tree-like structure with decision nodes and outcomes              | Classification & Regression tasks, interpretable models                           |
| **Random Forests**       | Both            | Improve accuracy via ensemble learning       | Combines multiple decision trees with bagging (majority vote/average)       | Reducing overfitting, improving prediction accuracy                               |
| **Support Vector Machine (SVM)** | Both | Separate classes or predict continuous values | Finds hyperplane maximizing margin (classification) or regression function | Classification & Regression, effective in high-dimensional spaces                 |
| **K-Nearest Neighbors (KNN)** | Both | Predict class/value from nearest data points | Uses distance metric to find *k* closest neighbors (majority/average vote)  | Classification & Regression, sensitive to noisy or unscaled data                  |
| **Gradient Boosting**    | Both            | Build strong learner from weak models        | Iteratively corrects errors of prior models (boosting, ensemble method)     | High-performance Classification & Regression tasks                                |
| **Naive Bayes**          | Classification | Predict class with independence assumption   | Bayes’ theorem assuming feature independence                                | Text classification, spam filtering, sentiment analysis, medical diagnosis        |

---

## 🔹 Unsupervised Learning

| Algorithm                  | Type                        | Purpose                                      | Methodology                                                                 | Common Use Cases                                                                 |
|-----------------------------|-----------------------------|----------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **K-Means**                 | Clustering (Unsupervised)  | Group similar data points into clusters       | Iteratively minimizes sum of squared distances to cluster centroids         | Customer segmentation, document clustering, image compression                    |
| **DBSCAN**                  | Clustering (Unsupervised)  | Identify dense regions and mark outliers      | Groups points by density reachability, no need to predefine K               | Outlier detection, spatial clustering, noise identification                      |
| **Hierarchical Clustering** | Clustering (Unsupervised)  | Build a hierarchy of clusters (dendrogram)    | Agglomerative (bottom-up) or Divisive (top-down) cluster merging/splitting  | Taxonomy creation, genetic sequencing, data structure visualization              |
| **PCA (Principal Component Analysis)** | Dimensionality Reduction | Reduce number of features while retaining variance | Projects high-dimensional data onto lower-dimensional orthogonal components | Data compression, noise reduction, 2D/3D visualization                           |
| **Apriori**                 | Association Rules (Unsupervised) | Discover frequent itemsets & generate rules | Uses "apriori property" to mine frequent patterns efficiently               | Market basket analysis, cross-selling, recommendation systems, web usage mining  |

---
# Supervised Learning: Applications, Advantages and Disadvantages

This table outlines the **common applications**, **key advantages**, and **major challenges** associated with **Supervised Learning algorithms**.

---

## Applications

| Industry / Domain             | Application                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| **Banking (Fraud Detection)**  | Detects fraudulent vs. legitimate transactions using labeled transaction data. |
| **Healthcare (Parkinson’s Disease Prediction)** | Predicts progression of Parkinson’s disease using patient medical data. |
| **Telecom / SaaS (Customer Churn Prediction)** | Identifies features associated with churn to predict customer retention. |
| **Medical Diagnosis (Cancer Cell Classification)** | Classifies cancer cells as **malignant** or **benign** based on their features. |
| **Finance (Stock Price Prediction)** | Predicts buy/sell signals for stocks using historical financial data. |

---

## Advantages of Supervised Learning

| Advantage            | Detail                                                                 |
|-----------------------|------------------------------------------------------------------------|
| **Simplicity & Clarity** | Easy to understand and implement since it learns from labeled examples. |
| **High Accuracy**        | Achieves strong predictive performance when sufficient labeled data is available. |
| **Versatility**          | Works for both **classification** (spam detection, disease prediction) and **regression** (price forecasting). |
| **Generalization**       | With diverse training data, models generalize well to unseen inputs. |
| **Wide Application**     | Applied in **speech recognition, medical diagnosis, sentiment analysis, fraud detection**, and more. |

---

## Disadvantages of Supervised Learning

| Disadvantage          | Detail                                                                 |
|------------------------|------------------------------------------------------------------------|
| **Requires Labeled Data** | Preparing large labeled datasets is costly and time-consuming.       |
| **Bias from Data**        | Biased/unbalanced datasets may lead to biased predictions.           |
| **Overfitting Risk**      | Small datasets may cause models to memorize instead of generalize.   |
| **Limited Adaptability**  | Performance drops on data distributions very different from training data. |
| **Not Scalable for Some Problems** | In tasks with millions of labels (e.g., NLP), supervised labeling is impractical. |

---
# Unsupervised Learning: Applications, Advantages and Challenges

This table outlines the **diverse uses**, **key benefits**, and **critical challenges** associated with **Unsupervised Learning algorithms**.

---

## Applications

| Key Concept              | Description / Detail                                                                 |
|---------------------------|-------------------------------------------------------------------------------------|
| **Customer Segmentation** | Clusters customers by behavior or demographics, enabling **targeted marketing**.    |
| **Anomaly Detection**     | Identifies unusual patterns in data → **fraud detection, cybersecurity, equipment failure prevention**. |
| **Recommendation Systems**| Suggests products, movies, or music by analyzing user **behavior & preferences**.   |
| **Image & Text Clustering** | Groups similar images/documents for **organization, classification, content recommendation**. |
| **Social Network Analysis** | Detects **communities or trends** in social media interactions.                    |

---

## Advantages of Unsupervised Learning

| Key Concept                 | Description / Detail                                                        |
|------------------------------|----------------------------------------------------------------------------|
| **No Labeled Data Needed**  | Works with **raw, unlabeled data**, reducing time & cost of annotation.     |
| **Discovers Hidden Patterns** | Finds **natural groupings** that might be missed by human analysis.        |
| **Handles Complex Datasets** | Effective for **high-dimensional or massive datasets**.                     |
| **Useful for Anomaly Detection** | Identifies **outliers/unusual data** without prior examples.            |

---

##  Challenges

| Key Concept              | Description / Detail                                                                 |
|---------------------------|-------------------------------------------------------------------------------------|
| **Noisy Data & Outliers** | Outliers/noise can distort results and reduce effectiveness.                        |
| **Assumption Dependence** | Algorithms rely on assumptions (e.g., **cluster shape**) that may not match reality. |
| **Overfitting Risk**      | May capture **noise instead of meaningful patterns**.                               |
| **Limited Guidance**      | Lack of labels makes it hard to **steer results toward desired outcomes**.          |
| **Cluster Interpretability** | Results may lack **clear meaning** or alignment with real-world categories.       |
| **Lack of Ground Truth**  | No labels → difficult to **objectively evaluate accuracy**.                         |

---

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
## 🔹 [Support Vector Machine (SVM) Case Study](https://github.com/MohsenSafari83/Supervised-Learning-/tree/main/classification/Support%20Vector%20Machines%20) 

- ## Goal:**
A comprehensive guide to implementing **Support Vector Machine (SVM)**, emphasizing **data preparation**, **feature scaling**, and **hyperparameter analysis**.  
This project uses the **Iris dataset** to clearly demonstrate the mechanism of the maximum-margin classifier.

- ## Topics Covered**

  - **Core Concepts:** Intuitive explanation of the **Hyperplane**, **Margin**, and **Support Vectors**.
  - **Parameter `C` Analysis:** Deep dive into how the **Regularization Parameter `C`** controls the essential trade-off between maximizing the margin width and minimizing classification errors (**Soft vs. Hard Margin classification**).
  - **Visualization:** Using **Principal Component Analysis (PCA)** to reduce the 4-dimensional data to 2D for plotting and visualizing the final **Decision Boundary**.

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
| ├── SVM case study
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

