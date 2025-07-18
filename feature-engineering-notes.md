
# Notes on Feature Engineering

This document outlines the core principles, techniques, and challenges of feature engineering in production machine learning systems. Creating effective features is often the most critical factor in a model's success, frequently providing a greater performance boost than algorithmic tuning alone.

*Reference: Huyen, C. (2022). Chapter 5, Feature Engineering. In Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications. O'Reilly Media.*

---

## 1. Learned vs. Engineered Features

-   **Engineered Features:** Manually crafted features based on domain knowledge. For text, this traditionally involved steps like lemmatization, stopword removal, and creating n-grams.
-   **Learned Features:** Features automatically extracted by deep learning models. Deep learning has automated much of the feature engineering for raw data like text and images.

However, most ML systems require features beyond what can be learned from raw inputs. For example, a spam detection model needs more than just the comment text; it needs features about the user (account age, post frequency), the thread (popularity), and the comment itself (number of upvotes). This process of selecting, extracting, and transforming this information is **feature engineering**.

---

## 2. Common Feature Engineering Operations

### 2.1. Handling Missing Values

Data in production is rarely complete. Understanding *why* data is missing is crucial for handling it correctly.

-   **Types of Missing Values:**
    -   **Missing Not at Random (MNAR):** The reason for the missing value is related to the value itself (e.g., high-income individuals not disclosing their income).
    -   **Missing at Random (MAR):** The missingness is related to another observed variable (e.g., people of a certain gender not disclosing their age).
    -   **Missing Completely at Random (MCAR):** There is no pattern to the missing data. This is rare.

-   **Handling Strategies:**
    -   **Deletion:**
        -   **Column Deletion:** Removing a feature if it has too many missing values. Risks losing important information.
        -   **Row Deletion:** Removing a sample if it has missing values. Only safe for MCAR data and a very small number of affected rows.
    -   **Imputation (Filling Values):**
        -   **Common Methods:** Filling with a default (e.g., empty string), mean, median, or mode.
        -   **Caution:** Avoid imputing with a value that has a real-world meaning (e.g., filling missing `age` with `0`), as it can confuse the model. Imputation can introduce bias and noise.

### 2.2. Feature Scaling

Scaling ensures that numerical features are within a similar range, preventing models from assigning undue importance to features with larger absolute values.

-   **Min-Max Scaling:** Rescales features to a fixed range, typically `[0, 1]` or `[-1, 1]`.
-   **Standardization (Z-score Normalization):** Rescales features to have a mean of 0 and a standard deviation of 1. Assumes data follows a normal distribution.
-   **Log Transformation:** Useful for mitigating the effects of highly skewed distributions.

### 2.3. Discretization (Binning)

The process of converting continuous features into discrete categorical features by grouping values into "buckets." This can help models learn from data with limited granularity but introduces sharp, artificial boundaries.

### 2.4. Encoding Categorical Features

Categorical features, especially those with a large or changing number of unique values (high cardinality), pose a challenge in production.

-   **The Hashing Trick:** Uses a hash function to map a potentially infinite number of categories to a fixed-size vector.
    -   **Benefit:** Handles new, unseen categories gracefully without needing an `UNKNOWN` token.
    -   **Drawback:** **Hash collisions** (different categories mapping to the same hash). However, research shows that even with a high collision rate, the impact on performance is often minimal.

### 2.5. Feature Crossing

Combining two or more features to create a new, interaction feature. This is crucial for helping linear models and tree-based models capture non-linear relationships. For example, crossing `marital_status` and `number_of_children`.

### 2.6. Positional Embeddings

Used to encode the order of items in a sequence, which is critical for models like Transformers that process data in parallel.

-   **Learned Embeddings:** An embedding matrix is created for positions, and the embeddings are learned during training (similar to word embeddings).
-   **Fixed Embeddings (Fourier Features):** Uses periodic functions (sine and cosine) to create a unique, fixed vector for each position. This can be extended to continuous coordinates.

---

## 3. Data Leakage

Data leakage occurs when information from outside the training data (often, information related to the label) "leaks" into the feature set. This leads to models that perform exceptionally well in evaluation but fail in production because the leaked signal is not available at inference time.

### 3.1. Common Causes of Data Leakage

1.  **Splitting Time-Correlated Data Randomly:** For time-series data (e.g., stock prices, user activity), data must be split by time (e.g., train on weeks 1-4, test on week 5). A random split would leak future information into the training set.
2.  **Scaling or Imputing Before Splitting:** Using statistics (mean, variance, etc.) calculated from the *entire dataset* (including test data) to transform the training data. **Always split your data first**, then calculate statistics for scaling/imputation from the **training split only** and apply them to all splits.
3.  **Data Duplication:** Having identical or near-identical samples in both the train and test splits.
4.  **Group Leakage:** When data points are not independent but are split across train/test sets (e.g., different medical scans from the same patient, or photos of the same object).
5.  **Leakage from the Data Generation Process:** Subtle signals in the data that are correlated with the label but are artifacts of the collection process (e.g., a specific font used by a hospital with more severe cases).

### 3.2. Detecting Data Leakage

-   Monitor the predictive power of individual features. If a feature is unusually predictive, investigate it.
-   Perform ablation studies to measure the impact of removing certain features.
-   Be extremely cautious when inspecting the test set. Any insights gained can lead to biased decisions.

---

## 4. Engineering Good Features

A good feature is one that is both **important** to the model and **generalizes** well to unseen data.

### 4.1. Feature Importance

-   **Methods:** Use built-in functions (e.g., in XGBoost) or model-agnostic tools like **SHAP** (SHapley Additive exPlanations) to measure how much each feature contributes to a model's predictions.
-   **Pareto Principle:** Often, a small number of features account for the majority of a model's performance.

### 4.2. Feature Generalization

-   **Coverage:** The percentage of samples that have a non-missing value for a feature. Low coverage can be a warning sign, unless the missingness itself is a strong signal.
-   **Distribution:** The distribution of a feature's values should be similar between the training and test sets. A mismatch indicates a potential distribution shift or data leakage.

---

## 5. Summary of Best Practices

1.  **Split data correctly:** For time-correlated data, always split by time.
2.  **Process after splitting:** Perform scaling, imputation, and oversampling *after* splitting the data, using statistics from the training set only.
3.  **Understand data generation:** Know how your data is collected and processed. Involve domain experts.
4.  **Track data lineage:** Maintain a record of where your data comes from.
5.  **Evaluate feature importance and generalization:** Ensure features are both predictive and likely to perform well on unseen data.
6.  **Prune useless features:** Regularly remove features that no longer contribute to the model to reduce technical debt and prevent overfitting.
