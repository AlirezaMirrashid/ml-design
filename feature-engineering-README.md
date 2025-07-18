
# Practical Feature Engineering Demo

This repository provides a hands-on Python implementation of the fundamental feature engineering techniques discussed in Chapter 5 of "Designing Machine Learning Systems" by Chip Huyen.

The goal is to demonstrate how to correctly apply common preprocessing steps to a raw dataset, with a special focus on preventing data leakage—a critical concept for building reliable production models.

---

## Techniques Demonstrated

The `feature_engineering_demo.py` script walks through the following key operations using a sample dataset.

### 1. **Handling Missing Values (Imputation)**
   - **Numerical Imputation:** Missing numerical values (e.g., `Annual_Income`) are filled using the **median** of the column. The median is a robust choice as it is less sensitive to outliers than the mean.
   - **Categorical Imputation:** Missing categorical values (e.g., `Marital_Status`) are filled using the **mode** (most frequent value) of the column.

### 2. **Feature Scaling (Standardization)**
   - Numerical features (`Age`, `Annual_Income`) are rescaled using **Standardization** (Z-score). This transforms the data to have a mean of 0 and a standard deviation of 1, which is a prerequisite for many ML algorithms like Logistic Regression and SVMs.

### 3. **Encoding Categorical Features (One-Hot Encoding)**
   - Categorical features (`Gender`, `Marital_Status`) are converted into a numerical format using **One-Hot Encoding**. This creates a new binary column for each category, preventing the model from assuming a false ordinal relationship between them.

### 4. **Feature Crossing**
   - A new interaction feature is created by combining `Marital_Status` and `Num_Children`. This is a manual way to help the model capture non-linear relationships that might exist between these two features.

---

## The Correct Workflow: Preventing Data Leakage

The most critical concept demonstrated in this script is **how to avoid data leakage** during feature engineering. Leakage occurs when information from the test set inadvertently influences the training process, leading to an overly optimistic performance evaluation and a model that fails in production.

The script implements the correct, leak-proof workflow:

1.  **Split Data First:** The raw data is immediately split into a training set and a testing set. **No feature engineering is done before this step.**

2.  **Fit on Training Data Only:** All preprocessing "transformers" (the imputer, scaler, and encoder) are fitted using **only the training data**. This means any statistics calculated (like the median for imputation or the mean/std for scaling) are derived exclusively from the training set.

3.  **Transform Both Sets:** The transformers fitted on the training data are then used to transform **both the training set and the test set**. This simulates a real-world scenario where the model must process new, unseen data (the test set) using only the knowledge it gained from the training data.

---

## How to Run the Script

### Prerequisites

You need to have Python and the following libraries installed:
- pandas
- scikit-learn

You can install them using pip:
```bash
pip install pandas scikit-learn
```

### Execution

Simply run the Python script from your terminal:
```bash
python feature_engineering_demo.py
```
The script will print the state of the DataFrame at each stage of the transformation process, clearly showing the "before" and "after" for both the training and testing sets.
