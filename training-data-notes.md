
# Notes on Training Data in Machine Learning Systems

This document covers the essential concepts and methodologies for creating and managing high-quality training data for machine learning applications. The process is detailed from a data science perspective, emphasizing practical challenges and solutions for production environments.

*Reference: Huyen, C. (2022). Chapter 4, Training Data. In Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications. O'Reilly Media.*

---

## 1. The Role of Data and Potential Biases

The foundation of any ML model is its training data. The quality of this data dictates the performance and reliability of the final model. It's crucial to understand that production data is not a static "dataset" but a dynamic and evolving entity.

**A Word of Caution on Bias:** Data is susceptible to biases from various sources, including collection methods, sampling strategies, and labeling processes. Historical data often contains latent human biases, which models can learn and amplify. It is imperative to approach data with a critical mindset.

---

## 2. Data Sampling Techniques

Sampling is a fundamental step in the ML workflow, used to select data for training, validation, and monitoring.

### 2.1. Nonprobability Sampling

These methods do not use random selection and can introduce significant bias, but are often used for their convenience.

-   **Convenience Sampling:** Selecting the most easily accessible data.
-   **Snowball Sampling:** Using initial data points to find other, related ones.
-   **Judgment Sampling:** Allowing domain experts to select the data.
-   **Quota Sampling:** Selecting a fixed number of samples from predefined groups without randomization.

### 2.2. Probability-Based Sampling

These methods provide a more statistically sound basis for creating training sets.

-   **Simple Random Sampling:** Every sample has an equal probability of being chosen. Prone to missing rare classes.
-   **Stratified Sampling:** The population is divided into strata (groups), and random sampling is conducted within each. This ensures representation from all groups, including rare ones.
-   **Weighted Sampling:** Each sample is assigned a specific weight that determines its selection probability. Useful for emphasizing more important data (e.g., recent data) or correcting for distributional imbalances.
-   **Reservoir Sampling:** An algorithm for drawing a random sample of fixed size `k` from a data stream of unknown length, ensuring each item has an equal chance of being selected.
-   **Importance Sampling:** A technique to estimate properties of a distribution by sampling from a different, easier-to-sample distribution. Samples are weighted to correct for the difference between distributions.

---

## 3. Data Labeling Strategies

Most production ML models are supervised, making data labeling a critical, and often challenging, step.

### 3.1. Hand Labeling

Manual annotation of data. It is often expensive, slow, and can pose privacy risks.

-   **Challenges:**
    -   **Label Multiplicity:** Disagreements between annotators. Requires clear guidelines and a consensus strategy.
    -   **Data Lineage:** The practice of tracking the origin and processing history of data and labels. Essential for debugging and identifying sources of bias.

### 3.2. Natural Labels

Labels that are generated automatically as a byproduct of user interaction or system behavior.

-   **Examples:** Ad clicks, product purchases, user ratings, trip completion times.
-   **Feedback Loop:** The time delay between a model's prediction and the availability of its ground-truth label. Loops can be short (minutes) or long (months), impacting how quickly a model can be updated.

---

## 4. Techniques for Handling a Lack of Labels

When manual labeling is not feasible, these methods can be used to generate or leverage existing labels.

| Method             | How it Works                                                                                             | Ground Truth Required?                                                       |
| :----------------- | :------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **Weak Supervision**   | Uses heuristics (labeling functions) to programmatically generate noisy labels.                          | No, but a small set of hand labels is recommended to develop the heuristics. |
| **Semi-Supervision** | Leverages structural assumptions about the data to propagate labels from a small seed set to unlabeled data. | Yes, a small number of initial labels are needed.                            |
| **Transfer Learning**  | Reuses a model pretrained on a different task as a starting point for a new task.                        | No for zero-shot learning. Yes for fine-tuning, but often far fewer labels.  |
| **Active Learning**    | A model intelligently queries a human for labels on the most informative unlabeled samples.               | Yes, for the samples selected by the model.                                  |

---

## 5. Managing Class Imbalance

Class imbalance, where classes are not represented equally, is common in real-world datasets and poses several challenges for model training.

### 5.1. Challenges of Imbalance

1.  **Insufficient Signal:** The model may not see enough minority class examples to learn effectively.
2.  **Bias Towards Majority Class:** Models can achieve high accuracy by simply predicting the majority class, ignoring the minority class entirely.
3.  **Asymmetric Costs of Error:** The consequence of misclassifying a minority sample (e.g., a fraudulent transaction) is often far greater.

### 5.2. Solutions for Imbalance

1.  **Use Appropriate Evaluation Metrics:**
    -   **Precision, Recall, and F1-Score:** Focus on the performance of a specific class.
    -   **ROC Curve & AUC:** Evaluates model performance across all classification thresholds.
    -   **Precision-Recall (PR) Curve:** More informative than ROC for highly imbalanced datasets.

2.  **Data-Level Methods (Resampling):**
    -   **Undersampling:** Removing samples from the majority class.
    -   **Oversampling:** Adding more samples to the minority class. A popular technique is **SMOTE** (Synthetic Minority Over-sampling Technique), which creates new synthetic samples rather than just duplicating existing ones.

3.  **Algorithm-Level Methods:**
    -   **Cost-Sensitive Learning:** Assigns a higher misclassification cost to minority class samples.
    -   **Class-Balanced Loss:** Weights the loss function to give more importance to the minority class.
    -   **Focal Loss:** A modified loss function that focuses training on "hard" examples that the model misclassifies.

---

## 6. Data Augmentation

A collection of techniques used to increase the size and diversity of the training set by creating modified or synthetic data.

-   **Simple Label-Preserving Transformations:** Applying modifications that do not change the data's label.
    -   **For Images:** Cropping, rotating, flipping, color shifts.
    -   **For NLP:** Back-translation, synonym replacement.
-   **Perturbation (Adversarial Augmentation):** Adding small amounts of noise to data to make the model more robust against minor variations and adversarial attacks.
-   **Data Synthesis:** Creating new data points from scratch.
    -   **For NLP:** Using templates to generate new sentences.
    -   **For Images:** Using techniques like **mixup** (linearly combining two images and their labels) or **Generative Adversarial Networks (GANs)** to create new, realistic images.

