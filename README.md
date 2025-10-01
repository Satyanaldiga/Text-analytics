

# INFO 617 – Mental Health QA & Question Category Prediction

This repository contains code and analysis for **two main tasks (Q1 and Q2)** based on the datasets:

* `INFO 617 Mental Health QA_LIWC.csv`
* `INFO 617_QA_Question_Category.csv`

The tasks involve text classification using CNNs and LSTMs, correlation analysis with LIWC features, Gradient Boosting with feature importance and hyperparameter tuning, and TF-IDF based classification.

---

## 📌 Question 1: Predicting “Bonus”

### Dataset Preparation

* From `INFO 617 Mental Health QA_LIWC.csv`.
* Created a binary variable `Bonus`:

  * `Bonus = 1` if `Received_Bonus_Yes_No >= 1`
  * `Bonus = 0` otherwise

### Part (a): Correlation Analysis

* Identified **LIWC variables** (starting with `WC`) with the **highest positive and negative Pearson correlation** with `Bonus`.
* Proposed hypotheses explaining why these linguistic features correlate with bonus likelihood.

### Part (b): Gradient Boosting with LIWC Variables

1. Removed **LIWC variables with zero variance**.
2. Used **GradientBoostingClassifier** to predict `Bonus` with all LIWC features.
3. Extracted **feature importance scores** and identified **top 5 predictors**.
4. Explored **all possible subsets of these top 5 predictors** to test if smaller feature sets can outperform the full model (default GB settings).

### Part (c): Hyperparameter Tuning

* Tuned the following Gradient Boosting parameters using the validation set:

  * **Learning Rate**
  * **Number of Estimators**
  * **Max Depth**
* Evaluated whether tuning improved **accuracy** on the test set.

### CNN for Text Classification

* Built a **Convolutional Neural Network (CNN)** to predict `Bonus` using `Answer_English`.
* Addressed **class imbalance** with resampling techniques before model training.
* Split dataset into **train, validation, test sets**.
* Reported results on test set using:

  * **Accuracy**
  * **Precision**
  * **Recall**
  * **F1-score**

### LSTM for Text Classification

* Built two LSTM models:

  1. **Unidirectional LSTM**
  2. **Bidirectional LSTM**
* Used the same resampled dataset as in CNN.
* Compared both models on the test set using the four metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## 📌 Question 2: Question Category Prediction

### Dataset: `INFO 617_QA_Question_Category.csv`

### Part (a): Token Count Feature

* Created a new feature `Answer_Token_Count` by:

  * Tokenizing `Question_English` text
  * Counting the number of words

### Part (b): TF-IDF + Classification

* Preprocessed the dataset:

  * Removed unnecessary columns and duplicates
  * Handled missing values
  * Encoded categorical variables
  * Removed rare categories:

    * “Sexuality and sex”
    * “Social incidents and cultural issues”
    * “Physical health”
* Computed **TF-IDF vectors** of questions.
* Built a classification model to predict `Cat1` from TF-IDF features.
* Evaluated model performance (accuracy and classification metrics).

---

## ⚙️ Tools & Libraries

* **Python 3.x**
* **pandas, numpy, scikit-learn**
* **tensorflow / keras** (for CNN & LSTM models)
* **NLTK / spaCy** (for tokenization)
* **matplotlib, seaborn** (for visualization)

---

## 📊 Evaluation Metrics

All models are evaluated using the following:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

---

## 📝 Notes

* No formal hyperparameter tuning was required for CNN and LSTM.
* Gradient Boosting tuning was limited to three key parameters.
* Oversampling or undersampling was applied where class imbalance existed.

---

