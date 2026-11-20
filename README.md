# Evaluating the Robustness and Fairness of Diabetes Risk Prediction Models Across Demographic Groups

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-Audited-brightgreen.svg)](https://fairlearn.org/)

---

## 1. Project Overview & Research Question

**Primary Research Question:**  
*Do diabetes prediction models perform differently across age, sex, or other demographic groups?*

Machine learning risk scores are increasingly deployed in preventive population healthcare to prioritize screening and lifestyle interventions. However, clinical models trained on imbalanced survey data often exhibit hidden disparities across protected demographic groups or brittle performance under real-world data corruption.

This research project conducts an exhaustive, scientifically rigorous investigation into:
1. **Overall Predictive Performance:** Comparing baseline linear classifiers, tree ensembles, margin-based models, and gradient boosting.
2. **Demographic Fairness:** Auditing model behavior across biological sex (Female vs. Male) and epidemiological age cohorts (18–34, 35–49, 50–64, and 65+ years) using Fairlearn metrics and bootstrap confidence intervals.
3. **Data-Quality Robustness:** Stress-testing models under controlled missingness (MCAR at 5%, 10%, 20%), input feature corruption (bit-flips, ordinal shifts, numerical noise), and clinical subpopulation covariate shifts.
4. **Clinical Trade-offs:** Analyzing the critical cost asymmetry between False Negatives (missed diagnoses leading to progressive microvascular/macrovascular damage) and False Positives (confirmatory lab testing).

---

## 2. Dataset Architecture & CDC BRFSS Codebook

The study utilizes the **CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)** Diabetes Health Indicators dataset ($N = 253,680$ observations, 21 input features).

### Target Interpretation
In compliance with clinical early-detection screening objectives:
- **Class 0 (Negative):** No diabetes ($N = 213,703$, $84.24\%$)
- **Class 1 (Positive):** Prediabetes or Diabetes risk ($N = 39,977$, $15.76\%$)

### Feature Taxonomy
- **Demographics:**
  - `Sex`: 0 = Female ($N = 141,974$, prevalence $14.80\%$), 1 = Male ($N = 111,706$, prevalence $16.97\%$).
  - `Age`: 13 discrete CDC ordinal tiers mapped into 4 standardized epidemiological cohorts:
    - `18–34 years`: Tiers 1–3 ($N = 24,421$, prevalence $2.84\%$)
    - `35–49 years`: Tiers 4–6 ($N = 49,799$, prevalence $8.28\%$)
    - `50–64 years`: Tiers 7–9 ($N = 90,390$, prevalence $17.51\%$)
    - `65+ years`: Tiers 10–13 ($N = 89,070$, prevalence $21.75\%$)
- **Binary Clinical & Behavioral Indicators (13 features):** `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`.
- **Ordinal Indicators (3 features):** `GenHlth` (1=Excellent to 5=Poor), `Education` (1–6), `Income` (1–8).
- **Numerical Features (3 features):** `BMI` (Body Mass Index), `MentHlth` (days poor mental health/month), `PhysHlth` (days physical illness/month).

---

## 3. Directory Layout

```text
Diabetes/
│
├── data/
│   ├── raw/                             # Raw CDC BRFSS CSV files
│   └── processed/                       # 80:20 stratified train/test partitions
│
├── notebooks/                           # Executable Jupyter Research Notebooks
│   ├── 01_dataset_exploration.ipynb     # Schema validation, profiling & EDA
│   ├── 02_preprocessing.ipynb           # Pipeline construction & leakage checks
│   ├── 03_model_training.ipynb          # Model training, CV, test evaluation
│   ├── 04_fairness_evaluation.ipynb     # Age & Sex subgroup equity analysis
│   └── 05_robustness_evaluation.ipynb   # Missingness, noise, and shift stress tests
│
├── src/                                 # Modular Python Research Source Code
│   ├── data_loader.py                   # Data ingestion, validation & profiling
│   ├── preprocessing.py                 # Leakage-free ColumnTransformer pipeline
│   ├── train_models.py                  # Model development & 5-fold CV
│   ├── evaluate_models.py               # Test evaluation & curve generation
│   ├── fairness_analysis.py             # Subgroup audits & Fairlearn disparities
│   ├── robustness_analysis.py           # Missingness, noise, and shift experiments
│   ├── statistical_analysis.py          # Paired t-tests, Wilcoxon & McNemar tests
│   ├── combined_comparison.py           # 3D comparative synthesis matrix
│   ├── create_notebooks.py              # Programmatic notebook builder
│   └── generate_reports.py              # PDF research report compiler
│
├── models/                              # Serialized trained model artifacts
│   ├── preprocessor.joblib              # Fitted ColumnTransformer pipeline
│   ├── logistic_regression/model.joblib # Baseline L2-regularized logistic model
│   ├── random_forest/model.joblib       # 100-tree balanced ensemble
│   ├── svm/model.joblib                 # Calibrated linear support vector classifier
│   └── xgboost/model.joblib             # Gradient boosted decision trees
│
├── results/                             # Quantitative CSV outputs & configurations
│   ├── dataset_profile.csv              # Overall dataset distribution metrics
│   ├── feature_metadata.csv             # Feature data types and unique counts
│   ├── cv_scores.csv                    # Folds 1-5 CV metrics across all models
│   ├── cv_summary.csv                   # CV Mean and Standard Deviation table
│   ├── overall_performance.csv          # Held-out test performance (9 metrics)
│   ├── age_fairness.csv                 # Subgroup metrics across 4 age cohorts
│   ├── sex_fairness.csv                 # Subgroup metrics across biological sexes
│   ├── fairness_metrics.csv             # DPD, EOD, EODD, and FNRD summary
│   ├── robustness_results.csv           # Missingness & noise stress tests
│   ├── distribution_shift_results.csv   # Metabolic risk covariate shift
│   ├── feature_divergence_ks.csv        # Kolmogorov-Smirnov distribution divergence
│   ├── repeated_cv_scores.csv           # Multi-seed repeated cross-validation
│   ├── statistical_results.csv          # Paired t-tests & Wilcoxon tests
│   ├── mcnemar_results.csv              # McNemar test-set discordance tests
│   ├── combined_comparison.csv          # Three-dimensional comparative matrix
│   └── model_configurations.json        # Hyperparameter configurations
│
├── figures/                             # Publication-Grade Visualizations (300 DPI)
│   ├── dataset_analysis/                # Class, age, sex distributions & heatmap
│   ├── model_performance/               # ROC curves, PR curves, CMs, bar charts
│   ├── fairness/                        # Age/Sex recall, FNR, EOD, EODD charts
│   └── robustness/                      # Degradation slopes, radar comparisons
│
├── reports/                             # Formal Academic PDF Research Reports
│   ├── preprocessing_report.pdf         # Data audit & leakage prevention report
│   ├── model_comparison_report.pdf      # Algorithmic performance & clinical utility
│   ├── fairness_report.pdf              # Demographic equity & subgroup disparities
│   └── final_research_report.pdf        # Thesis-grade comprehensive publication
│
├── dashboard/
│   └── app.py                           # 5-page interactive Streamlit dashboard
│
├── tests/                               # Pytest Unit Test Suite
│   ├── test_preprocessing.py            # Preprocessing & leakage tests
│   ├── test_metrics.py                  # Metric & bootstrap CI unit tests
│   └── test_robustness.py               # Perturbation mask & bound tests
│
├── requirements.txt                     # Pinned Python package dependencies
├── README.md                            # Comprehensive study documentation
└── main.py                              # Master end-to-end command-line runner
```

---

## 4. Installation & Environment Setup

Ensure Python 3.10+ (tested on Python 3.13) is installed.

```bash
# Clone or navigate to the project directory
cd "d:/Sem 7/Diabetes"

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Execution Guide

### Run Full End-to-End Research Pipeline
Executes data ingestion, profiling, preprocessing, model training, evaluation, fairness audits, robustness stress tests, statistical testing, and report generation:

```bash
python main.py
```

### Launch Interactive Streamlit Research Dashboard
Opens the professional 5-page results dashboard in your browser:

```bash
streamlit run dashboard/app.py
# Or via main CLI
python main.py --dashboard
```

### Run Unit Tests
Execute the automated test suite with pytest:

```bash
pytest tests/ -v
```

### Recompile PDF Research Reports
```bash
python main.py --reports-only
```

---

## 6. Empirical Research Results

### A. Overall Predictive Performance (Held-out Test Set, $N = 50,736$)

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 73.03% | 0.3407 | 76.05% | 72.47% | 0.4706 | 0.8176 | 23.95% |
| **Random Forest** | 72.84% | 0.3393 | 76.41% | 72.17% | 0.4699 | 0.8220 | 23.59% |
| **Support Vector Machine (Linear)** | **84.81%** | **0.5525** | 19.02% | **97.12%** | 0.2830 | 0.8174 | **80.98%** |
| **XGBoost Classifier** | 72.12% | 0.3356 | **78.50%** | 70.92% | **0.4701** | **0.8247** | **21.50%** |

*All values reflect true held-out test set performance.*

#### Critical Clinical Takeaway on Accuracy vs. False Negatives:
- **The Accuracy Trap:** Linear SVM achieved the highest raw accuracy ($84.81\%$), yet missed **$80.98\%$ of true positive patients** (False Negative Rate) at the default decision threshold. In clinical screening, missing 4 out of 5 patients with prediabetes or diabetes is disastrous.
- **Superior Clinical Utility of XGBoost:** By incorporating class-frequency gradient reweighting (`scale_pos_weight = 5.35`), XGBoost achieved the lowest False Negative Rate ($21.50\%$) and highest ROC-AUC ($0.8247$), making it the most clinically effective screening tool.

---

### B. Demographic Fairness Evaluation

#### 1. Biological Sex Parity (Female vs. Male)
| Model | Female Recall | Male Recall | Equal Opportunity Diff (EOD) | Demographic Parity Diff (DPD) | Disparity Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 75.96% | 76.15% | **0.0019** (0.19%) | 0.0649 | Equal Parity |
| **Random Forest** | 77.58% | 75.06% | **0.0251** (2.51%) | 0.0383 | Higher Recall for Female |
| **Support Vector Machine** | 19.53% | 18.36% | **0.0117** (1.17%) | 0.0072 | Higher Recall for Female |
| **XGBoost Classifier** | 78.63% | 78.33% | **0.0030** (0.30%) | 0.0641 | Equal Parity |

*Finding:* All models exhibit near-perfect Equal Opportunity parity across biological sexes ($EOD < 0.025$).

#### 2. Age-Based Disparities (18–34 vs. 65+ Years)
| Model | 18–34 FNR (Missed) | 35–49 FNR | 50–64 FNR | 65+ FNR | Equal Opportunity Diff (EOD) | Direction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **67.16%** | 39.81% | 23.44% | **15.86%** | **0.5130** | Severe Underdiagnosis of Young |
| **Random Forest** | **77.41%** | 45.47% | 22.09% | **17.33%** | **0.6008** | Severe Underdiagnosis of Young |
| **Support Vector Machine** | **97.09%** | 90.79% | 83.17% | **77.30%** | **0.1981** | Catastrophic Misses in All Ages |
| **XGBoost Classifier** | **78.84%** | 44.86% | 19.78% | **14.28%** | **0.6450** | Severe Underdiagnosis of Young |

*Finding:* Models suffer from acute **age-based diagnostic bias**. Younger adults (18–34) experience False Negative Rates exceeding $67\%$ to $79\%$, because age is heavily weighted in the risk function and baseline diabetes prevalence is lower in younger cohorts.

---

### C. Robustness Stress-Testing

#### 1. Missing Data Robustness (MCAR at 20% Missingness)
- Input missingness was handled using the training-fitted median/mode imputer pipeline.
- **Most Robust:** Logistic Regression (ROC-AUC drop = **$0.0264$**, $3.23\%$ relative drop).
- Linear SVM drop: $0.0267$.
- XGBoost drop: $0.0273$.
- Random Forest drop: $0.0280$.

#### 2. Input Noise Robustness (20% Feature Corruption)
- Evaluated under 20% binary bit-flips, adjacent ordinal steps, and numerical noise.
- **Most Robust:** Random Forest (ROC-AUC drop = **$0.0404$**, $4.91\%$ relative drop).
- Linear SVM drop: $0.0419$.
- Logistic Regression drop: $0.0436$.
- XGBoost drop: $0.0469$.
- *Insight:* Random Forest's bootstrap aggregating (bagging) and random subspace feature selection provide superior resistance against localized corrupted features.

#### 3. Subgroup Covariate Shift
- Testing on natural high-metabolic risk cohorts (Obese BMI $\ge 30$, Hypertension $= 1$) vs. baseline test population produced an expected drop in ROC-AUC of $\sim 0.106$ to $0.111$, with XGBoost maintaining the highest shifted AUC ($0.7188$).

---

### D. Statistical Validation Summary

- **Paired $t$-tests on Fold ROC-AUC:**
  - XGBoost vs. Logistic Regression: Mean Difference $= +0.0074$ ($t = 30.66, p = 2.05 \times 10^{-10}$, Statistically Significant).
  - XGBoost vs. Random Forest: Mean Difference $= +0.0029$ ($t = 18.31, p = 1.97 \times 10^{-8}$, Statistically Significant).
  - Random Forest vs. Logistic Regression: Mean Difference $= +0.0045$ ($t = 14.94, p = 1.17 \times 10^{-7}$, Statistically Significant).
- **McNemar's Discordance Test:**
  - Statistically significant error divergence confirmed across all model pairs on the held-out test partition ($p < 10^{-13}$).

---

## 7. Answers to Final Research Deliverable Questions

1. **Which model performed best overall?**  
   **XGBoost Classifier** achieved the strongest overall discriminative performance with a test ROC-AUC of **0.8247**, Recall of **78.50%**, and the lowest False Negative Rate (**21.50%**), confirmed by rigorous paired statistical testing ($p < 10^{-8}$).
2. **Which model performed best for each demographic group?**  
   - For older adults (65+): **XGBoost** achieved the highest Recall (**85.72%**) and lowest FNR (**14.28%**).  
   - For younger adults (18–34): **Logistic Regression** achieved the highest sensitivity (**32.84%**), outperforming tree models that overfit to age thresholds.  
   - For biological sexes: **XGBoost** achieved identical superior sensitivity across both females (**78.63%**) and males (**78.33%**).
3. **Which demographic groups experienced higher false-negative rates?**  
   **Younger adults aged 18–34 years** suffered vastly higher False Negative Rates ($67.2\%$ to $78.8\%$ in clinical models, and $97.1\%$ in SVM), compared to only $14.3\%$ in seniors aged 65+. Biological sexes exhibited no significant disparity.
4. **Which model demonstrated the smallest fairness disparities?**  
   Among clinically viable models with reasonable sensitivity, **Logistic Regression** had the smallest age disparity ($EOD = 0.5130$, compared to $0.6450$ for XGBoost). Linear SVM had lower raw disparity ($0.1981$) but failed entirely due to universal underdiagnosis ($FNR = 80.98\%$).
5. **Which model maintained performance best under missing and noisy data?**  
   **Logistic Regression** retained performance best under missing data (AUC drop = $0.0264$ at 20%), while **Random Forest** demonstrated the greatest resilience to input feature noise (AUC drop = $0.0404$ at 20%).
6. **Is the highest-accuracy model also the fairest and most robust?**  
   **No.** Linear SVM achieved the highest test accuracy ($84.81\%$), but proved to be the most clinically dangerous model, missing over $80\%$ of diabetic patients. Optimizing accuracy on imbalanced clinical data produces models that predict the majority negative class while ignoring disease risk.
7. **What limitations should be acknowledged in the research paper?**  
   - **Self-Reported Telephonic Survey:** BRFSS indicators rely on self-reported survey answers rather than standardized clinical assays (HbA1c, OGTT, fasting glucose).
   - **Cross-Sectional Structure:** The dataset lacks longitudinal follow-ups; distribution shift was evaluated via subpopulation covariate divergence rather than temporal cohort validation.
   - **Fixed Threshold Constraint:** All models were evaluated at a uniform decision threshold ($p = 0.5$); age-stratified decision boundary optimization represents an essential future direction to mitigate young-adult underdiagnosis.

---

## 8. License

This research codebase is distributed under the MIT License. Academic citations should reference the CDC BRFSS 2015 study and this repository.

