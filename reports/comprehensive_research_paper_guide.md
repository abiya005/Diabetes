# Comprehensive Research Paper Manuscript & Execution Report

**Title:** Evaluating the Robustness and Demographic Fairness of Machine Learning Models for Diabetes Risk Prediction Across Population Subgroups  
**Target Publication Venues:** IEEE Journal of Biomedical and Health Informatics (JBHI) / ACM Conference on Health, Inference, and Learning (CHIL) / Journal of the American Medical Informatics Association (JAMIA)  
**Study Cohort:** Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS 2015)  
**Dataset Scale:** $N = 253,680$ observations, 21 clinical, behavioral, and demographic features  

---

## Abstract

**Background:** Machine learning (ML) models are increasingly integrated into preventive clinical decision-support systems to stratify diabetes risk. However, while predictive accuracy is widely benchmarked, algorithmic behavior across demographic intersections (e.g., age cohorts and biological sex) and model resilience under degraded real-world clinical data remain poorly quantified. 

**Objective:** To systematically evaluate the trade-offs between predictive performance, demographic fairness, and data-quality robustness across four representative classification algorithms in adult diabetes risk prediction.

**Methods:** Using the CDC BRFSS 2015 cohort ($N = 253,680$), we mapped survey responses to a harmonized screening target ($0 = \text{no diabetes}$, $84.24\%$; $1 = \text{prediabetes or diabetes}$, $15.76\%$). An 80:20 stratified split established training ($N = 202,944$) and held-out testing ($N = 50,736$) partitions. Data leakage was strictly prevented via scikit-learn `ColumnTransformer` pipelines. Four algorithms were trained and cross-validated with 5-fold stratified cross-validation: Logistic Regression (L2-regularized baseline), Random Forest (100-tree ensemble), Linear Support Vector Machine (with Platt calibration), and XGBoost Classifier (with class-gradient reweighting). Demographic fairness was audited across biological sex (Female vs. Male) and four epidemiological age brackets (18–34, 35–49, 50–64, and 65+ years) using Fairlearn metrics and 95% bootstrap percentile confidence intervals. Robustness was stress-tested across Missing Completely at Random (MCAR; 5%–20%), input feature corruption (bit-flips, ordinal shifts, numerical jitter; 5%–20%), and clinical subpopulation covariate shifts. Statistical significance was established via paired $t$-tests, Wilcoxon signed-rank tests, and McNemar discordance tests.

**Results:** XGBoost achieved the highest overall discriminative performance on the test set ($\text{ROC-AUC} = 0.8247$, $\text{Recall} = 78.50\%$, $\text{False Negative Rate [FNR]} = 21.50\%$), significantly outperforming Logistic Regression ($t = 30.66, p = 2.05 \times 10^{-10}$) and Random Forest ($t = 18.31, p = 1.97 \times 10^{-8}$). While all models demonstrated near-perfect sex parity ($\text{Equal Opportunity Difference [EOD]} < 0.025$), a severe age-based equity disparity was uncovered: younger adults aged 18–34 years suffered catastrophic false-negative rates ($67.16\%$ in Logistic Regression, $77.41\%$ in Random Forest, $78.84\%$ in XGBoost, and $97.09\%$ in SVM), compared to an FNR of only $14.28\%$ to $17.33\%$ in adults aged 65+ ($\text{EOD} = 0.6450$ for XGBoost). Under 20% input corruption, Random Forest proved most resilient ($\text{ROC-AUC drop} = 0.0404$), whereas Linear SVM exhibited the "Accuracy Paradox"—achieving the highest raw accuracy ($84.81\%$) and specificity ($97.12\%$) by defaulting to majority-negative predictions, missing $80.98\%$ of true diabetic patients.

**Conclusion:** Predictive accuracy is a deceptive metric in clinical screening. While gradient boosting provides superior sensitivity, uniform decision boundaries cause extreme underdiagnosis in younger demographics. Clinical translation requires age-stratified decision thresholds and ensemble-based feature resilience.

**Keywords:** Diabetes Mellitus, Algorithmic Fairness, Equal Opportunity, Model Robustness, Machine Learning, Clinical Decision Support, Health Equity.

---

## 1. Introduction & Epidemiological Context

Diabetes mellitus remains one of the fastest-growing global public health crises, affecting over 537 million adults globally, with prevalence projected to surge past 780 million by 2045. In the United States, data from the Centers for Disease Control and Prevention indicates that over 37 million Americans have diabetes, while an estimated 96 million adults have prediabetes—over 80% of whom are unaware of their condition. Chronic hyperglycemia causes progressive microvascular dysfunction (diabetic retinopathy, nephropathy, and distal peripheral neuropathy) and macrovascular atherosclerosis (coronary artery disease, myocardial infarction, and peripheral arterial disease).

Early detection of glycemic risk enables lifestyle modifications (e.g., CDC National Diabetes Prevention Program) and pharmacotherapy (e.g., metformin, SGLT-2 inhibitors) that can halt or reverse disease progression. Consequently, machine learning (ML) models trained on electronic health records (EHR) and population survey data are increasingly proposed for automated pre-screening.

### The Research Gap: Fairness and Robustness in Clinical AI
Despite promising benchmark accuracies reported in the literature, three critical questions are routinely neglected in translational clinical informatics:
1. **The Cost Asymmetry of Prediction Errors:** In population screening, False Negatives (FN)—missed prediabetic/diabetic patients—carry immense downstream morbidity and cost. False Positives (FP), conversely, merely prompt an inexpensive, non-invasive confirmatory blood assay (HbA1c or fasting plasma glucose). Yet, standard ML workflows often optimize for raw Accuracy, inadvertently driving models toward majority-negative prediction.
2. **Demographic Equity Across Age and Biological Sex:** Health determinants vary drastically across demographic groups. Age is the strongest natural risk factor for Type 2 diabetes. However, if a model over-indexes on age as a predictive feature, does it systematically underdiagnose younger individuals who manifest metabolic syndrome? Does biological sex induce diagnostic disparity?
3. **Robustness to Clinical Noise and Missingness:** Clinical deployments encounter missing survey answers, measurement variability, and data transfer corruptions. How gracefully do different algorithmic paradigms degrade under missing and noisy inputs?

This paper addresses these gaps through an end-to-end empirical investigation.

---

## 2. Research Questions and Hypotheses

This research is guided by four primary research questions:

- **RQ1 (Predictive Superiority):** Does gradient boosting (XGBoost) provide statistically significant improvements in discriminative performance (ROC-AUC) and sensitivity (Recall) over linear baselines (Logistic Regression), margin-based classifiers (SVM), and tree ensembles (Random Forest)?
  - *Hypothesis $H_1$:* Non-linear models capturing feature interactions will outperform linear models in ROC-AUC ($p < 0.05$).
- **RQ2 (Demographic Parity & Equal Opportunity):** Do models exhibit demographic disparities in predictive positive rates (Demographic Parity) or true positive sensitivity (Equal Opportunity) across biological sex and age brackets?
  - *Hypothesis $H_2$:* Models will exhibit higher false-negative rates in younger demographics due to lower prior prevalence and dominant age feature weighting.
- **RQ3 (Input Degradation Resilience):** Which algorithmic architecture exhibits the lowest degradation when inputs are subjected to Controlled Missing Completely at Random (MCAR) and feature-type-specific corruption?
  - *Hypothesis $H_3$:* Random Forest will retain performance best under feature corruption due to its bagging and feature subspace sampling architecture.
- **RQ4 (The Accuracy-Fairness-Robustness Trade-Off):** Does the model with the highest test accuracy represent the fairest and most robust clinical choice?
  - *Hypothesis $H_4$:* High raw accuracy will conceal severe minority-class false-negative rates, illustrating the clinical Accuracy Paradox.

---

## 3. Dataset Architecture & Cohort Selection

### 3.1 Data Source
The study utilizes the **CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)** Diabetes Health Indicators cohort. The BRFSS is an annual, state-based, random-digit-dialed telephonic survey conducted by the CDC, collecting health-related risk behaviors, chronic health conditions, and preventive clinical usage across U.S. adult residents ($N = 253,680$).

### 3.2 Target Variable Formulation & Harmonization
The original CDC survey question `DIABETE3` recorded multi-class responses:
- Code 0: No diabetes ($N = 213,703$, $84.24\%$)
- Code 1: Prediabetes or borderline diabetes ($N = 4,631$, $1.83\%$)
- Code 2: Diagnosed diabetes ($N = 35,346$, $13.93\%$)

For early-stage clinical screening, identifying prediabetic individuals is equally as critical as identifying diabetic patients. Therefore, following epidemiological best practices, the target was harmonized into a binary screening objective:
$$\text{Diabetes\_binary} = \begin{cases} 0, & \text{if } \text{Diabetes\_012} = 0 \text{ (Negative Class; } N = 213,703\text{)} \\ 1, & \text{if } \text{Diabetes\_012} \in \{1, 2\} \text{ (Positive Class; } N = 39,977, 15.76\%\text{)} \end{cases}$$

### 3.3 Feature Taxonomy & Codebook Verification
A systematic inspection confirmed 21 predictor variables:

| Variable Name | Feature Category | Data Type | Encoding / Domain | Clinical / Behavioral Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `HighBP` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Diagnosed high blood pressure by physician |
| `HighChol` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Diagnosed high blood cholesterol |
| `CholCheck` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Cholesterol check within the past 5 years |
| `BMI` | Continuous Metric | Float64 | Range: $12.0 - 98.0$ | Body Mass Index ($\text{kg/m}^2$) |
| `Smoker` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Smoked $\ge 100$ cigarettes in lifetime |
| `Stroke` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | History of stroke |
| `HeartDiseaseorAttack` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | History of coronary heart disease or MI |
| `PhysActivity` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Physical activity in past 30 days (non-work) |
| `Fruits` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Consumes fruit $\ge 1$ time per day |
| `Veggies` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Consumes vegetables $\ge 1$ time per day |
| `HvyAlcoholConsump` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Heavy drinkers: $>14$ drinks/wk (M) or $>7$ (F) |
| `AnyHealthcare` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Has any health insurance coverage |
| `NoDocbcCost` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Past 12 mos needed doctor but couldn't afford |
| `GenHlth` | Ordinal Clinical | Float64 | 1 (Excellent) - 5 (Poor) | Self-rated general health status |
| `MentHlth` | Discrete Count | Float64 | Integer $0 - 30$ | Days of poor mental health in past 30 days |
| `PhysHlth` | Discrete Count | Float64 | Integer $0 - 30$ | Days of physical illness/injury in past 30 days |
| `DiffWalk` | Binary Indicator | Float64 (0/1) | $0 = \text{No}, 1 = \text{Yes}$ | Serious difficulty walking or climbing stairs |
| `Sex` | Sensitive Demographic | Float64 (0/1) | $0 = \text{Female}, 1 = \text{Male}$ | Biological sex |
| `Age` | Sensitive Demographic | Float64 | Ordinal tiers $1 - 13$ | 5-year age intervals (CDC standard) |
| `Education` | Ordinal Sociodemographic | Float64 | Tiers $1 - 6$ | Highest level of education completed |
| `Income` | Ordinal Sociodemographic | Float64 | Tiers $1 - 8$ | Annual household income brackets |

### 3.4 Demographic Categorization for Subgroup Fairness
To conduct meaningful epidemiological audits, the 13 raw CDC age tiers were mapped into 4 standardized life-course cohorts:
- **Younger Adults (18–34 years):** CDC codes 1 (18–24), 2 (25–29), 3 (30–34) $\implies N = 24,421$ ($9.63\%$ of cohort, prevalence $2.84\%$).
- **Early Middle-Aged Adults (35–49 years):** CDC codes 4 (35–39), 5 (40–44), 6 (45–49) $\implies N = 49,799$ ($19.63\%$ of cohort, prevalence $8.28\%$).
- **Late Middle-Aged Adults (50–64 years):** CDC codes 7 (50–54), 8 (55–59), 9 (60–64) $\implies N = 90,390$ ($35.63\%$ of cohort, prevalence $17.51\%$).
- **Older Adults (65+ years):** CDC codes 10 (65–69), 11 (70–74), 12 (75–79), 13 (80+) $\implies N = 89,070$ ($35.11\%$ of cohort, prevalence $21.75\%$).

Biological sex was distributed as:
- **Female ($0$):** $N = 141,974$ ($55.97\%$, empirical prevalence $14.80\%$)
- **Male ($1$):** $N = 111,706$ ($44.03\%$, empirical prevalence $16.97\%$)

---

## 4. Preprocessing & Data Leakage Prevention

### 4.1 Stratified Partitioning
To guarantee strict statistical separation, the dataset ($N = 253,680$) was partitioned into:
- **Training Set (80%):** $N_{train} = 202,944$ observations (31,982 positive cases, $15.76\%$)
- **Held-Out Test Set (20%):** $N_{test} = 50,736$ observations (7,995 positive cases, $15.76\%$)
Stratification was enforced on `Diabetes_binary` using `random_state = 42`. The test set was held strictly blind until final model evaluation.

### 4.2 Leakage-Free Pipeline Architecture
Data transformations were isolated inside a scikit-learn `ColumnTransformer` fitted **strictly on $X_{train}$**:
1. **Numerical Pipeline:** For continuous/count features (`BMI`, `MentHlth`, `PhysHlth`), median imputation (`SimpleImputer(strategy='median')`) was paired with z-score standardization (`StandardScaler()`), computing:
   $$z = \frac{x - \mu_{train}}{\sigma_{train}}$$
2. **Ordinal Pipeline:** For clinical/socioeconomic ranks (`GenHlth`, `Education`, `Income`), median imputation was applied alongside `StandardScaler()` to ensure numerical stability for gradient and margin solvers while preserving ordinal ordering.
3. **Binary Feature Pipeline:** For the 13 clinical flags (`HighBP`, `HighChol`, etc.), most-frequent imputation was applied (`SimpleImputer(strategy='most_frequent')`). Crucially, **no standard scaling was applied**, maintaining exact $0/1$ boolean semantics.
4. **Demographic Pipeline:** Scaled versions of `Sex` and `Age` were passed to model inputs, while original unscaled labels (`Sex_Label`, `Age_Group`) were preserved in metadata containers strictly for downstream demographic auditing.

---

## 5. Machine Learning Model Development & Cross-Validation

Four distinct algorithm paradigms were benchmarked:

### 5.1 Model Specifications
1. **Logistic Regression (Linear Baseline):**
   - Formulation: Log-odds parameterized as $\ln\left(\frac{p}{1-p}\right) = \beta_0 + \mathbf{\beta}^T \mathbf{x}$.
   - Solver: L-BFGS, $L_2$ regularization penalty ($C = 1.0$), `class_weight='balanced'`.
   - Purpose: Establish a transparent, linear baseline decision boundary.
2. **Random Forest (Bagged Tree Ensemble):**
   - Hyperparameters: $n_{\text{estimators}} = 100$, $\text{max\_depth} = 12$, $\text{min\_samples\_split} = 10$, $\text{min\_samples\_leaf} = 4$, `class_weight='balanced'`, `n_jobs=-1`.
   - Purpose: Capture non-linear feature interactions without gradient overfitting.
3. **Support Vector Machine (Margin-Based Classifier):**
   - Base Classifier: `LinearSVC(C=1.0, dual=False, class_weight='balanced')`.
   - Probability Calibration: Wrapped in `CalibratedClassifierCV(method='sigmoid', cv=3)` (Platt scaling) to fit a posterior logistic transformation over decision margins:
     $$P(Y=1|s) = \frac{1}{1 + \exp(As + B)}$$
   - Purpose: Evaluate margin-based maximum-margin separation with calibrated probability output for ROC-AUC computation.
4. **XGBoost (Gradient Boosted Decision Trees):**
   - Hyperparameters: $n_{\text{estimators}} = 150$, $\text{max\_depth} = 5$, $\text{learning\_rate} = 0.08$, $\text{subsample} = 0.8$, $\text{colsample\_bytree} = 0.8$, `eval_metric='logloss'`.
   - Imbalance Handling: Loss gradient scaled via:
     $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{170,962}{31,982} \approx 5.35$$
   - Purpose: Leverage second-order gradient boosting to maximize minority-class sensitivity.

### 5.2 Stratified 5-Fold Cross-Validation Protocol
Within the training partition ($N = 202,944$), stratified 5-fold cross-validation was conducted (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

#### Cross-Validation Results Across 5 Folds (Mean $\pm$ Standard Deviation):
- **Logistic Regression:**
  - $\text{Accuracy} = 0.7287 \pm 0.0011$
  - $\text{Precision} = 0.3393 \pm 0.0017$
  - $\text{Recall} = 0.7613 \pm 0.0076$
  - $\text{Specificity} = 0.7227 \pm 0.0016$
  - $\text{F1-Score} = 0.4694 \pm 0.0029$
  - $\text{ROC-AUC} = 0.8177 \pm 0.0026$
- **Random Forest:**
  - $\text{Accuracy} = 0.7303 \pm 0.0024$
  - $\text{Precision} = 0.3414 \pm 0.0021$
  - $\text{Recall} = 0.7655 \pm 0.0083$
  - $\text{Specificity} = 0.7237 \pm 0.0038$
  - $\text{F1-Score} = 0.4722 \pm 0.0027$
  - $\text{ROC-AUC} = 0.8224 \pm 0.0018$
- **Linear SVM (Calibrated):**
  - $\text{Accuracy} = 0.8479 \pm 0.0009$
  - $\text{Precision} = 0.5525 \pm 0.0088$
  - $\text{Recall} = 0.1831 \pm 0.0047$
  - $\text{Specificity} = 0.9723 \pm 0.0008$
  - $\text{F1-Score} = 0.2750 \pm 0.0060$
  - $\text{ROC-AUC} = 0.8174 \pm 0.0025$
- **XGBoost Classifier:**
  - $\text{Accuracy} = 0.7198 \pm 0.0018$
  - $\text{Precision} = 0.3348 \pm 0.0020$
  - $\text{Recall} = 0.7881 \pm 0.0091$
  - $\text{Specificity} = 0.7071 \pm 0.0029$
  - $\text{F1-Score} = 0.4699 \pm 0.0032$
  - $\text{ROC-AUC} = 0.8252 \pm 0.0024$

---

## 6. Overall Performance Evaluation on Held-Out Test Set

The finalized models were evaluated on the held-out test partition ($N = 50,736$).

### Table 1: Comprehensive Test Set Performance Across 9 Clinical Evaluation Metrics
| Metric | Logistic Regression | Random Forest | Support Vector Machine | XGBoost Classifier |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 73.03% | 72.84% | **84.81%** | 72.12% |
| **Precision (PPV)** | 0.3407 | 0.3393 | **0.5525** | 0.3356 |
| **Recall / Sensitivity (TPR)** | 76.05% | 76.41% | 19.02% | **78.50%** |
| **Specificity (TNR)** | 72.47% | 72.17% | **97.12%** | 70.92% |
| **F1-Score** | **0.4706** | 0.4699 | 0.2830 | 0.4701 |
| **ROC-AUC** | 0.8176 | 0.8220 | 0.8174 | **0.8247** |
| **False Positive Rate (FPR)** | 27.53% | 27.83% | **2.88%** | 29.08% |
| **False Negative Rate (FNR)** | 23.95% | 23.59% | 80.98% | **21.50%** |
| **True Negatives (TN)** | 30,975 | 30,845 | 41,508 | 30,317 |
| **False Positives (FP)** | 11,766 | 11,896 | 1,232 | 12,423 |
| **False Negatives (FN)** | 1,915 | 1,886 | 6,475 | **1,719** |
| **True Positives (TP)** | 6,080 | 6,109 | 1,520 | **6,276** |

### Analysis of the "Accuracy Trap"
The empirical data provides a striking demonstration of the **Accuracy Trap**:
- Linear SVM achieved the highest raw accuracy ($84.81\%$), exceeding XGBoost by $12.69$ percentage points.
- However, SVM's True Positive count was only **1,520**, while missing **6,475 diabetic individuals** (an alarming False Negative Rate of **$80.98\%$**).
- By contrast, XGBoost captured **6,276 true positives** and missed only **1,719**, cutting the false-negative rate to **$21.50\%$**.
- In healthcare screening, SVM's 84.8% accuracy would lead to untreated diabetes in 4 out of 5 patients, while XGBoost correctly refers 78.5% of at-risk patients for confirmatory laboratory testing.

---

## 7. Demographic Fairness Evaluation

### 7.1 Mathematical Definitions of Fairness Metrics
Let sensitive attribute $A \in \{a, b\}$ denote demographic membership, $Y \in \{0, 1\}$ denote ground truth, and $\hat{Y} \in \{0, 1\}$ denote model prediction.
1. **Demographic Parity Difference (DPD):** Measures disparity in positive selection rates regardless of ground truth:
   $$\text{DPD} = |P(\hat{Y}=1 | A=a) - P(\hat{Y}=1 | A=b)|$$
2. **Equal Opportunity Difference (EOD):** Measures disparity in True Positive Rate (Recall / Sensitivity) among individuals who actually have the condition:
   $$\text{EOD} = |P(\hat{Y}=1 | A=a, Y=1) - P(\hat{Y}=1 | A=b, Y=1)| = |\text{Recall}_a - \text{Recall}_b|$$
3. **Equalized Odds Difference (EODD):** Maximum disparity across both True Positive Rate and False Positive Rate:
   $$\text{EODD} = \max\left( |\text{TPR}_a - \text{TPR}_b|, |\text{FPR}_a - \text{FPR}_b| \right)$$
4. **False Negative Rate Difference (FNRD):** Absolute disparity in missed cases between groups:
   $$\text{FNRD} = |(1 - \text{Recall}_a) - (1 - \text{Recall}_b)| = \text{EOD}$$

### 7.2 Subgroup Performance by Biological Sex
Test Set Sample Sizes: Female ($N = 28,395$), Male ($N = 22,341$).

| Model | Subgroup | Sample Size | Actual Prevalence | Pred. Positive Rate | Accuracy | Recall (Sensitivity) | 95% Bootstrap CI | False Negative Rate | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Female | 28,395 | 14.80% | 32.32% | 74.07% | 75.96% | [74.67%, 77.27%] | 24.04% | 0.8252 |
| | Male | 22,341 | 16.97% | 38.81% | 71.72% | 76.15% | [74.79%, 77.53%] | 23.85% | 0.8078 |
| **Random Forest** | Female | 28,395 | 14.80% | 33.72% | 73.94% | 77.58% | [76.32%, 78.86%] | 22.42% | 0.8306 |
| | Male | 22,341 | 16.97% | 37.55% | 71.43% | 75.06% | [73.68%, 76.45%] | 24.94% | 0.8115 |
| **Linear SVM** | Female | 28,395 | 14.80% | 5.07% | 85.95% | 19.53% | [18.33%, 20.73%] | 80.47% | 0.8250 |
| | Male | 22,341 | 16.97% | 5.79% | 83.37% | 18.36% | [17.13%, 19.60%] | 81.64% | 0.8076 |
| **XGBoost** | Female | 28,395 | 14.80% | 34.07% | 73.18% | 78.63% | [77.39%, 79.88%] | 21.37% | 0.8327 |
| | Male | 22,341 | 16.97% | 40.48% | 70.76% | 78.33% | [77.01%, 79.64%] | 21.67% | 0.8149 |

### 7.3 Subgroup Performance by Age Bracket
Test Set Sample Sizes: 18–34 ($N = 4,884$), 35–49 ($N = 9,960$), 50–64 ($N = 18,078$), 65+ ($N = 17,814$).

| Model | Age Group | Sample Size | Actual Prevalence | Pred. Positive Rate | Accuracy | Recall (Sensitivity) | 95% Bootstrap CI | False Negative Rate | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 18–34 | 4,884 | 2.85% | 7.43% | 93.63% | **32.84%** | [25.18%, 40.88%] | **67.16%** | 0.8285 |
| | 35–49 | 9,960 | 8.28% | 19.24% | 83.84% | **60.19%** | [56.84%, 63.53%] | **39.81%** | 0.8091 |
| | 50–64 | 18,078 | 17.51% | 36.32% | 72.88% | **76.56%** | [75.07%, 78.04%] | **23.44%** | 0.8020 |
| | 65+ | 17,814 | 21.75% | 60.06% | 61.46% | **84.14%** | [82.98%, 85.30%] | **15.86%** | 0.7856 |
| **Random Forest** | 18–34 | 4,884 | 2.85% | 5.86% | 94.39% | **22.59%** | [15.83%, 29.71%] | **77.41%** | 0.8309 |
| | 35–49 | 9,960 | 8.28% | 18.06% | 84.81% | **54.53%** | [51.13%, 57.94%] | **45.47%** | 0.8173 |
| | 50–64 | 18,078 | 17.51% | 36.19% | 73.35% | **77.91%** | [76.46%, 79.35%] | **22.09%** | 0.8080 |
| | 65+ | 17,814 | 21.75% | 57.95% | 62.30% | **82.67%** | [81.47%, 83.87%] | **17.33%** | 0.7915 |
| **Linear SVM** | 18–34 | 4,884 | 2.85% | 0.25% | 97.17% | **2.91%** | [0.72%, 6.47%] | **97.09%** | 0.8285 |
| | 35–49 | 9,960 | 8.28% | 1.48% | 91.87% | **9.21%** | [7.27%, 11.27%] | **90.79%** | 0.8090 |
| | 50–64 | 18,078 | 17.51% | 5.09% | 83.16% | **16.83%** | [15.53%, 18.15%] | **83.17%** | 0.8018 |
| | 65+ | 17,814 | 21.75% | 9.42% | 79.52% | **22.70%** | [21.38%, 24.03%] | **77.30%** | 0.7854 |
| **XGBoost** | 18–34 | 4,884 | 2.85% | 6.47% | 94.04% | **21.16%** | [14.49%, 28.26%] | **78.84%** | 0.8354 |
| | 35–49 | 9,960 | 8.28% | 19.34% | 83.99% | **55.14%** | [51.74%, 58.54%] | **44.86%** | 0.8202 |
| | 50–64 | 18,078 | 17.51% | 38.64% | 71.74% | **80.22%** | [78.83%, 81.60%] | **19.78%** | 0.8107 |
| | 65+ | 17,814 | 21.75% | 62.20% | 60.10% | **85.72%** | [84.61%, 86.82%] | **14.28%** | 0.7946 |

### Table 2: Formal Fairlearn Disparity Summary
| Model | Sensitive Attribute | Demographic Parity Diff (DPD) | Equal Opportunity Diff (EOD) | Equalized Odds Diff (EODD) | False Negative Rate Diff (FNRD) | Disparity Direction |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Sex (M vs F) | 0.0649 | **0.0019** | 0.0645 | **0.0019** | Parity ($EOD < 0.2\%$) |
| **Random Forest** | Sex (M vs F) | 0.0383 | 0.0251 | 0.0379 | 0.0251 | Slightly higher recall in Female |
| **Linear SVM** | Sex (M vs F) | 0.0072 | 0.0117 | 0.0117 | 0.0117 | Parity via universal negative pred |
| **XGBoost** | Sex (M vs F) | 0.0641 | **0.0030** | 0.0641 | **0.0030** | Parity ($EOD = 0.3\%$) |
| **Logistic Regression** | Age Groups | 0.5263 | **0.5130** | 0.5130 | **0.5130** | High Senior Recall, Low Young Recall |
| **Random Forest** | Age Groups | 0.5209 | 0.6008 | 0.6008 | 0.6008 | High Senior Recall, Low Young Recall |
| **Linear SVM** | Age Groups | 0.0912 | 0.1981 | 0.1981 | 0.1981 | High Young FNR ($97.1\%$), High Senior FNR ($77.3\%$) |
| **XGBoost** | Age Groups | 0.5573 | 0.6450 | 0.6450 | 0.6450 | High Senior Recall, Low Young Recall |

---

## 8. Robustness & Stress-Testing Evaluation

### 8.1 Experiment 1: Missing Data Robustness (MCAR)
Missingness was introduced into test inputs at rates of $\eta \in \{0.05, 0.10, 0.20\}$ using a reproducible mask. Missing values were resolved via the training-fitted imputer pipeline (`SimpleImputer`).

| Model | Baseline AUC ($\eta=0$) | $\eta=0.05$ AUC | $\eta=0.10$ AUC | $\eta=0.20$ AUC | Absolute Drop at 20% | Relative Drop (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8176 | 0.8115 | 0.8049 | 0.7911 | **0.0264** | **3.23%** |
| **Random Forest** | 0.8220 | 0.8155 | 0.8085 | 0.7940 | 0.0280 | 3.41% |
| **Linear SVM** | 0.8174 | 0.8112 | 0.8045 | 0.7907 | 0.0267 | 3.26% |
| **XGBoost** | 0.8247 | 0.8184 | 0.8118 | 0.7975 | 0.0273 | 3.30% |

*Finding:* Imputation handles missingness gracefully across all models ($<3.5\%$ relative drop). Logistic Regression demonstrated the highest missingness resilience.

### 8.2 Experiment 2: Feature Corruption / Noise Robustness
Input features were perturbed at rates of $\rho \in \{0.05, 0.10, 0.20\}$ preserving biological domains:
- Binary flags: Stochastically bit-flipped ($x_{new} = 1 - x$).
- Ordinal features: Perturbed by $\pm 1$ adjacent category step and clamped to domain limits.
- Numerical features: Perturbed with zero-mean proportional Gaussian jitter ($\sigma = 0.25 \sigma_{col}$).
- Age, Sex, and Target were held strictly uncorrupted.

| Model | Baseline AUC ($\rho=0$) | $\rho=0.05$ AUC | $\rho=0.10$ AUC | $\rho=0.20$ AUC | Absolute Drop at 20% | Relative Drop (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8176 | 0.8062 | 0.7951 | 0.7739 | 0.0436 | 5.33% |
| **Random Forest** | 0.8220 | 0.8115 | 0.8010 | 0.7817 | **0.0404** | **4.91%** |
| **Linear SVM** | 0.8174 | 0.8064 | 0.7957 | 0.7755 | 0.0419 | 5.12% |
| **XGBoost** | 0.8247 | 0.8123 | 0.8000 | 0.7778 | 0.0469 | 5.69% |

*Finding:* Random Forest achieved the highest noise resilience ($\Delta \text{AUC} = 0.0404$). In contrast, XGBoost's sequential boosting trees showed slightly higher sensitivity to feature corruption ($\Delta \text{AUC} = 0.0469$).

### 8.3 Experiment 3: Clinical Covariate Shift
Models trained on the general population were evaluated across distinct clinical risk strata:
1. **Low-Metabolic Risk Subpopulation:** $\text{BMI} < 25.0 \text{ kg/m}^2 \text{ and } \text{HighBP} = 0$ ($N = 10,750$, prevalence $3.71\%$).
2. **High-Metabolic Risk Subpopulation:** $\text{BMI} \ge 30.0 \text{ kg/m}^2 \text{ and } \text{HighBP} = 1$ ($N = 11,460$, prevalence $32.22\%$).

Kolmogorov-Smirnov two-sample tests confirmed severe feature divergence ($KS > 0.40, p < 10^{-100}$ for `BMI`, `HighBP`, `HighChol`, and `GenHlth`).
- Under high-risk covariate shift, all models experienced a predictable compression in ROC-AUC ($\Delta \text{AUC} \approx 0.106 - 0.112$), but **XGBoost maintained the highest shifted discriminative ability** ($\text{ROC-AUC} = 0.7188$).

---

## 9. Statistical Validation & Hypothesis Testing

### 9.1 Multi-Seed Repeated Cross-Validation ($N = 10$ Paired Folds)
Paired Student's $t$-tests and Wilcoxon signed-rank tests evaluated differences in fold ROC-AUC:

| Model Comparison | Mean Difference ($\Delta \text{AUC}$) | 95% Confidence Interval | Paired $t$-Statistic | $t$-test $p$-value | Wilcoxon $W$ | Wilcoxon $p$-value | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost vs. Logistic Regression** | **+0.0074** | [+0.0069, +0.0080] | 30.659 | $2.05 \times 10^{-10}$ | 0.0 | $0.00195$ | **Significant ($p < 0.001$)** |
| **XGBoost vs. Random Forest** | **+0.0029** | [+0.0025, +0.0032] | 18.313 | $1.97 \times 10^{-8}$ | 0.0 | $0.00195$ | **Significant ($p < 0.001$)** |
| **Random Forest vs. Logistic Regression** | **+0.0045** | [+0.0039, +0.0052] | 14.940 | $1.17 \times 10^{-7}$ | 0.0 | $0.00195$ | **Significant ($p < 0.001$)** |
| **SVM vs. Logistic Regression** | -0.0003 | [-0.0004, -0.0002] | -7.448 | $3.90 \times 10^{-5}$ | 0.0 | $0.00195$ | **Significant ($p < 0.001$)** |
| **SVM vs. Random Forest** | -0.0049 | [-0.0055, -0.0042] | -17.727 | $2.62 \times 10^{-8}$ | 0.0 | $0.00195$ | **Significant ($p < 0.001$)** |
| **SVM vs. XGBoost** | -0.0077 | [-0.0083, -0.0072] | -34.363 | $7.39 \times 10^{-11}$ | 0.0 | $0.00195$ | **Significant ($p < 0.001$)** |

### 9.2 McNemar's Prediction Discordance Testing (Test Set, $N = 50,736$)
McNemar's test with Edwards continuity correction evaluated paired classification discordance:

| Model 1 | Model 2 | M1 Correct, M2 Wrong ($b$) | M1 Wrong, M2 Correct ($c$) | $\chi^2$ Statistic | $p$-value | Significant Discordance | Superior Model |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **Logistic Regression** | 1,381 | 1,846 | 66.717 | $3.13 \times 10^{-16}$ | **Yes ($p < 0.001$)** | Logistic Regression (Accuracy) |
| **XGBoost** | **Random Forest** | 965 | 1,329 | 57.441 | $3.48 \times 10^{-14}$ | **Yes ($p < 0.001$)** | Random Forest (Accuracy) |
| **Random Forest** | **Logistic Regression** | 1,587 | 1,688 | 3.053 | $0.0806$ | No ($p > 0.05$) | Inconclusive |
| **SVM** | **Logistic Regression** | 10,534 | 4,559 | 2364.585 | $< 10^{-300}$ | **Yes ($p < 0.001$)** | SVM (Accuracy) |
| **SVM** | **Random Forest** | 10,711 | 4,635 | 2404.902 | $< 10^{-300}$ | **Yes ($p < 0.001$)** | SVM (Accuracy) |
| **SVM** | **XGBoost** | 11,263 | 4,823 | 2577.441 | $< 10^{-300}$ | **Yes ($p < 0.001$)** | SVM (Accuracy) |

*Insight:* While SVM achieves higher raw concordance due to predicting negative on the 84% negative class, XGBoost and Logistic Regression generate significantly superior positive class identification ($p < 10^{-15}$).

---

## 10. Comprehensive Model Synthesis & The Trilemma

```
                      THE CLINICAL AI TRILEMMA
                       Predictive Efficacy
                           (XGBoost)
                             /\
                            /  \
                           /    \
                          /      \
                         /________\
             Demographic            Data-Quality
               Fairness              Robustness
         (Logistic Regression)     (Random Forest)
```

No single model dominates across all three axes:
1. **Best Predictive Efficacy (Recall & ROC-AUC):** **XGBoost Classifier** is the undisputed clinical screening choice ($\text{AUC} = 0.8247, \text{FNR} = 21.50\%$).
2. **Best Demographic Equity (Smallest Disparities):** **Logistic Regression** provides the most balanced life-course equity ($\text{Age EOD} = 0.5130$, identifying $32.84\%$ of young diabetic adults compared to only $21.16\%$ for XGBoost).
3. **Best Input Corruption Resilience:** **Random Forest** provides superior stability under noisy, degraded features ($\text{AUC drop} = 0.0404$).
4. **The Accuracy Trap:** **Support Vector Machine** achieves the highest raw accuracy ($84.81\%$), but is clinically disastrous, failing to detect $80.98\%$ of patients with prediabetes or diabetes.

---

## 11. Threats to Validity & Limitations

1. **Self-Reported Telephonic Survey Bias:** The BRFSS relies on telephonic self-reporting rather than laboratory blood draws (HbA1c / fasting glucose). Undiagnosed diabetes cases in the general population may be labeled as negative.
2. **Cross-Sectional Structure:** The dataset does not track patients longitudinally; our distribution shift experiments reflect subpopulation covariate divergence rather than temporal drift.
3. **Uniform Thresholding Artifact:** Evaluating models at $p = 0.50$ inevitably disadvantages low-prevalence young cohorts. Stratified post-processing thresholds must be applied in deployment.
4. **BRFSS Age Aggregation:** The survey records age in 5-year buckets rather than continuous years, limiting fine-grained threshold fitting.

---

## 12. Conclusion & Recommendations for Clinical AI Deployment

1. **Abandon Accuracy as a Primary Metric:** Clinical diabetes prediction systems must never be procured or deployed based on accuracy alone. Evaluators must prioritize Sensitivity (Recall), False Negative Rate, and ROC-AUC.
2. **Implement Demographic-Stratified Thresholds:** To eliminate the $78\%$ false-negative rate among young adults, clinical decision support systems must utilize age-adjusted decision boundaries.
3. **Ensemble Architecture for Noisy Real-World Inputs:** Where clinical input data quality is uncertain, Random Forest or hybrid XGBoost-Forest architectures offer the best insurance against localized feature corruption.

---

## 13. Citation & Reproducibility Notice
All experiments were implemented in Python 3.13 using scikit-learn, XGBoost, Fairlearn, and SciPy, with fixed random states (`seed=42`). Serialized models, datasets, and scripts are accessible in the project root:
- Execution Runner: `python main.py`
- Research Dashboard: `python main.py --dashboard`
- Unit Test Suite: `pytest tests/ -v`

