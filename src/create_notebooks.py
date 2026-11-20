"""
src/create_notebooks.py
Generates the 5 required research Jupyter Notebooks (.ipynb) using nbformat:
  1. notebooks/01_dataset_exploration.ipynb
  2. notebooks/02_preprocessing.ipynb
  3. notebooks/03_model_training.ipynb
  4. notebooks/04_fairness_evaluation.ipynb
  5. notebooks/05_robustness_evaluation.ipynb
"""

import os

import nbformat as nbf

os.makedirs("notebooks", exist_ok=True)


def create_exploration_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 01. Dataset Exploration and Profiling\n"
            "**Research Project:** Evaluating Robustness and Fairness of Diabetes Risk Prediction Models Across Demographic Groups\n\n"
            "This notebook inspects the CDC BRFSS 2015 cohort, validates schema types, evaluates target distribution, "
            "and generates demographic profiles."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.append('..')\n"
            "import pandas as pd\n"
            "from src.data_loader import generate_dataset_profile, load_dataset, plot_dataset_visualizations\n\n"
            "# Load dataset\n"
            "df = load_dataset('../data/raw/diabetes_012_health_indicators_BRFSS2015.csv')\n"
            "print('Dataset Shape:', df.shape)\n"
            "df.head()"
        ),
        nbf.v4.new_code_cell(
            "# Generate profile & metadata\n"
            "profile = generate_dataset_profile(df, output_dir='../results')\n"
            "profile"
        ),
        nbf.v4.new_code_cell(
            "# Feature Metadata Breakdown\n"
            "meta_df = pd.read_csv('../results/feature_metadata.csv')\n"
            "meta_df"
        ),
        nbf.v4.new_markdown_cell("## Visualizing Class and Demographic Distributions"),
        nbf.v4.new_code_cell(
            "plot_dataset_visualizations(df, output_dir='../figures/dataset_analysis')\n"
            "from IPython.display import Image, display\n"
            "display(Image(filename='../figures/dataset_analysis/diabetes_class_distribution.png'))\n"
            "display(Image(filename='../figures/dataset_analysis/prevalence_by_age.png'))\n"
            "display(Image(filename='../figures/dataset_analysis/prevalence_by_sex.png'))"
        ),
    ]
    with open("notebooks/01_dataset_exploration.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/01_dataset_exploration.ipynb")


def create_preprocessing_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 02. Data Preprocessing & Leakage Prevention Pipeline\n"
            "This notebook builds the Scikit-learn preprocessing pipeline, validates the 80:20 stratified split, "
            "and confirms that imputers and scalers fit strictly on the training partition."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.append('..')\n"
            "from src.preprocessing import prepare_data\n\n"
            "data = prepare_data(filepath='../data/raw/diabetes_012_health_indicators_BRFSS2015.csv', output_dir='../data/processed')\n"
            "print('X_train shape:', data['X_train'].shape)\n"
            "print('X_test shape: ', data['X_test'].shape)\n"
            "print('Positive class prevalence in train:', data['y_train'].mean())\n"
            "print('Positive class prevalence in test: ', data['y_test'].mean())"
        ),
        nbf.v4.new_code_cell("# Inspect preprocessor structure\ndata['preprocessor']"),
        nbf.v4.new_markdown_cell("## Preprocessing Audit Report"),
        nbf.v4.new_code_cell(
            "with open('../reports/preprocessing_summary.txt') as f:\n"
            "    print(f.read())"
        ),
    ]
    with open("notebooks/02_preprocessing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/02_preprocessing.ipynb")


def create_model_training_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 03. Machine Learning Model Training & Evaluation\n"
            "Trains and cross-validates Logistic Regression, Random Forest, Linear SVM, and XGBoost.\n"
            "Evaluates performance on the held-out test partition across all 9 metrics."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.append('..')\n"
            "import pandas as pd\n"
            "from src.evaluate_models import evaluate_on_test_set, load_all_models\n"
            "from src.preprocessing import prepare_data\n\n"
            "data = prepare_data(filepath='../data/raw/diabetes_012_health_indicators_BRFSS2015.csv')\n"
            "models = load_all_models('../models')\n"
            "perf_df, y_pred_dict, y_prob_dict = evaluate_on_test_set(models, data['X_test'], data['y_test'])\n"
            "perf_df"
        ),
        nbf.v4.new_markdown_cell(
            "## Cross-Validation & Test Set Evaluation Visualizations"
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image(filename='../figures/model_performance/roc_curves.png'))\n"
            "display(Image(filename='../figures/model_performance/precision_recall_curves.png'))\n"
            "display(Image(filename='../figures/model_performance/confusion_matrices.png'))"
        ),
        nbf.v4.new_code_cell(
            "# Paired Statistical Testing\n"
            "stat_df = pd.read_csv('../results/statistical_results.csv')\n"
            "stat_df[stat_df['Metric'] == 'ROC_AUC']"
        ),
    ]
    with open("notebooks/03_model_training.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/03_model_training.ipynb")


def create_fairness_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 04. Demographic Fairness Evaluation\n"
            "Audits model behavior across Age (18-34, 35-49, 50-64, 65+) and Sex (Female, Male) groups using Fairlearn."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.append('..')\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "fairness_df = pd.read_csv('../results/fairness_metrics.csv')\n"
            "fairness_df"
        ),
        nbf.v4.new_markdown_cell(
            "## Age-Based Disparities (Equal Opportunity & Recall)"
        ),
        nbf.v4.new_code_cell(
            "age_df = pd.read_csv('../results/age_fairness.csv')\n"
            "display(Image(filename='../figures/fairness/age_group_recall.png'))\n"
            "display(Image(filename='../figures/fairness/age_group_fnr.png'))\n"
            "age_df[['Model', 'Age_Group', 'Sample_Size', 'Recall', 'False_Negative_Rate', 'Recall_CI_Lower', 'Recall_CI_Upper']]"
        ),
        nbf.v4.new_markdown_cell("## Sex-Based Disparities"),
        nbf.v4.new_code_cell(
            "sex_df = pd.read_csv('../results/sex_fairness.csv')\n"
            "display(Image(filename='../figures/fairness/sex_group_recall.png'))\n"
            "display(Image(filename='../figures/fairness/demographic_parity_comparison.png'))\n"
            "sex_df[['Model', 'Sex', 'Sample_Size', 'Recall', 'False_Negative_Rate']]"
        ),
    ]
    with open("notebooks/04_fairness_evaluation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/04_fairness_evaluation.ipynb")


def create_robustness_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 05. Model Robustness & Stress-Testing\n"
            "Evaluates model resilience under controlled Missing Data (MCAR), Input Feature Noise, "
            "and Subgroup Distribution Shift."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.append('..')\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "rob_df = pd.read_csv('../results/robustness_results.csv')\n"
            "display(Image(filename='../figures/robustness/missing_data_degradation.png'))\n"
            "display(Image(filename='../figures/robustness/noise_data_degradation.png'))\n"
            "display(Image(filename='../figures/robustness/model_robustness_comparison.png'))"
        ),
        nbf.v4.new_markdown_cell("## Subgroup-Based Covariate Shift"),
        nbf.v4.new_code_cell(
            "shift_df = pd.read_csv('../results/distribution_shift_results.csv')\n"
            "display(Image(filename='../figures/robustness/distribution_shift_degradation.png'))\n"
            "shift_df"
        ),
        nbf.v4.new_markdown_cell("## Consolidated Research Synthesis"),
        nbf.v4.new_code_cell(
            "comb_df = pd.read_csv('../results/combined_comparison.csv')\ncomb_df"
        ),
    ]
    with open("notebooks/05_robustness_evaluation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/05_robustness_evaluation.ipynb")


if __name__ == "__main__":
    create_exploration_notebook()
    create_preprocessing_notebook()
    create_model_training_notebook()
    create_fairness_notebook()
    create_robustness_notebook()
    print("All 5 Jupyter notebooks generated successfully.")
